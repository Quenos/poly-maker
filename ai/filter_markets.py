import logging
import os
import sys
import re
import concurrent.futures
import json
import requests

import pandas as pd

# Ensure repository root on sys.path for direct script execution
try:
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)
except Exception:
    pass

# Local project imports
from data_updater.trading_utils import get_clob_client  # type: ignore
from data_updater.find_markets import get_all_markets as _get_all_markets  # type: ignore

from openai import OpenAI  # type: ignore
from wallet_pnl import get_best_price  # type: ignore


logger = logging.getLogger(__name__)


def get_all_markets() -> pd.DataFrame:
    """
    Fetch and return all markets as a pandas DataFrame using the existing
    implementation in data_updater.find_markets.

    Returns:
        pd.DataFrame: All markets; empty DataFrame on failure.
    """
    try:
        if get_clob_client is None or _get_all_markets is None:
            logger.error("Project imports unavailable (data_updater.*); returning empty DataFrame")
            return pd.DataFrame()

        client = get_clob_client()
        if client is None:
            logger.error("CLOB client is not initialized; returning empty DataFrame")
            return pd.DataFrame()
        df = _get_all_markets(client)
        if df is None:
            logger.warning("get_all_markets returned None; converting to empty DataFrame")
            return pd.DataFrame()
        # Filter to only active and not closed markets when columns exist
        try:
            if "active" in df.columns:
                df = df[df["active"] == True]  # noqa: E712
            if "closed" in df.columns:
                df = df[df["closed"] == False]  # noqa: E712
            df = df.reset_index(drop=True)
        except Exception:
            logger.debug("Failed to apply active/closed filters; returning unfiltered DataFrame", exc_info=True)
        return df
    except Exception:
        logger.exception("Failed to fetch all markets; returning empty DataFrame")
        return pd.DataFrame()


def get_market_name_token_pairs() -> pd.DataFrame:
    """
    Return a DataFrame with only market name (question) and the token pair ids.

    Columns: question, token1, token2
    """
    df = get_all_markets()
    if df is None or df.empty:
        return pd.DataFrame(columns=["question", "token1", "token2"])  # stable schema

    records = []
    try:
        for _, row in df.iterrows():
            try:
                q = str(row.get("question", ""))
            except Exception:
                q = ""
            t1 = ""
            t2 = ""
            try:
                tokens = row.get("tokens", [])
                if isinstance(tokens, list):
                    if len(tokens) > 0 and isinstance(tokens[0], dict):
                        t1 = str(tokens[0].get("token_id", ""))
                    if len(tokens) > 1 and isinstance(tokens[1], dict):
                        t2 = str(tokens[1].get("token_id", ""))
            except Exception:
                t1 = t2 = ""
            records.append({"question": q, "token1": t1, "token2": t2})
    except Exception:
        logger.exception("Error extracting name/token pairs; returning partial results if any")

    out = pd.DataFrame.from_records(records, columns=["question", "token1", "token2"]) if records else pd.DataFrame(columns=["question", "token1", "token2"])
    return out

__all__ = ["get_all_markets", "get_market_name_token_pairs", "remove_markets_to_avoid"]


def _build_market_filter_prompts(market_titles: list[str]) -> tuple[str, str]:
    """Construct system and user prompts for market filtering."""
    system_prompt = (
        "You are an expert prediction market analyst. You will be given a list of Polymarket markets. "
        "For each market, decide if it is ELIGIBLE for automated market making or should be AVOIDED based on risks like "
        "illiquidity, insider information, manipulability, ambiguity, or extreme horizon. Be strict in filtering.\n\n"
        "Rules for Avoidance:\n"
        "A market should be marked AVOID if:\n"
        "\t•\tLow Liquidity / Niche topic\n"
        "\t•\tInsider-prone or private knowledge needed\n"
        "\t•\tVery short horizon (hours / micro-events)\n"
        "\t•\tAmbiguous or disputable resolution criteria\n"
        "\t•\tMeme/celebrity/hype-driven\n"
        "\t•\tMulti-outcome / too complex\n"
        "\t•\tLong-tail horizon (>18 months)\n"
        "\t•\tEasily manipulable (cheap to influence outcome)\n\n"
        "Output format (JSON array):\n\n"
        "[\n  {\n    \"market\": \"<market title>\",\n    \"decision\": \"AVOID\" | \"ELIGIBLE\",\n    \"reason\": \"short explanation\"\n  },\n  ...\n]\n"
    )

    lines = ["Markets:"]
    for title in market_titles:
        # Provide plain quoted titles without indices or bullets
        lines.append(f"\t\"{title}\"")
    user_prompt = "\n".join(lines)
    logger.info("System prompt: %s", system_prompt)
    logger.info("User prompt: %s", user_prompt)
    return system_prompt, user_prompt


def _normalize_market_title(name: str) -> str:
    """Normalize market title by stripping indices/bullets and surrounding quotes."""
    if name is None:
        return ""
    s = str(name).strip()
    # Drop surrounding quotes
    if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
        s = s[1:-1].strip()
    # Remove leading list markers like '1.', '1)', '-', '•'
    s = re.sub(r"^\s*(?:[-•]+\s*|\d+\s*[\.)-]\s*)", "", s)
    return s.strip()


def remove_markets_to_avoid(pairs_df: pd.DataFrame | None = None,
                            chunk_size: int = 100,
                            model: str = "gpt-4o-mini",
                            max_workers: int = 4) -> pd.DataFrame:
    """
    Call OpenAI with a strict filtering prompt and remove markets labeled AVOID.

    Args:
        pairs_df: Optional DataFrame with columns [question, token1, token2]. If None, it is generated.
        model: OpenAI model to use. Defaults to "gpt-4o-mini".

    Returns:
        DataFrame filtered to only ELIGIBLE markets, preserving columns question, token1, token2.
    """
    try:
        if pairs_df is None:
            pairs_df = get_market_name_token_pairs()
        if pairs_df is None or pairs_df.empty:
            logger.info("No markets to filter; returning empty DataFrame")
            return pd.DataFrame(columns=["question", "token1", "token2"])  # stable schema

        titles_series = pairs_df.get("question")
        if titles_series is None:
            logger.error("Input DataFrame missing 'question' column; returning unchanged")
            return pairs_df
        titles = [str(t).strip() for t in titles_series.tolist() if str(t).strip()]
        if not titles:
            logger.info("No non-empty market titles found; returning unchanged DataFrame")
            return pairs_df

        # Helper to process a single chunk; returns set of ELIGIBLE titles
        def process_chunk(chunk_index: int, chunk_titles: list[str]) -> set[str]:
            local_eligible: set[str] = set()
            system_prompt, user_prompt = _build_market_filter_prompts(chunk_titles)
            try:
                api_key = os.getenv("OPENAI_API_KEY", "").strip()
                client_local = OpenAI(api_key=api_key) if api_key else OpenAI()
                resp = client_local.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.2,
                )
                content = (resp.choices[0].message.content or "").strip()
            except Exception:
                logger.exception("OpenAI request failed for chunk %s-%s; skipping this chunk", chunk_index, chunk_index + len(chunk_titles))
                return local_eligible

            decisions: list[dict] = []
            try:
                decisions = json.loads(content)
            except Exception:
                try:
                    start = content.find("[")
                    end = content.rfind("]")
                    if start != -1 and end != -1 and end > start:
                        decisions = json.loads(content[start:end + 1])
                except Exception:
                    logger.exception("Failed to parse OpenAI JSON for chunk %s-%s; skipping", chunk_index, chunk_index + len(chunk_titles))
                    return local_eligible

            if not isinstance(decisions, list):
                logger.error("OpenAI response not a JSON array for chunk %s-%s; skipping", chunk_index, chunk_index + len(chunk_titles))
                return local_eligible

            for item in decisions:
                try:
                    name = _normalize_market_title(item.get("market", ""))
                    decision = str(item.get("decision", "")).upper()
                    if name and decision == "ELIGIBLE":
                        local_eligible.add(name)
                except Exception:
                    continue
            return local_eligible

        # Build chunks and process in parallel
        chunks: list[list[str]] = [titles[i:i + chunk_size] for i in range(0, len(titles), chunk_size)]
        eligible_titles: set[str] = set()
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_chunk, idx * chunk_size, chunk) for idx, chunk in enumerate(chunks)]
            for future in concurrent.futures.as_completed(futures):
                try:
                    res = future.result()
                    if res:
                        eligible_titles.update(res)
                except Exception:
                    logger.exception("Parallel worker failed; continuing")

        if not eligible_titles:
            logger.info("No titles marked ELIGIBLE; returning empty result DataFrame")
            return pd.DataFrame(columns=["question", "token1", "token2"])  # empty schema

        # Normalize input titles for robust matching
        normalized_questions = pairs_df["question"].astype(str).map(_normalize_market_title)
        filtered = pairs_df[normalized_questions.isin(eligible_titles)].copy()
        filtered = filtered.reset_index(drop=True)
        logger.info("Filtered markets: kept %d of %d", len(filtered.index), len(pairs_df.index))
        return filtered
    except Exception:
        logger.exception("Unexpected error during market filtering; returning unchanged DataFrame")
        return pairs_df if pairs_df is not None else pd.DataFrame(columns=["question", "token1", "token2"])


EDGE_SYSTEM_PROMPT = (
    "You are an Edge Estimation AI working for an automated market maker on Polymarket.\n\n"
    "You will receive a list of markets, each defined as:\n{\n  \"title\": \"<market title>\",\n  \"yes_price\": <float 0..1>,\n  \"no_price\": <float 0..1>\n}\n\n"
    "Your job is to:\n"
    "1. Parse each market.\n"
    "2. Classify the market into one of the following categories:\n   - politics (elections, leaders, laws, government decisions)\n   - sports (competitions, players, injuries, matches)\n   - finance/economy (central banks, jobs report, inflation, markets)\n   - crypto/tech (tokens, exchange listings, AI releases)\n   - awards/entertainment (Oscars, Emmys, Nobel Prize, pop culture)\n   - other (anything else, but you must still attempt analysis)\n\n"
    "3. Based on the category, you MUST generate search queries using the query templates provided below.\n   You may call Tavily multiple times per market. Always run 2–4 relevant queries per market.\n\n"
    "4. Extract structured evidence from the search results:\n   - Latest factual updates (within 30 days preferred, older if foundational).\n   - Official announcements, filings, schedules, rosters, injury reports.\n   - Polls, betting odds, expert consensus.\n   - News events or major developments.\n   - Social signals only if no better evidence is available.\n\n"
    "5. From extracted evidence:\n   - Compute market implied probability P_mkt = yes_price.\n   - Estimate external consensus probability P_consensus.\n   - Compute edge = P_consensus – P_mkt.\n   - Assign confidence 0–1 based on number and quality of sources.\n\n"
    "6. Return an Edge Decision Contract for each market in strict JSON format:\n{\n  \"title\": \"<market title>\",\n  \"p_mkt\": <float>,\n  \"p_consensus\": <float|null>,\n  \"edge\": <float|null>,\n  \"confidence\": <float 0..1>,\n  \"sources\": [\n    {\n      \"source\": \"<domain>\",\n      \"type\": \"official|news|bookmaker|poll|social\",\n      \"date\": \"YYYY-MM-DD\",\n      \"summary\": \"<short summary ≤40 words>\",\n      \"probability\": <0..1|null>,\n      \"how_prob_was_derived\": \"odds|poll|implied|none\",\n      \"relevance\": <float 0..1>\n    }\n  ]\n}\n\n"
    "Do not output anything other than the final JSON array.\n\n---\n\n"
    "### QUERY TEMPLATES\n\n"
    "Politics\n- \"latest polls {CANDIDATE} {ELECTION}\"\n- \"approval rating {CANDIDATE} site:538.com\"\n- \"betting odds {ELECTION} site:oddschecker.com\"\n- \"official election commission schedule {COUNTRY}\"\n\n"
    "Sports\n- \"latest odds {LEAGUE} {EVENT}\"\n- \"injury report {TEAM}\"\n- \"official {LEAGUE} standings\"\n- \"lineups {TEAM} vs {OPPONENT} {YEAR}\"\n\n"
    "Finance / Economy\n- \"ECB meeting schedule September 2025 site:ecb.europa.eu\"\n- \"US jobs report August 2025 consensus forecast\"\n- \"FOMC minutes September 2025 site:federalreserve.gov\"\n- \"market odds {EVENT} site:oddschecker.com\"\n\n"
    "Crypto / Tech\n- \"Coinbase listing {TOKEN} site:blog.coinbase.com\"\n- \"latest news {TOKEN} 2025\"\n- \"AI release {MODEL} official site\"\n- \"crypto odds {EVENT} site:oddschecker.com\"\n\n"
    "Awards / Entertainment\n- \"Emmy 2025 nominees site:emmys.com\"\n- \"Oscar shortlist 2025 official site:oscar.org\"\n- \"betting odds {AWARD} 2025 site:oddschecker.com\"\n- \"recent reviews {TITLE} site:rottentomatoes.com\"\n\n"
    "Other\n- \"latest news {EVENT}\"\n- \"official statement {EVENT}\"\n- \"betting odds {EVENT} site:oddschecker.com\"\n\n---\n"
)


def _build_edge_user_prompt(events: list[dict]) -> str:
    parts = [
        "EVENTS:\n",
        json.dumps(events, ensure_ascii=False),
        "\n\nTASKS:\n",
        "For each event above, follow the workflow in the system prompt:\n",
        "- classify category,\n",
        "- generate 2-4 Tavily queries based on templates,\n",
        "- collect evidence,\n",
        "- compute P_consensus and edge,\n",
        "- return a JSON array of Edge Decision Contracts.\n",
    ]
    return "".join(parts)


def _tavily_search(query: str) -> dict:
    """Run a Tavily search via their API using TAVILY_API_KEY; return JSON response or empty dict."""
    try:
        api_key = os.getenv("TAVILY_API_KEY", "").strip()
        if not api_key:
            logger.warning("Tavily API key missing; skipping query: %s", query)
            return {}
        logger.info("Tavily query: %s", query)
        resp = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": api_key, "query": query, "max_results": 5, "search_depth": "advanced"},
            timeout=30,
        )
        if not resp.ok:
            logger.info(
                "Tavily response not ok (%s) for query: %s", resp.status_code, query
            )
            return {}
        data = resp.json()
        try:
            count = 0
            if isinstance(data, dict) and "results" in data:
                results = data.get("results", [])
                count = len(results) if isinstance(results, list) else 0
            elif isinstance(data, list):
                count = len(data)
            logger.info("Tavily returned %s results for query: %s", count, query)
        except Exception:
            pass
        return data
    except Exception:
        logger.exception("Tavily search failed for query: %s", query)
        return {}


def _extract_prices_for_pairs(pairs_df: pd.DataFrame) -> list[dict]:
    events: list[dict] = []
    for _, r in pairs_df.iterrows():
        title = str(r.get("question", "")).strip()
        t1 = str(r.get("token1", "")).strip()
        t2 = str(r.get("token2", "")).strip()
        if not title or not t1 or not t2:
            continue
        try:
            # Get market prices via existing function
            yes_price = float(get_best_price(t1, "sell"))  # liquidate long -> sell
            no_price = float(get_best_price(t2, "sell"))
            # Best effort normalization 0..1
            if yes_price > 1:
                yes_price = yes_price / 100.0
            if no_price > 1:
                no_price = no_price / 100.0
        except Exception:
            yes_price = 0.0
            no_price = 0.0
        events.append({"title": title, "yes_price": yes_price, "no_price": no_price})
    return events


def compute_edge_for_markets(pairs_df: pd.DataFrame,
                             chunk_size: int = 50,
                             model: str = "gpt-4o-mini",
                             max_workers: int = 4) -> list[dict]:
    """
    For cleaned market pairs, fetch prices, then ask OpenAI to estimate edge.
    If the AI asks for a Tavily search, run it and feed the results back, chunked and parallelized.
    Returns a list of Edge Decision Contracts (dicts).
    """
    try:
        events = _extract_prices_for_pairs(pairs_df)
        if not events:
            return []

        def _extract_queries_from_ai_response(text: str) -> list[str]:
            """Extract probable Tavily queries from AI free-text or JSON-like content."""
            queries: list[str] = []
            # Try JSON parse first and look for common keys
            try:
                obj = json.loads(text)
                # If obj is a dict with queries-like fields
                if isinstance(obj, dict):
                    for key in ("queries", "tavily_queries", "search_queries"):
                        val = obj.get(key)
                        if isinstance(val, list):
                            for q in val:
                                if isinstance(q, str) and len(q.strip()) >= 4:
                                    queries.append(q.strip())
                # If obj is a list of dicts each possibly containing queries
                if not queries and isinstance(obj, list):
                    for item in obj:
                        if isinstance(item, dict):
                            for key in ("queries", "tavily_queries", "search_queries"):
                                val = item.get(key)
                                if isinstance(val, list):
                                    for q in val:
                                        if isinstance(q, str) and len(q.strip()) >= 4:
                                            queries.append(q.strip())
            except Exception:
                pass

            # Fallback: quoted strings
            if not queries:
                for q in re.findall(r"\"([^\"]{4,})\"", text):
                    qs = q.strip()
                    # Heuristic: looks like a search phrase (spaces or site: operator)
                    if " " in qs or "site:" in qs:
                        queries.append(qs)

            # Fallback: bullet lines with quotes
            if not queries:
                for line in text.splitlines():
                    m = re.search(r"-\s*\"([^\"]{4,})\"", line)
                    if m:
                        queries.append(m.group(1).strip())

            # Deduplicate, keep order
            seen: set[str] = set()
            uniq: list[str] = []
            for q in queries:
                if q not in seen:
                    seen.add(q)
                    uniq.append(q)
            return uniq

        def process_chunk(chunk_index: int, chunk_events: list[dict]) -> list[dict]:
            system_prompt = EDGE_SYSTEM_PROMPT
            user_prompt = _build_edge_user_prompt(chunk_events)
            api_key = os.getenv("OPENAI_API_KEY", "").strip()
            client_local = OpenAI(api_key=api_key) if api_key else OpenAI()
            # First ask
            try:
                resp = client_local.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.2,
                )
                content = (resp.choices[0].message.content or "").strip()
                logger.info("Edge AI initial response (chunk %s): %s", chunk_index, content)
            except Exception:
                logger.exception("Edge OpenAI request failed for chunk %s; skipping", chunk_index)
                return []

            # If the model asked for more Tavily queries, try to detect and run them
            lower = content.lower()
            possible_queries = _extract_queries_from_ai_response(content) if ("query" in lower or "search" in lower or "tavily" in lower) else []
            if possible_queries:
                logger.info("AI requested Tavily queries (chunk %s): %s", chunk_index, possible_queries)
                tavily_results = []
                for q in possible_queries[:8]:
                    res = _tavily_search(q)
                    if res:
                        tavily_results.append({"query": q, "result": res})
                # Send back a follow-up with results
                try:
                    follow = client_local.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                            {"role": "assistant", "content": content},
                            {"role": "user", "content": "Here are the Tavily results you requested: " + json.dumps(tavily_results) + "\nPlease return only the final JSON array as specified."},
                        ],
                        temperature=0.2,
                    )
                    content = (follow.choices[0].message.content or "").strip()
                    logger.info("Edge AI follow-up response (chunk %s): %s", chunk_index, content)
                except Exception:
                    logger.exception("Edge follow-up OpenAI request failed for chunk %s; returning initial content", chunk_index)

            # Parse final JSON array
            try:
                result = json.loads(content)
                if isinstance(result, list):
                    return result
                # try extract array
                start = content.find("[")
                end = content.rfind("]")
                if start != -1 and end != -1 and end > start:
                    result = json.loads(content[start:end + 1])
                    return result if isinstance(result, list) else []
            except Exception:
                logger.debug("Failed to parse edge JSON for chunk %s", chunk_index)
                return []

            return []

        # Run chunks in parallel
        outputs: list[dict] = []
        chunks = [events[i:i + chunk_size] for i in range(0, len(events), chunk_size)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_chunk, idx, chunk) for idx, chunk in enumerate(chunks)]
            for future in concurrent.futures.as_completed(futures):
                try:
                    res = future.result()
                    if res:
                        outputs.extend(res)
                except Exception:
                    logger.exception("Edge parallel worker failed; continuing")
        return outputs
    except Exception:
        logger.exception("Unexpected error during compute_edge_for_markets")
        return []


if __name__ == "__main__":
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    df = get_all_markets()
    try:
        logger.info("Fetched %d markets", len(df.index))
    except Exception:
        logger.info("Fetched markets frame with shape: %s", getattr(df, "shape", None))
    # Also log a preview of the name/token pair DataFrame
    try:
        pairs = get_market_name_token_pairs()
        logger.info("Pair preview: %s", pairs.head())
        clean_pairs = remove_markets_to_avoid(pairs)
        logger.info("Cleaned pairs length: %s", len(clean_pairs))
        logger.info("Cleaned pairs: %s", clean_pairs.head())
        edge_results = compute_edge_for_markets(clean_pairs)
        logger.info("Edge results length: %s", len(edge_results))
        logger.info("Edge results: %s", edge_results)
    except Exception:
        logger.debug("Unable to compute name/token pair DataFrame", exc_info=True)