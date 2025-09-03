import logging
import argparse
import datetime as dt
import os
import sys
import re
import concurrent.futures
import json
import pandas as pd
from dotenv import load_dotenv  # type: ignore

# Ensure repository root on sys.path for direct script execution
try:
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)
except Exception:
    pass

# Load environment variables early for CLI usage
try:
    load_dotenv()
except Exception:
    pass

# Local project imports
from data_updater.trading_utils import get_clob_client  # type: ignore
from data_updater.find_markets import get_all_markets as _get_all_markets  # type: ignore
from data_updater.find_markets import get_order_book_with_retry  # type: ignore

from openai import OpenAI  # type: ignore
from ai.search_agent import tavily_search_impl, tavily_get_impl  # type: ignore


logger = logging.getLogger(__name__)
try:
    CLEANED_PAIRS_PATH_DEFAULT = os.path.join(ROOT_DIR, "data", "cleaned_pairs.csv")
except Exception:
    CLEANED_PAIRS_PATH_DEFAULT = "data/cleaned_pairs.csv"

try:
    LOG_DIR_DEFAULT = os.path.join(ROOT_DIR, "logs")
except Exception:
    LOG_DIR_DEFAULT = "logs"
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
    Return a DataFrame with only market name (question), optional rules, and the token pair ids.

    Columns: question, token1, token2, rules
    """
    df = get_all_markets()
    if df is None or df.empty:
        return pd.DataFrame(columns=["question", "token1", "token2", "rules"])  # stable schema

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
            try:
                rules_text = _extract_rules_from_market_row(row)
            except Exception:
                rules_text = ""
            records.append({"question": q, "token1": t1, "token2": t2, "rules": rules_text})
    except Exception:
        logger.exception("Error extracting name/token pairs; returning partial results if any")

    out = pd.DataFrame.from_records(records, columns=["question", "token1", "token2", "rules"]) if records else pd.DataFrame(columns=["question", "token1", "token2", "rules"]) 
    return out

__all__ = ["get_all_markets", "get_market_name_token_pairs", "remove_markets_to_avoid"]


def _build_market_filter_prompts(market_titles: list[str]) -> tuple[str, str]:
    """Construct system and user prompts for market filtering."""
    system_prompt = """You are an expert prediction-market analyst. For each market, decide if it is ELIGIBLE for automated market making or should be AVOIDED. 
Liquidity is checked separately, do not consider it here.

Key rules:
- SPORTS: ELIGIBLE if major league/tournament winner or match result. Avoid props or niche leagues.
- ELECTIONS: ELIGIBLE if national/statewide/general elections or balance of power. Avoid only if ambiguous (no clear candidate/ballot definition).
- CRYPTO/FX/INDEX: ELIGIBLE if price thresholds/ranges with clear reference index or multi-exchange benchmark. Avoid only if source is undefined.
- GEOPOLITICS/NEWS: ELIGIBLE if explicit resolution source (UN, EU Official Journal, govt press release). Avoid if terms ambiguous.
- ENTERTAINMENT: ELIGIBLE if box office/grossing with named source. Avoid meme/award/vague criteria.
- HORIZON: Long horizon (but less than 1 year) is fine unless combined with ambiguity.

Output JSON array:
[
  {
    "market": "<title>",
    "decision": "ELIGIBLE" | "AVOID",
    "reason": "tags like [sports, election, crypto, clear-source, ambiguous, prop, vague]",
    "confidence": 0.0-1.0
  }
]"""

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


def _extract_rules_from_market_row(row: pd.Series) -> str:
    """Extract resolution rules/criteria text from a markets row if present.

    Tries multiple common field names and concatenates distinct, non-empty values.
    Returns a compact single string. Fallback is an empty string.
    """
    try:
        candidate_keys = [
            "rules",
            "description",
            "resolution_criteria",
            "resolutionCriteria",
            "resolution_description",
            "resolve_description",
            "criteria",
            "question_details",
            "extra_info",
        ]
        parts: list[str] = []
        seen: set[str] = set()
        for key in candidate_keys:
            try:
                val = row.get(key)
            except Exception:
                val = None
            if val is None:
                continue
            text = str(val).strip()
            if not text:
                continue
            if text not in seen:
                seen.add(text)
                parts.append(text)
        # Join with clear separator if multiple sources
        return " \n\n".join(parts) if parts else ""
    except Exception:
        return ""


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
            return pd.DataFrame(columns=["question", "token1", "token2", "rules"])  # stable schema

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
                    temperature=0.0,
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
            return pd.DataFrame(columns=["question", "token1", "token2", "rules"])  # empty schema

        # Normalize input titles for robust matching
        normalized_questions = pairs_df["question"].astype(str).map(_normalize_market_title)
        filtered = pairs_df[normalized_questions.isin(eligible_titles)].copy()
        filtered = filtered.reset_index(drop=True)
        logger.info("Filtered markets: kept %d of %d", len(filtered.index), len(pairs_df.index))
        try:
            _save_cleaned_pairs(filtered)
        except Exception:
            logger.debug("Unable to save cleaned pairs", exc_info=True)
        return filtered
    except Exception:
        logger.exception("Unexpected error during market filtering; returning unchanged DataFrame")
        return pairs_df if pairs_df is not None else pd.DataFrame(columns=["question", "token1", "token2"])


EDGE_SYSTEM_PROMPT = (
    "You are an Odds Estimation AI.\n\n"
    "INPUT: an array of {\"title\": string, \"rules\": string?}. You do NOT receive market prices. Use 'rules' to understand scope/resolution when provided.\n\n"
    "GOAL: For EACH item, you MUST output an object with:\n"
    "{\n"
    "  \"title\": \"<event>\",\n"
    "  \"category\": \"sports|politics|finance/economy|crypto/tech|awards/entertainment|other\",\n"
    "  \"p_model\": <float 0..1>,       // round to 3 decimals\n"
    "  \"confidence\": <float 0..1>,    // round to 2 decimals\n"
    "  \"rationale_short\": \"<=50 words\",\n"
    "  \"sources\": [\n"
    "    {\"source\":\"<domain>\",\"date\":\"YYYY-MM-DD\",\"type\":\"official|news|data|other\",\"note\":\"<=15w\"}\n"
    "  ]\n"
    "}\n\n"
    "HARD REQUIREMENTS\n"
    "- Produce ONE object per input title. Never return an empty array.\n"
    "- Always output a numeric p_model, even if you cannot browse or found nothing.\n"
    "- If web tools fail or evidence is weak, use the base-rate fallback rules below (still produce numbers).\n\n"
    "WORKFLOW (per event)\n"
    "1) Classify category from the title: sports | politics | finance/economy | crypto/tech | awards/entertainment | other.\n"
    "2) Decide if searching helps. Only use web tools if it likely shifts p_model by ≥0.03 OR raises confidence by ≥0.10.\n"
    "   - If you search: run up to 3 focused queries, prefer official data, polls/surveys, schedules/rosters, tier-1 news (≤30 days).\n"
    "3) Extract signals and map to probability using the conversion rules below.\n"
    "4) If signals insufficient, use BASE-RATE FALLBACK for that category (still output numbers).\n"
    "5) Set confidence by source quality/recency/quantity:\n"
    "   - Strong (official+independent confirmation): 0.70–0.90\n"
    "   - Mixed/partial: 0.45–0.65\n"
    "   - Fallback/no sources: 0.10–0.30\n"
    "6) Output the object with rounded numbers and ≤50-word rationale. If you used web results, include up to 4 sources.\n\n"
    "CONVERSION RULES (simple & transparent)\n"
    "A) Politics (elections, leadership changes)\n"
    "   - If polls: let L = leader margin (pp), N = sample size (use 800 if unknown).\n"
    "     Approx SE ≈ sqrt(0.25/N). Z = L/SE. Prob ≈ Φ(Z). (Clamp 0.05..0.95). \n"
    "     Adjust −0.03 for >60 days to election; +incumbency bonus +0.02 if applicable.\n"
    "   - If official disqualification/ballot issues confirmed → shift ±0.05 to 0.15 depending on severity.\n"
    "   - No polls? Use base rate (below).\n\n"
    "B) Sports (season outcomes, awards, matches)\n"
    "   - If current standings or official injury/roster news:\n"
    "     Start from uniform over plausible contenders K (choose K from context, usually 4–8 for divisions/awards),\n"
    "     then adjust ±0.02..0.08 for strong/weak signals (injury to star, clinched status, etc.).\n"
    "   - One-off match with known rankings? Higher-rated side 0.60–0.75, else 0.50 ± small.\n"
    "   - No info? Use base rate.\n\n"
    "C) Finance/Economy (Fed/ECB decisions, CPI prints, jobs report bins)\n"
    "   - If consensus forecasts available (tier-1 outlets): Map “in-line” to 0.50; surprises (hawkish/dovish) ±0.05..0.15.\n"
    "   - If official guidance/calendar implies low/hi likelihood, adjust ±0.05..0.10.\n"
    "   - No info? Use base rate.\n\n"
    "D) Crypto/Tech (listings/releases)\n"
    "   - Official roadmap/announcement present: 0.60–0.80 depending on specificity & date.\n"
    "   - Credible rumor only: 0.35–0.55.\n"
    "   - No credible signals: base rate.\n\n"
    "E) Awards/Entertainment\n"
    "   - Official nomination shortlists: nominee 0.30–0.50 depending on field size & critics’ momentum; non-nominee 0.05–0.15.\n"
    "   - Critics’/guild momentum adds +0.05..0.10.\n"
    "   - No info? Base rate.\n\n"
    "BASE-RATE FALLBACKS (when evidence insufficient or no browsing)\n"
    "- Binary leadership/election unknowns: p_model = 0.50, confidence = 0.15.\n"
    "- Multi-team sports division/league title (4–8 contenders): p_model = 1/K (pick K=6 if unknown → 0.167), conf = 0.20.\n"
    "- Season awards with many contenders (8–12): p_model = 1/K (use K=10 → 0.10), conf = 0.20.\n"
    "- Speculative tech/listings without signals: p_model = 0.25, conf = 0.15.\n"
    "- Macroeconomic bin without forecasts: p_model = 0.50, conf = 0.20.\n"
    "- Ambiguous/other: p_model = 0.50, conf = 0.10.\n\n"
    "STRICT POLICIES\n"
    "- Never use bookmaker or exchange prices to set p_model. Polls, official data, news, schedules are allowed.\n"
    "- Round p_model to 3 decimals, confidence to 2 decimals.\n"
    "- Do NOT return prose outside the JSON array. Do NOT return an empty array.\n"
    "\nOUTPUT: A JSON object { \"results\": [ ... ] } where results is the array of objects described above. Never return an empty array; produce one object per input title.\n"
)


RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "odds_array_wrapper",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["results"],
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "title",
                            "category",
                            "p_model",
                            "confidence",
                            "rationale_short",
                            "sources",
                        ],
                        "properties": {
                            "title": {"type": "string"},
                            "category": {
                                "type": "string",
                                "enum": [
                                    "sports",
                                    "politics",
                                    "finance/economy",
                                    "crypto/tech",
                                    "awards/entertainment",
                                    "other",
                                ],
                            },
                            "p_model": {"type": "number"},
                            "confidence": {"type": "number"},
                            "rationale_short": {"type": "string"},
                            "sources": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "required": ["source", "date", "type", "note"],
                                    "properties": {
                                        "source": {"type": "string"},
                                        "date": {"type": "string"},
                                        "type": {"type": "string", "enum": ["official", "news", "data", "other"]},
                                        "note": {"type": "string"},
                                    },
                                },
                            },
                        },
                    },
                }
            }
        }
    }
}

def _build_edge_user_prompt(events: list[dict]) -> str:
    items = []
    for e in events:
        title = (e.get("title", "") or "").strip()
        rules = (e.get("rules", "") or "").strip()
        if rules:
            combined = f"{title}\n\nDescription: {rules}"
        else:
            combined = title
        items.append({"title": combined, "rules": rules})
    pretty = json.dumps(items, ensure_ascii=False, indent=2)
    return (
        f"EVENTS:\n{pretty}\n\n"
        "TASK: Return ONLY:\n{ \"results\": [ ...objects per the system spec... ] }"
    )


def _save_cleaned_pairs(df: pd.DataFrame, path: str = CLEANED_PAIRS_PATH_DEFAULT) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False, encoding='utf-8-sig')
        logger.info("Saved cleaned pairs to %s (rows=%d)", path, len(df.index))
    except Exception:
        logger.exception("Failed to save cleaned pairs to %s", path)


def _load_cleaned_pairs(path: str = CLEANED_PAIRS_PATH_DEFAULT) -> pd.DataFrame:
    try:
        if not os.path.exists(path):
            logger.error("Cleaned pairs file not found at %s", path)
            return pd.DataFrame(columns=["question", "token1", "token2"])
        df = pd.read_csv(path)
        # Ensure expected schema
        for col in ("question", "token1", "token2"):
            if col not in df.columns:
                logger.error("Cleaned pairs file missing required column: %s", col)
                return pd.DataFrame(columns=["question", "token1", "token2"])
        # Preserve optional 'rules' column if present
        cols = [c for c in ["question", "token1", "token2", "rules"] if c in df.columns]
        df = df[cols].copy()
        df["question"] = df["question"].astype(str)
        df["token1"] = df["token1"].astype(str)
        df["token2"] = df["token2"].astype(str)
        if "rules" in df.columns:
            df["rules"] = df["rules"].astype(str)
        df = df.reset_index(drop=True)
        logger.info("Loaded cleaned pairs from %s (rows=%d)", path, len(df.index))
        return df
    except Exception:
        logger.exception("Failed to load cleaned pairs from %s", path)
        return pd.DataFrame(columns=["question", "token1", "token2"])


# ---------- LOGGING SETUP ----------

def _ensure_log_dir(path: str) -> str:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        logger.exception("Failed to create log directory: %s", path)
    return path


def _create_run_log_file(log_dir: str = LOG_DIR_DEFAULT, base_name: str = "filter_markets") -> str:
    dir_path = _ensure_log_dir(log_dir)
    timestamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_name}_{timestamp}.log"
    return os.path.join(dir_path, filename)


def _prune_old_logs(log_dir: str = LOG_DIR_DEFAULT, base_name: str = "filter_markets", keep: int = 3) -> None:
    try:
        if not os.path.isdir(log_dir):
            return
        entries = [
            os.path.join(log_dir, f)
            for f in os.listdir(log_dir)
            if f.startswith(f"{base_name}_") and f.endswith(".log")
        ]
        entries.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        for old in entries[keep:]:
            try:
                os.remove(old)
            except Exception:
                logger.debug("Failed to remove old log file: %s", old, exc_info=True)
    except Exception:
        logger.debug("Log pruning failed", exc_info=True)


def _configure_logging_for_run(log_dir: str = LOG_DIR_DEFAULT, base_name: str = "filter_markets") -> None:
    log_path = _create_run_log_file(log_dir=log_dir, base_name=base_name)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    # Configure ROOT logger so all modules' logs (ai.search_agent, httpx, etc.)
    # go to both console and the per-run file.
    root_logger = logging.getLogger()

    # Ensure a console handler exists on root
    has_console = any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers)
    if not has_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO))
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # Always add a fresh file handler for this run on root
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(getattr(logging, os.getenv("LOG_FILE_LEVEL", os.getenv("LOG_LEVEL", "INFO")).upper(), logging.INFO))
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Set root level from env (default INFO) to avoid overly verbose logs
    try:
        root_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
        root_level = getattr(logging, root_level_name, logging.INFO)
    except Exception:
        root_level = logging.INFO
    root_logger.setLevel(root_level)

    # Exclude noisy libraries
    try:
        httpcore_logger = logging.getLogger("httpcore")
        httpcore_logger.setLevel(logging.CRITICAL + 1)
        httpcore_logger.propagate = False
    except Exception:
        pass

    # Make this module logger delegate to root and avoid duplicate handlers
    try:
        for h in list(logger.handlers):
            logger.removeHandler(h)
    except Exception:
        pass
    logger.setLevel(root_logger.level)
    logger.propagate = True

    # Prune old log files
    _prune_old_logs(log_dir=log_dir, base_name=base_name, keep=int(os.getenv("LOG_KEEP", "3")))


def _extract_titles_only(pairs_df: pd.DataFrame) -> list[dict]:
    events: list[dict] = []
    for _, r in pairs_df.iterrows():
        title = str(r.get("question", "")).strip()
        rules = str(r.get("rules", "")).strip()
        if not title:
            continue
        events.append({"title": title, "rules": rules})
    return events


def build_odds_market_comparison(odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine the model odds (p_model, confidence) with current Polymarket prices, compute differences,
    and flag significant gaps. Input must be a DataFrame with at least columns: title, p_model, confidence.
    """
    try:
        threshold_env = os.getenv("EDGE_DIFF_THRESHOLD", "0.10")
        threshold = float(threshold_env)
    except Exception:
        threshold = 0.10

    # Normalize titles for join
    odds_df = odds_df.copy()
    odds_df["title_norm"] = odds_df["title"].astype(str).map(_normalize_market_title)
    # Filter low-confidence predictions (< 0.3)
    try:
        odds_df["confidence"] = pd.to_numeric(odds_df.get("confidence", 0), errors="coerce").fillna(0.0)
        odds_df = odds_df[odds_df["confidence"] >= 0.3].copy()
        if odds_df.empty:
            logger.info("All predictions filtered out by confidence < 0.3; returning empty DataFrame")
            return pd.DataFrame(columns=[
                "title", "category", "p_model", "confidence", "rationale", "question", "token1", "token2",
                "reward_min_size", "reward_paid", "p_actual", "edge", "abs_edge", "significant",
            ])
    except Exception:
        logger.debug("Failed to filter by confidence; proceeding without filter", exc_info=True)

    # Get current markets and question-token mapping
    markets_df = get_all_markets()
    if markets_df is None or markets_df.empty:
        logger.warning("No markets available for comparison; returning empty DataFrame")
        return pd.DataFrame()

    pairs_df = get_market_name_token_pairs()
    if pairs_df is None or pairs_df.empty:
        logger.warning("No token pairs available for comparison; returning empty DataFrame")
        return pd.DataFrame()

    pairs_df = pairs_df.copy()
    pairs_df["question_norm"] = pairs_df["question"].astype(str).map(_normalize_market_title)

    # Prepare rewards mappings from markets
    rewards_min_size_map: dict[str, float] = {}
    rewards_paid_map: dict[str, bool] = {}
    try:
        markets_norm = markets_df.copy()
        markets_norm["question_norm"] = markets_norm["question"].astype(str).map(_normalize_market_title)

        def _extract_reward_min_size(rewards_obj) -> float:
            try:
                if isinstance(rewards_obj, dict):
                    val = rewards_obj.get("min_size")
                    if val is None:
                        return 0.0
                    try:
                        return float(val)
                    except Exception:
                        try:
                            return float(str(val).strip())
                        except Exception:
                            return 0.0
                return 0.0
            except Exception:
                return 0.0

        def _has_reward(rewards_obj) -> bool:
            try:
                if not isinstance(rewards_obj, dict):
                    return False
                # Consider reward present if any of the expected fields exist
                if any(k in rewards_obj for k in ("min_size", "max_spread", "reward_epoch", "in_game_multiplier")):
                    return True
                return False
            except Exception:
                return False

        if "rewards" in markets_norm.columns:
            markets_norm["reward_min_size"] = markets_norm["rewards"].apply(_extract_reward_min_size)
            markets_norm["reward_paid"] = markets_norm["rewards"].apply(_has_reward)
        else:
            markets_norm["reward_min_size"] = 0.0
            markets_norm["reward_paid"] = False
        rewards_min_size_map = dict(zip(markets_norm["question_norm"], markets_norm["reward_min_size"]))
        rewards_paid_map = dict(zip(markets_norm["question_norm"], markets_norm["reward_paid"]))
    except Exception:
        logger.debug("Failed to build rewards min_size map from markets_df", exc_info=True)

    # Inner join odds to token pairs by normalized title
    joined = odds_df.merge(pairs_df, left_on="title_norm", right_on="question_norm", how="inner")
    if joined.empty:
        logger.warning("No matches between odds titles and markets questions after normalization")
        return pd.DataFrame()

    # Fetch midpoints for token1
    client = get_clob_client()
    if client is None:
        logger.error("CLOB client unavailable; cannot fetch market prices")
        return pd.DataFrame()

    def fetch_midpoint(token_id: str) -> float:
        try:
            book = get_order_book_with_retry(client, token_id)
            raw_bids = getattr(book, "bids", []) or []
            raw_asks = getattr(book, "asks", []) or []

            def extract_price(entry) -> float:
                try:
                    if isinstance(entry, dict):
                        return float(entry.get("price", 0.0))
                    # Some SDKs return objects with .price attribute
                    price_val = getattr(entry, "price", 0.0)
                    return float(price_val)
                except Exception:
                    return 0.0

            bid_prices = [extract_price(e) for e in raw_bids if extract_price(e) > 0.0]
            ask_prices = [extract_price(e) for e in raw_asks if extract_price(e) > 0.0]

            best_bid = max(bid_prices) if bid_prices else 0.0
            best_ask = min(ask_prices) if ask_prices else 0.0

            if best_bid == 0.0 and best_ask == 0.0:
                return 0.0
            if best_bid == 0.0:
                return best_ask
            if best_ask == 0.0:
                return best_bid
            return (best_bid + best_ask) / 2.0
        except Exception:
            logger.debug("Failed to fetch order book for token=%s", token_id, exc_info=True)
            return 0.0

    # Map token1 -> midpoint
    unique_tokens: list[str] = [t for t in joined["token1"].astype(str).unique().tolist() if t]
    token_to_mid: dict[str, float] = {}
    for t in unique_tokens:
        token_to_mid[t] = fetch_midpoint(t)

    # Build output
    out = joined.copy()
    out["p_actual"] = out["token1"].map(lambda t: token_to_mid.get(str(t), 0.0))
    # Attach reward min size by normalized question
    try:
        out["reward_min_size"] = out["question_norm"].map(lambda q: float(rewards_min_size_map.get(str(q), 0.0)))
    except Exception:
        out["reward_min_size"] = 0.0
    # Attach reward paid boolean
    try:
        out["reward_paid"] = out["question_norm"].map(lambda q: bool(rewards_paid_map.get(str(q), False)))
    except Exception:
        out["reward_paid"] = False
    out["edge"] = out["p_model"].astype(float) - out["p_actual"].astype(float)
    out["abs_edge"] = out["edge"].abs()
    out["significant"] = out["abs_edge"] >= threshold

    # Include rationale text from predictions
    try:
        if "rationale_short" in out.columns:
            out["rationale"] = out["rationale_short"].astype(str)
        elif "rationale" in out.columns:
            out["rationale"] = out["rationale"].astype(str)
        else:
            out["rationale"] = ""
    except Exception:
        out["rationale"] = ""

    # Select and sort
    cols = [
        "title", "category", "p_model", "confidence", "rationale", "question", "token1", "token2",
        "reward_min_size", "reward_paid",
        "p_actual", "edge", "abs_edge", "significant",
    ]
    present_cols = [c for c in cols if c in out.columns]
    out = out[present_cols].sort_values(by=["significant", "edge"], ascending=[False, False]).reset_index(drop=True)
    return out

def compute_edge_for_markets(pairs_df: pd.DataFrame,
                             chunk_size: int = 12,
                             model: str = "gpt-5",
                             max_workers: int = 4) -> list[dict]:
    """
    For cleaned market pairs, build title-only events, then ask OpenAI to estimate p_model/confidence.
    If the AI decides to browse, tool calls are handled, with salvage on heavy 403/429.
    Returns a list of odds estimation objects as specified by the system prompt.
    """
    try:
        events = _extract_titles_only(pairs_df)
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

        # Determine if Tavily is available to avoid repeated exceptions
        tavily_enabled = bool((os.getenv("TAVILY_API_KEY") or "").strip())

        # --- Retryable tracking helpers ---
        RETRYABLE_HTTP = {403, 429, 503}

        class ChunkStats:
            def __init__(self) -> None:
                self.successes: int = 0
                self.errors: list[dict] = []  # {status:int|None, url:str, kind:"search"|"get"}

            def record_success(self) -> None:
                self.successes += 1

            def record_error(self, status: int | None, url: str, kind: str) -> None:
                self.errors.append({"status": status, "url": url, "kind": kind})

            def only_retryables_and_no_success(self) -> bool:
                if self.successes != 0 or len(self.errors) == 0:
                    return False
                statuses = [e.get("status") for e in self.errors]
                known = [s for s in statuses if isinstance(s, int)]
                if not known:
                    return False
                return all(s in RETRYABLE_HTTP for s in known)

        def _extract_http_status(exc: Exception) -> int | None:
            try:
                resp = getattr(exc, "response", None)
                code = getattr(resp, "status_code", None)
                return int(code) if code is not None else None
            except Exception:
                return None

        def should_salvage(now_ts: dt.datetime, start_ts: dt.datetime, time_budget_s: int, stats: ChunkStats) -> bool:
            remaining = time_budget_s - int((now_ts - start_ts).total_seconds())
            return stats.only_retryables_and_no_success() and remaining >= 30

        SALVAGE_SYSTEM = (
            "You attempted web research but all fetches returned 403/429 (blocked).\n"
            "Propose 3 alternative SEARCH QUERIES that avoid blocked domains and still inform the probability.\n\n"
            "RULES:\n"
            "- Do NOT use these domains: oddschecker.com, betfair.com, pinnacle.com\n"
            "- Prefer accessible sources: official league/team sites, Wikipedia, Reuters/AP/AFP, ESPN/Kicker/WhoScored/Transfermarkt, government/NGO pages.\n"
            "- Keep each query concise (<=12 words), no commentary. Return JSON: {\"queries\":[ \"...\", \"...\", \"...\" ]}.\n"
        )

        def salvage_user_prompt(titles: list[str]) -> str:
            return "EVENTS:\n" + json.dumps([{"title": t} for t in titles], ensure_ascii=False) + \
                   "\n\nTASK: Return only JSON with 3 better queries."

        BLOCKED = ["oddschecker.com", "betfair.com", "pinnacle.com"]
        FAST_GET_WHITELIST = [
            "reuters.com","apnews.com","bundesliga.com","dfl.de",
            "kicker.de","whoscored.com","transfermarkt","espn.com","wikipedia.org",
        ]

        def process_chunk(chunk_index: int, chunk_events: list[dict]) -> list[dict]:
            logger.debug("Chunk %s: start; events=%d", chunk_index, len(chunk_events))
            system_prompt = EDGE_SYSTEM_PROMPT
            user_prompt = _build_edge_user_prompt(chunk_events)
            # Keep logs light; skip noisy previews
            api_key = os.getenv("OPENAI_API_KEY", "").strip()
            client_local = OpenAI(api_key=api_key) if api_key else OpenAI()
            stats = ChunkStats()
            salvage_attempted = False

            tools_spec = [
                {
                    "type": "function",
                    "function": {
                        "name": "tavily_search",
                        "description": "Search the web with Tavily and return fresh, authoritative results for the given query.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "Full search string."},
                                "time_range": {"type": "string", "enum": ["d", "w", "m", "y", "all"], "description": "d=day, w=week, m=month, y=year."},
                                "max_results": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
                            },
                            "required": ["query"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "tavily_get",
                        "description": "Fetch and extract readable text from a URL for probability, odds, polls, or official announcements.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "url": {"type": "string"},
                            },
                            "required": ["url"],
                        },
                    },
                },
            ]

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            try:
                resp = client_local.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=tools_spec,
                    tool_choice="auto",
                    response_format=RESPONSE_FORMAT,
                )
            except Exception:
                logger.warning("OpenAI initial request failed for chunk %s; skipping", chunk_index)
                return []

            # Allow more tool-call steps; configurable via env
            try:
                max_steps = int(os.getenv("EDGE_TOOL_MAX_STEPS", "40"))
            except Exception:
                max_steps = 40
            if max_steps < 1:
                max_steps = 1
            # Time cap per chunk to avoid long runs
            try:
                max_seconds = int(os.getenv("EDGE_CHUNK_MAX_SECONDS", "180"))
            except Exception:
                max_seconds = 180
            start_time = dt.datetime.utcnow()
            steps = 0
            content: str = ""
            while steps < max_steps:
                steps += 1
                try:
                    choice = resp.choices[0]
                except Exception:
                    logger.debug("No choices returned (chunk %s)", chunk_index)
                    break

                finish = getattr(choice, "finish_reason", None)
                tool_calls = getattr(choice.message, "tool_calls", None)
                content = (choice.message.content or "").strip() if getattr(choice, "message", None) else ""
                logger.debug("Chunk %s: step=%d finish=%s tool_calls=%d content_len=%d", chunk_index, steps, str(finish), len(tool_calls or []), len(content))

                if finish == "stop" and not tool_calls:
                    break

                if tool_calls:
                    for call in tool_calls or []:
                        try:
                            fn_name = call.function.name
                            args = json.loads(call.function.arguments or "{}")
                        except Exception:
                            logger.debug("Malformed tool call in chunk %s", chunk_index)
                            continue

                        if fn_name == "tavily_search":
                            query = str(args.get("query", "")).strip()
                            time_range = str(args.get("time_range", os.getenv("TAVILY_TIME_RANGE", "m")))
                            max_results = int(args.get("max_results", int(os.getenv("TAVILY_MAX_RESULTS", "5"))))
                            if tavily_enabled:
                                try:
                                    results = tavily_search_impl(query=query, time_range=time_range, max_results=max_results)
                                    if isinstance(results, dict) and len(results.get("results", []) or []) > 0:
                                        stats.record_success()
                                    else:
                                        stats.record_error(None, query, "search")
                                except Exception as exc:
                                    status = _extract_http_status(exc)
                                    logger.debug("tavily_search_impl error: chunk=%s status=%s query=%r", chunk_index, status, query)
                                    stats.record_error(status, query, "search")
                                    results = {"results": [], "meta": {"query": query}}
                            else:
                                logger.warning("TAVILY_API_KEY not set; skipping tavily_search for query: %s", query)
                                results = {"results": [], "meta": {"query": query, "skipped": True}}

                            messages.append({"role": "assistant", "tool_calls": [call]})
                            messages.append({
                                "role": "tool",
                                "tool_call_id": call.id,
                                "name": "tavily_search",
                                "content": json.dumps(results),
                            })
                        elif fn_name == "tavily_get":
                            url = str(args.get("url", "")).strip()
                            if tavily_enabled:
                                try:
                                    page = tavily_get_impl(url)
                                    if isinstance(page, dict) and (page.get("text") or page.get("title")):
                                        stats.record_success()
                                except Exception as exc:
                                    status = _extract_http_status(exc)
                                    logger.debug("tavily_get_impl error: chunk=%s status=%s url=%s", chunk_index, status, url)
                                    stats.record_error(status, url, "get")
                                    page = {"url": url, "title": None, "text": None, "length": 0}
                            else:
                                logger.warning("TAVILY_API_KEY not set; skipping tavily_get for url: %s", url)
                                page = {"url": url, "title": None, "text": None, "length": 0, "skipped": True}

                            messages.append({"role": "assistant", "tool_calls": [call]})
                            messages.append({
                                "role": "tool",
                                "tool_call_id": call.id,
                                "name": "tavily_get",
                                "content": json.dumps(page),
                            })

                    try:
                        logger.debug("Chunk %s: sending follow-up completion (post tool results)", chunk_index)
                        resp = client_local.chat.completions.create(
                            model=model,
                            messages=messages,
                            tools=tools_spec,
                            tool_choice="auto",
                            response_format=RESPONSE_FORMAT,
                        )
                        # Check time budget
                        elapsed = (dt.datetime.utcnow() - start_time).total_seconds()
                        # Optional salvage before giving up, if only retryables and no success
                        if not salvage_attempted and should_salvage(dt.datetime.utcnow(), start_time, max_seconds, stats):
                            try:
                                salvage_attempted = True
                                titles = [e.get("title", "") for e in chunk_events]
                                logger.debug("Chunk %s: running salvage pass", chunk_index)
                                salvage_resp = client_local.chat.completions.create(
                                    model=model,
                                    messages=[{"role": "system", "content": SALVAGE_SYSTEM}, {"role": "user", "content": salvage_user_prompt(titles)}],
                                    tool_choice="none",
                                )
                                qtext = (salvage_resp.choices[0].message.content or "").strip()
                                queries = []
                                try:
                                    qjson = json.loads(qtext)
                                    queries = list(qjson.get("queries", []) or [])
                                except Exception:
                                    queries = []
                                queries = [q for q in queries if isinstance(q, str) and q.strip()][:3]
                                found: list[dict] = []
                                for q in queries[:2]:
                                    try:
                                        data = tavily_search_impl(q, time_range="m", max_results=4, search_depth="basic")
                                        for r in data.get("results", []) or []:
                                            u = (r.get("url") or "").lower()
                                            if any(b in u for b in BLOCKED):
                                                continue
                                            if any(k in u for k in FAST_GET_WHITELIST):
                                                try:
                                                    doc = tavily_get_impl(r.get("url") or "")
                                                    if isinstance(doc, dict) and (doc.get("text") or doc.get("title")):
                                                        stats.record_success()
                                                        found.append({"title": r.get("title"), "url": r.get("url"), "snippet": r.get("snippet")})
                                                        break
                                                except Exception as exc2:
                                                    status2 = _extract_http_status(exc2)
                                                    stats.record_error(status2, r.get("url") or "", "get")
                                    except Exception as exc1:
                                        status1 = _extract_http_status(exc1)
                                        stats.record_error(status1, q, "search")
                                    if found:
                                        break
                                if found:
                                    messages.append({"role": "user", "content": "SALVAGE: Additional snippets: " + json.dumps(found) + "\nReturn ONLY:\n{ \"results\": [ ...objects per the system spec... ] }"})
                                    resp = client_local.chat.completions.create(
                                        model=model,
                                        messages=messages,
                                        tools=tools_spec,
                                        tool_choice="none",
                                        response_format=RESPONSE_FORMAT,
                                    )
                            except Exception:
                                logger.debug("Chunk %s: salvage pass failed", chunk_index, exc_info=True)

                        if elapsed >= max_seconds:
                            logger.debug("Time budget exceeded for chunk %s after %ss; asking for final JSON", chunk_index, int(elapsed))
                            try:
                                messages.append({"role": "user", "content": "Time limit reached. Please return ONLY:\n{ \"results\": [ ...objects per the system spec... ] }"})
                                resp = client_local.chat.completions.create(
                                    model=model,
                                    messages=messages,
                                    tools=tools_spec,
                                    tool_choice="none",
                                    response_format=RESPONSE_FORMAT,
                                )
                            except Exception:
                                logger.debug("Finalization request failed for chunk %s", chunk_index, exc_info=True)
                            break
                        continue
                    except Exception:
                        logger.exception("Odds OpenAI follow-up request failed (chunk %s)", chunk_index)
                        break

                break

            try:
                if content:
                    parsed = json.loads(content)
                    if isinstance(parsed, dict) and isinstance(parsed.get("results"), list):
                        logger.debug("Chunk %s: parsed results array (%d items)", chunk_index, len(parsed["results"]))
                        return parsed["results"]
                    if isinstance(parsed, list):
                        logger.debug("Chunk %s: parsed final JSON array (%d items)", chunk_index, len(parsed))
                        return parsed
                    start = content.find("[")
                    end = content.rfind("]")
                    if start != -1 and end != -1 and end > start:
                        parsed = json.loads(content[start:end + 1])
                        logger.debug("Chunk %s: parsed bracketed JSON array (%d items)", chunk_index, len(parsed) if isinstance(parsed, list) else 0)
                        return parsed if isinstance(parsed, list) else []
                # One more attempt: explicitly request final JSON only
                messages.append({"role": "user", "content": "Please return ONLY:\n{ \"results\": [ ...objects per the system spec... ] }"})
                try:
                    resp2 = client_local.chat.completions.create(
                        model=model,
                        messages=messages,
                        tools=tools_spec,
                        tool_choice="none",
                        response_format=RESPONSE_FORMAT,
                    )
                    final_text = (resp2.choices[0].message.content or "").strip()
                    parsed2 = json.loads(final_text)
                    if isinstance(parsed2, dict) and isinstance(parsed2.get("results"), list):
                        logger.debug("Chunk %s: parsed final JSON-only results array (%d items)", chunk_index, len(parsed2["results"]))
                        return parsed2["results"]
                    if isinstance(parsed2, list):
                        logger.debug("Chunk %s: parsed final JSON-only array (%d items)", chunk_index, len(parsed2))
                        return parsed2
                    s = final_text.find("[")
                    e = final_text.rfind("]")
                    if s != -1 and e != -1 and e > s:
                        parsed2 = json.loads(final_text[s:e + 1])
                        logger.debug("Chunk %s: parsed bracketed JSON-only array (%d items)", chunk_index, len(parsed2) if isinstance(parsed2, list) else 0)
                        return parsed2 if isinstance(parsed2, list) else []
                except Exception:
                    logger.debug("Failed final JSON-only parse for chunk %s", chunk_index)
            except Exception:
                logger.debug("Failed to parse edge JSON for chunk %s", chunk_index)
            return []

        # Run chunks in parallel
        outputs: list[dict] = []
        chunks = [events[i:i + chunk_size] for i in range(0, len(events), chunk_size)]
        logger.info("Submitting %d chunks (chunk_size=%d)", len(chunks), chunk_size)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_chunk, idx, chunk) for idx, chunk in enumerate(chunks)]
            for future in concurrent.futures.as_completed(futures):
                try:
                    res = future.result()
                    if res:
                        outputs.extend(res)
                    logger.debug("Chunk completed with %d items; total so far=%d", len(res) if res else 0, len(outputs))
                except Exception:
                    logger.warning("Parallel worker failed; continuing")
        return outputs
    except Exception:
        logger.exception("Error during compute_edge_for_markets")
        return []


if __name__ == "__main__":
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        _configure_logging_for_run()
    except Exception:
        logger.debug("Failed to initialize per-run logging", exc_info=True)

    parser = argparse.ArgumentParser(description="Filter markets and compute edges with stage control")
    parser.add_argument(
        "--debug",
        choices=["remove", "compute"],
        help=(
            "Start from a specific stage: "
            "'remove' runs remove_markets_to_avoid then compute_edge_for_markets; "
            "'compute' loads cleaned pairs and runs compute_edge_for_markets."
        ),
    )
    parser.add_argument("--cleaned-path", default=CLEANED_PAIRS_PATH_DEFAULT, help="Path to cleaned pairs CSV (load/save)")
    parser.add_argument("--odds-csv", default=os.path.join(ROOT_DIR, "data", "model_predictions.json"), help="Path to model predictions JSON (or CSV) for comparison")
    args = parser.parse_args()

    # Decide starting stage
    start_stage = args.debug or "remove"
    logger.info("Starting pipeline from stage: %s", start_stage)

    # Stage: remove_markets_to_avoid
    if start_stage == "remove":
        # When explicitly debugging 'remove', load pairs from CSV instead of fetching markets
        if args.debug == "remove":
            pairs_df = _load_cleaned_pairs(args.cleaned_path)
            try:
                logger.info("[debug remove] Loaded pairs from %s (rows=%d)", args.cleaned_path, len(pairs_df.index))
            except Exception:
                logger.info("[debug remove] Loaded pairs from %s", args.cleaned_path)
        else:
            markets_df = get_all_markets()
            if markets_df is not None and not markets_df.empty:
                try:
                    logger.info("Fetched %d markets", len(markets_df.index))
                except Exception:
                    logger.info("Fetched markets frame with shape: %s", getattr(markets_df, "shape", None))
            pairs_df = get_market_name_token_pairs()
            try:
                logger.info("Pairs preview: %s", pairs_df.head())
            except Exception:
                pass
        clean_pairs = remove_markets_to_avoid(pairs_df)
        try:
            _save_cleaned_pairs(clean_pairs, path=args.cleaned_path)
        except Exception:
            logger.debug("Unable to save cleaned pairs to custom path", exc_info=True)
        try:
            logger.info("Cleaned pairs length: %s", len(clean_pairs))
            logger.info("Cleaned pairs head: %s", clean_pairs.head())
        except Exception:
            pass
    else:
        # Stage: compute_edge_for_markets (load cleaned pairs)
        clean_pairs = _load_cleaned_pairs(args.cleaned_path)
        try:
            logger.info("Loaded cleaned pairs from %s (rows=%d)", args.cleaned_path, len(clean_pairs.index))
        except Exception:
            logger.info("Loaded cleaned pairs from %s", args.cleaned_path)

    # Stage: compute_edge_for_markets
    odds_results = compute_edge_for_markets(clean_pairs)
    logger.info("Computed odds items: %d", len(odds_results))

    # Persist computed odds
    try:
        odds_out_path = os.path.join(ROOT_DIR, "data", "model_predictions.json")
        os.makedirs(os.path.dirname(odds_out_path), exist_ok=True)
        with open(odds_out_path, "w", encoding="utf-8") as fh:
            json.dump({"results": odds_results}, fh, ensure_ascii=False, indent=2)
        logger.info("Wrote model odds to %s (items=%d)", odds_out_path, len(odds_results))
    except Exception:
        logger.exception("Failed to write model odds JSON")

    # Build comparison input: default to computed odds, optionally override from file in compute-stage debug
    odds_df_in = pd.DataFrame.from_records(odds_results)
    if start_stage == "compute":
        try:
            if str(args.odds_csv).lower().endswith('.json'):
                with open(args.odds_csv, 'r', encoding='utf-8') as fh:
                    data = json.load(fh)
                if isinstance(data, dict) and isinstance(data.get('results'), list):
                    odds_df_in = pd.DataFrame.from_records(data.get('results', []))
                elif isinstance(data, list):
                    odds_df_in = pd.DataFrame.from_records(data)
                else:
                    logger.error('Debug odds JSON not in expected format at %s', args.odds_csv)
                    odds_df_in = pd.DataFrame()
            else:
                odds_df_in = pd.read_csv(args.odds_csv)

            # Mirror previous behavior: also save the loaded odds snapshot
            try:
                odds_records_dbg = odds_df_in.to_dict(orient='records')
                odds_out_path = os.path.join(ROOT_DIR, 'data', 'model_predictions.json')
                os.makedirs(os.path.dirname(odds_out_path), exist_ok=True)
                with open(odds_out_path, 'w', encoding='utf-8') as fh:
                    json.dump({"results": odds_records_dbg}, fh, ensure_ascii=False, indent=2)
                logger.info('Wrote model odds (debug override) to %s (items=%d)', odds_out_path, len(odds_records_dbg))
            except Exception:
                logger.warning('Failed to write model odds JSON (debug override)')
        except Exception:
            logger.warning('Failed to load odds from file for debug compute stage')

    comp_df = build_odds_market_comparison(odds_df_in)
    if comp_df is not None and not comp_df.empty:
        out_path = os.path.join(ROOT_DIR, 'data', 'odds_vs_market.csv')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        comp_df.to_csv(out_path, index=False, encoding='utf-8-sig')
        logger.info('Wrote odds vs market comparison to %s (rows=%d)', out_path, len(comp_df.index))
    else:
        logger.info('Odds vs market comparison produced no rows')
