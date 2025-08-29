import os
import re
import html
import logging
import random
import time
import datetime as dt

from typing import Dict, Any, Optional
from curl_cffi import requests as cffi_requests

from dotenv import load_dotenv
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from bs4 import BeautifulSoup
import html2text

TAVILY_ENDPOINT = "https://api.tavily.com/search"
TAVILY_TIMEOUT = 20  # seconds

# Map your tool's time_range to Tavily "days" window.
# d=day, w=week, m=month, y=year, all=None (no filter)
TIME_RANGE_TO_DAYS = {
    "d": 1,
    "w": 7,
    "m": 30,
    "y": 365,
    "all": None
}

class TavilyError(Exception):
    pass

def _now_utc_date() -> str:
    return dt.datetime.utcnow().strftime("%Y-%m-%d")

def _extract_published_date(candidate: Optional[str]) -> Optional[str]:
    """
    Normalize any timestamp-like field to YYYY-MM-DD if possible.
    Accepts '2025-08-27T10:22:00Z' or '2025-08-27', returns '2025-08-27'.
    """
    if not candidate:
        return None
    # Try ISO date first
    m = re.match(r"(\d{4}-\d{2}-\d{2})", candidate)
    if m:
        return m.group(1)
    # Fallback: try to parse 'Aug 27, 2025' etc. (very light heuristic)
    try:
        return dt.datetime.strptime(candidate, "%b %d, %Y").strftime("%Y-%m-%d")
    except Exception:
        return None


# ---------- INIT & LOGGER ----------

load_dotenv()
logger = logging.getLogger(__name__)


# ---------- SEARCH ----------

@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.8, min=0.8, max=4),
    retry=retry_if_exception_type((httpx.HTTPError, TavilyError))
)
def tavily_search_impl(query: str, time_range: str = "m", max_results: int = 5, search_depth: str = "basic") -> Dict[str, Any]:
    """
    Call Tavily Search API and normalize the results.
    Returns a dict: {"results": [{"title","url","snippet","date","score"}], "meta": {...}}
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise TavilyError("Missing TAVILY_API_KEY environment variable.")

    days = TIME_RANGE_TO_DAYS.get(time_range, 30)
    depth = "advanced" if str(search_depth).lower() == "advanced" else "basic"

    try:
        logger.info("Tavily search query=%r time_range=%s max_results=%s depth=%s", query, time_range, max_results, depth)
    except Exception:
        pass

    payload = {
        "api_key": api_key,
        "query": query,
        # Depth: "basic" is cheaper+faster; "advanced" crawls more.
        "search_depth": depth,
        # Include short synthesized answer? (we rely on raw results)
        "include_answer": False,
        # Limit number of links
        "max_results": max(1, min(int(max_results), 10)),
    }

    # If you want freshness, pass "days"
    if days is not None:
        payload["days"] = int(days)

    # Optional: You can add domain filters if you wish
    # payload["include_domains"] = []
    # payload["exclude_domains"] = []

    with httpx.Client(timeout=TAVILY_TIMEOUT) as client:
        resp = client.post(TAVILY_ENDPOINT, json=payload)
        resp.raise_for_status()
        data = resp.json()

    # Tavily typically returns {"results":[{"title","url","content","score",...},...], ...}
    raw_results = data.get("results", []) or []

    results = []
    for r in raw_results:
        title = r.get("title") or ""
        url = r.get("url") or ""
        # 'content' is a snippet/summary from Tavily
        snippet = (r.get("content") or "").strip()
        score = r.get("score")
        # Tavily may or may not provide a date; if not, leave None
        date = _extract_published_date(r.get("published_date")) or _extract_published_date(r.get("date"))

        results.append({
            "title": title,
            "url": url,
            "snippet": snippet,
            "date": date,
            "score": float(score) if score is not None else None
        })

    return {
        "results": results,
        "meta": {
            "query": query,
            "time_range": time_range,
            "days": days,
            "search_depth": depth,
            "response_time_ms": data.get("response_time", None),
            "fetched_at": _now_utc_date()
        }
    }


# ---------- GET & EXTRACT ----------

@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1.0, min=1, max=6),
    retry=retry_if_exception_type(httpx.HTTPError)
)
def tavily_get_impl(url: str) -> Dict[str, Any]:
    """
    Fetch a URL and extract readable text; hardened for sites with basic bot protection.
    Set PROXY_URL=http://user:pass@host:port if you want to proxy.
    If curl_cffi is installed, set USE_CURL_CFFI=1 to use real browser TLS fingerprinting.
    """

    PROXY_URL = os.getenv("PROXY_URL")
    USE_CURL_CFFI = os.getenv("USE_CURL_CFFI", "0") == "1"

    ua_pool = [
        # Recent stable desktop Chrome UAs
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    ]
    headers = {
        "User-Agent": random.choice(ua_pool),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-GB,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-User": "?1",
        "Sec-Fetch-Dest": "document",
        "Referer": "https://www.google.com/",
    }

    proxies = {"http://": PROXY_URL, "https://": PROXY_URL} if PROXY_URL else None

    # Try curl_cffi (better TLS fingerprint) if requested
    if USE_CURL_CFFI:
        try:
            with cffi_requests.Session() as s:
                s.headers.update(headers)
                if proxies:
                    s.proxies.update(proxies)
                resp = s.get(url, impersonate="chrome", timeout=25)
                if resp.status_code in (403, 429):
                    time.sleep(2.0)
                    resp = s.get(url, impersonate="chrome", timeout=25)
                resp.raise_for_status()
                html_bytes = resp.content
                content_type = resp.headers.get("content-type", "")
        except Exception as e:  # noqa: F841
            # Fall back to httpx below
            pass

    if 'html_bytes' not in locals():
        # httpx path with cookie persistence and retries
        transport = httpx.HTTPTransport(retries=0)  # tenacity handles retries
        with httpx.Client(timeout=25, headers=headers, follow_redirects=True,
                          transport=transport, proxies=proxies) as client:
            # Prime a Google visit to set some cookies/flow
            try:
                client.get("https://www.google.com", timeout=10)
                time.sleep(0.3)
            except Exception:
                pass

            resp = client.get(url)
            if resp.status_code in (403, 429):
                # rotate UA and backoff once before raising
                headers["User-Agent"] = random.choice(ua_pool)
                client.headers.update(headers)
                time.sleep(2.0)
                resp = client.get(url)

            resp.raise_for_status()
            html_bytes = resp.content
            content_type = resp.headers.get("content-type", "")

    if "pdf" in (content_type or "").lower():
        return {
            "url": url,
            "title": None,
            "text": None,
            "length": 0,
            "content_type": content_type,
            "fetched_at": _now_utc_date()
        }

    soup = BeautifulSoup(html_bytes, "html.parser")

    # Basic title
    page_title = soup.title.string.strip() if soup.title and soup.title.string else None

    # Strip noise
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    for tag_name in ["header", "footer", "nav", "aside"]:
        for t in soup.find_all(tag_name):
            t.decompose()

    h = html2text.HTML2Text()
    h.ignore_links = False
    h.ignore_images = True
    h.body_width = 0
    text = h.handle(str(soup))
    text = html.unescape(text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    return {
        "url": url,
        "title": page_title,
        "text": text,
        "length": len(text),
        "content_type": content_type or "text/html",
        "fetched_at": _now_utc_date()
    }