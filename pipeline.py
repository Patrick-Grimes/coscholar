import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import time
import datetime
import ipaddress
from urllib.parse import urlparse
import socket
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ai_scraper import extract_scholarship_data, extract_internship_data

_EXTRACT_FNS = {
    "scholarship": extract_scholarship_data,
    "internship":  extract_internship_data,
}

_DATA_KEYS = {
    "scholarship": "scholarships",
    "internship":  "internships",
}


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _extract_with_retry(html_content, mode="scholarship", provider="Gemini", api_key=None, ollama_host=None):
    fn = _EXTRACT_FNS[mode]
    return fn(html_content, provider=provider, api_key=api_key, ollama_host=ollama_host)


def _is_safe_url(url: str) -> bool:
    """Block private/internal IPs, non-HTTP schemes, and localhost."""
    try:
        parsed = urlparse(url)
    except Exception:
        return False

    if parsed.scheme not in ("http", "https"):
        return False

    hostname = parsed.hostname or ""
    if not hostname:
        return False

    blocked = {"localhost", "127.0.0.1", "0.0.0.0", "::1", "[::1]"}
    if hostname.lower() in blocked:
        return False

    try:
        resolved = socket.getaddrinfo(hostname, None)
        for _, _, _, _, sockaddr in resolved:
            ip = ipaddress.ip_address(sockaddr[0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return False
    except (socket.gaierror, ValueError):
        return False

    return True


def fetch_and_clean_html(url):
    """
    Downloads a webpage and strips scripts, styles, and junk
    to save tokens when sending to the LLM.
    """
    if not _is_safe_url(url):
        print(f"Blocked unsafe URL: {url}")
        return None

    print(f"Fetching: {url}...")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers, timeout=(5, 10), allow_redirects=False)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "img", "svg", "nav", "footer"]):
            tag.decompose()

        clean_html = str(soup.body)[:15000]
        return clean_html

    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return None


def is_listing_active(item: dict) -> bool:
    """
    Best-effort check: returns False only if we can clearly tell
    the listing's deadline has already passed.
    When in doubt, returns True (keeps the listing).
    """
    deadline = str(item.get("deadline", "")).strip().lower()

    if not deadline or deadline in ("", "nan", "none", "rolling", "varies"):
        return True

    current_year  = datetime.datetime.now().year
    current_month = datetime.datetime.now().month

    for year in range(2020, current_year):
        if str(year) in deadline:
            return False

    month_map = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4,  "may": 5,  "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    }
    for abbr, month_num in month_map.items():
        if abbr in deadline:
            if str(current_year) in deadline and month_num < current_month:
                return False
            break

    return True


def run_pipeline(
    urls: list,
    mode: str = "scholarship",
    progress_callback=None,
    provider: str = "Gemini",
    api_key: str = None,
    ollama_host: str = None,
) -> pd.DataFrame:
    """
    Fetches each URL, extracts listings via LLM, filters out inactive
    ones, and returns the results as a DataFrame (in-memory only).

    Args:
        urls:              List of URLs to scrape.
        mode:              "scholarship" or "internship".
        progress_callback: Optional callable invoked after each URL is processed.
    Returns:
        A pandas DataFrame of active listings (may be empty).
    """
    data_key = _DATA_KEYS[mode]
    all_items = []
    total = len(urls)

    for i, url in enumerate(urls):
        if progress_callback:
            progress_callback(
                url=url,
                site_num=i + 1,
                total_sites=total,
                found_so_far=len(all_items),
            )

        html_content = fetch_and_clean_html(url)

        if not html_content:
            print(f"  -> Skipping {url} (fetch failed)")
            continue

        print(f"  -> Sending to {provider}...")
        time.sleep(1)

        try:
            data       = _extract_with_retry(html_content, mode=mode, provider=provider, api_key=api_key, ollama_host=ollama_host)
            found_list = data.get(data_key, [])

            active   = 0
            inactive = 0
            for item in found_list:
                item["source_url"] = url
                if is_listing_active(item):
                    all_items.append(item)
                    active += 1
                else:
                    inactive += 1

            print(f"  -> {active} active, {inactive} filtered as closed/expired")

        except Exception as e:
            print(f"  -> Error processing {url}: {e}")

    if progress_callback and urls:
        progress_callback(
            url=urls[-1],
            site_num=total,
            total_sites=total,
            found_so_far=len(all_items),
            done=True,
        )

    if all_items:
        df = pd.DataFrame(all_items)
        df = df.fillna("")
        print(f"\n--- Session Results: {len(df)} active {mode}s ---")
        return df

    print(f"\nNo active {mode}s found.")
    return pd.DataFrame()


# Backward-compatible alias
def run_scholarship_pipeline(urls, progress_callback=None, provider="Gemini", api_key=None, ollama_host=None):
    return run_pipeline(urls, mode="scholarship", progress_callback=progress_callback,
                        provider=provider, api_key=api_key, ollama_host=ollama_host)


if __name__ == "__main__":
    target_urls = [
        "https://www.careeronestop.org/Toolkit/Training/find-scholarships.aspx?keyword=data%20science&sortcolumns=BestMatch&sortdirections=Descending&p=1",
    ]
    run_pipeline(target_urls, mode="scholarship")