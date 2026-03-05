import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import time
import datetime

from ai_scraper import extract_scholarship_data


def fetch_and_clean_html(url):
    """
    Downloads a webpage and strips scripts, styles, and junk
    to save tokens when sending to Gemini.
    """
    print(f"Fetching: {url}...")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers, timeout=(5, 10))
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "img", "svg", "nav", "footer"]):
            tag.decompose()

        clean_html = str(soup.body)[:15000]
        return clean_html

    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return None


def is_scholarship_active(scholarship: dict) -> bool:
    """
    Best-effort check: returns False only if we can clearly tell
    the scholarship deadline has already passed.
    When in doubt, returns True (keeps the scholarship).
    """
    deadline = str(scholarship.get("deadline", "")).strip().lower()

    if not deadline or deadline in ("", "nan", "none", "rolling", "varies"):
        return True

    current_year  = datetime.datetime.now().year
    current_month = datetime.datetime.now().month

    # Past year explicitly mentioned
    for year in range(2020, current_year):
        if str(year) in deadline:
            return False

    # Current year + past month
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


def run_scholarship_pipeline(urls: list, progress_callback=None) -> int:
    """
    Fetches each URL, extracts scholarships via Gemini, filters out inactive
    ones, and saves results to scholarship_database.csv.

    Args:
        urls:              List of URLs to scrape.
        progress_callback: Optional callable invoked after each URL is processed.
                           Signature: callback(url, site_num, total_sites,
                                               found_so_far, done=False)
                           Used by app.py to update the UI in real time.
    Returns:
        Number of scholarships saved.
    """
    all_scholarships = []
    total = len(urls)

    for i, url in enumerate(urls):
        if progress_callback:
            progress_callback(
                url=url,
                site_num=i + 1,
                total_sites=total,
                found_so_far=len(all_scholarships),
            )

        html_content = fetch_and_clean_html(url)

        if not html_content:
            print(f"  -> Skipping {url} (fetch failed)")
            continue

        print("  -> Sending to Gemini...")
        time.sleep(3)

        try:
            data       = extract_scholarship_data(html_content)
            found_list = data.get("scholarships", [])

            active   = 0
            inactive = 0
            for item in found_list:
                item["source_url"] = url
                if is_scholarship_active(item):
                    all_scholarships.append(item)
                    active += 1
                else:
                    inactive += 1

            print(f"  -> {active} active, {inactive} filtered as closed/expired")

        except Exception as e:
            print(f"  -> Error processing {url}: {e}")

    # Final callback so the UI shows the completed state
    if progress_callback and urls:
        progress_callback(
            url=urls[-1],
            site_num=total,
            total_sites=total,
            found_so_far=len(all_scholarships),
            done=True,
        )

    if all_scholarships:
        df = pd.DataFrame(all_scholarships)
        df = df.fillna("")
        df.to_csv("scholarship_database.csv", index=False)
        print(f"\n--- Database Updated: {len(df)} active scholarships ---")
        return len(df)
    else:
        print("\nNo active scholarships found.")
        return 0


if __name__ == "__main__":
    target_urls = [
        "https://www.careeronestop.org/Toolkit/Training/find-scholarships.aspx?keyword=data%20science&sortcolumns=BestMatch&sortdirections=Descending&p=1",
    ]
    run_scholarship_pipeline(target_urls)