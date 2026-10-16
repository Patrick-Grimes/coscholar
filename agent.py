from datetime import date
from urllib.parse import urlparse
from ddgs import DDGS

MAX_STATES_FOR_SEARCH = 4
MAX_TOTAL_URLS = 24
BOARD_MAX_RESULTS = 6  # slightly deeper per board angle
OPEN_WEB_MAX_RESULTS = 3

# High-yield job boards + ATS hosts (boost in ranking).
BOARD_DOMAINS = (
    "linkedin.com",
    "indeed.com",
    "glassdoor.com",
    "wellfound.com",
    "angel.co",
    "greenhouse.io",
    "lever.co",
    "jobs.lever.co",
    "ashbyhq.com",
    "myworkdayjobs.com",
    "wd1.myworkdayjobs.com",
    "wd5.myworkdayjobs.com",
    "simplify.jobs",
    "joinhandshake.com",
    "handshake.com",
)

# Domains that rarely have usable structured listings.
JUNK_DOMAINS = (
    "reddit.com",
    "quora.com",
    "pinterest.com",
    "facebook.com",
    "twitter.com",
    "x.com",
    "tiktok.com",
    "youtube.com",
    "medium.com",
    "blogspot.com",
    "wordpress.com",
    "news.ycombinator.com",
    "wikipedia.org",
)


def _search_year() -> int:
    return date.today().year


def _year_token() -> str:
    """Current year for queries; include next year during fall recruiting season."""
    y = _search_year()
    if date.today().month >= 9:
        return f"{y} OR {y + 1}"
    return str(y)


def _host(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower().lstrip("www.")
    except Exception:
        return ""


def _is_board_url(url: str) -> bool:
    host = _host(url)
    if not host:
        return False
    if any(host == d or host.endswith("." + d) for d in BOARD_DOMAINS):
        return True
    # Public Workday career sites often look like company.wdN.myworkdayjobs.com/.../jobs
    if "myworkdayjobs.com" in host:
        return True
    if "/jobs" in (urlparse(url).path or "").lower() and "workday" in host:
        return True
    return False


def _is_junk_url(url: str) -> bool:
    host = _host(url)
    if not host:
        return True
    return any(host == d or host.endswith("." + d) for d in JUNK_DOMAINS)


def _url_rank_score(url: str) -> int:
    """Higher is better — board/ATS first, junk last."""
    if _is_junk_url(url):
        return -100
    if _is_board_url(url):
        host = _host(url)
        # Prefer direct ATS / listing hubs slightly over aggregators.
        if any(x in host for x in ("greenhouse.io", "lever.co", "ashbyhq.com", "myworkdayjobs.com")):
            return 90
        if any(x in host for x in ("linkedin.com", "indeed.com", "glassdoor.com", "wellfound.com", "simplify.jobs")):
            return 80
        if "handshake" in host:
            return 75
        return 70
    return 10


def _rank_and_cap_urls(urls: list[str], max_total: int = MAX_TOTAL_URLS) -> list[str]:
    """Dedupe and rank so board/ATS domains win; demote junk. Cap ~24."""
    seen = set()
    unique = []
    for u in urls:
        if not u or u in seen:
            continue
        seen.add(u)
        unique.append(u)

    ranked = sorted(unique, key=_url_rank_score, reverse=True)
    # Soft-filter: drop junk if we still have enough non-junk after ranking.
    non_junk = [u for u in ranked if not _is_junk_url(u)]
    pool = non_junk if non_junk else ranked
    return pool[:max_total]


def _focus(profile: dict) -> str:
    return (
        profile.get("desired_role", "").strip()
        or profile.get("major", "").strip()
        or "undergraduate"
    )


def _board_site_queries(focus: str, year: str, *, kind: str) -> list[str]:
    """
    Board-first site: queries for jobs or internships.
    kind: "job" | "internship"
    """
    noun = "internship" if kind == "internship" else "jobs"
    # LinkedIn uses /jobs for both; Handshake shines for internships.
    sites = [
        ("linkedin.com/jobs", f"{focus} {noun} {year}"),
        ("indeed.com", f"{focus} {noun} {year}"),
        ("glassdoor.com", f"{focus} {noun} {year}"),
        ("wellfound.com", f"{focus} {noun} {year}"),
        ("greenhouse.io", f"{focus} {noun} {year}"),
        ("lever.co", f"{focus} {noun} {year}"),
        ("ashbyhq.com", f"{focus} {noun} {year}"),
        ("jobs.lever.co", f"{focus} {noun} {year}"),
        ("myworkdayjobs.com", f"{focus} {noun} {year}"),
        ("simplify.jobs", f"{focus} {noun} {year}"),
    ]
    if kind == "internship":
        sites.append(("joinhandshake.com", f"{focus} internship {year}"))

    queries = []
    for site, q in sites:
        queries.append(f"site:{site} {q}")
    # Public Workday job listing path pattern (often company.wdN.myworkdayjobs.com/.../jobs)
    queries.append(f'site:myworkdayjobs.com/jobs {focus} {noun} {year}')
    return queries


# ── Scholarship queries ──────────────────────────────────────────────────────

def build_queries(profile: dict) -> list[str]:
    """
    Turns a student profile into a list of targeted DDG search queries.
    Each query targets a different eligibility angle so we don't miss
    scholarships that have nothing to do with major.
    """
    queries = []
    year = _year_token()

    major     = profile.get("major", "").strip()
    state     = profile.get("state", "").strip()
    states    = profile.get("states") or []
    ethnicity = profile.get("ethnicity", "").strip()
    first_gen = profile.get("first_gen", False)
    income    = profile.get("income_based", False)

    if major:
        queries.append(f"{major} scholarships {year} apply")

    if states:
        search_states = [s.strip() for s in states if s.strip()][:MAX_STATES_FOR_SEARCH]
    elif state:
        search_states = [state]
    else:
        search_states = []

    for s in search_states:
        queries.append(f"{s} state scholarships undergraduate {year}")

    if ethnicity and ethnicity.lower() not in ("", "prefer not to say"):
        queries.append(f"{ethnicity} student scholarships {year} apply")

    if first_gen:
        queries.append(f"first generation college student scholarships {year}")

    if income:
        queries.append(f"need based undergraduate scholarships {year} apply")

    queries.append(f"undergraduate scholarships {year} no major restriction apply")

    return queries


# ── Internship queries ───────────────────────────────────────────────────────

def build_internship_queries(profile: dict) -> list[str]:
    """
    Board-first internship discovery with a small open-web fallback.
    """
    year = _year_token()
    focus = _focus(profile)
    major = profile.get("major", "").strip()
    desired_role = profile.get("desired_role", "").strip()
    location_pref = profile.get("location_pref", "Any").strip()
    class_year = profile.get("class_year", "").strip()
    state = profile.get("state", "").strip()
    states = profile.get("states") or []

    # Board angles first (spend budget here).
    queries = _board_site_queries(focus, year, kind="internship")

    # Small open-web fallback — keep short so board hits dominate the URL cap.
    if desired_role:
        queries.append(f"{desired_role} internship summer {year} apply")
    elif major:
        queries.append(f"{major} internship {year} undergraduate")
    else:
        queries.append(f"undergraduate internships summer {year} apply")

    loc = location_pref.lower()
    if loc == "remote":
        queries.append(f"site:linkedin.com/jobs remote internship {focus} {year}")
    elif loc in ("hybrid", "on-site") and (states or state):
        loc_label = (states[0] if states else state)
        queries.append(f"site:indeed.com internship {focus} {loc_label} {year}")

    if class_year:
        queries.append(f"site:joinhandshake.com {class_year} internship {focus} {year}")

    return queries


# ── Job queries ───────────────────────────────────────────────────────────────

def build_job_queries(profile: dict) -> list[str]:
    """
    Board-first full-time job discovery with a small open-web fallback.
    """
    year = _year_token()
    focus = _focus(profile)
    major = profile.get("major", "").strip()
    desired_role = profile.get("desired_role", "").strip()
    location_pref = profile.get("location_pref", "Any").strip()
    experience = profile.get("experience_level", "").strip()
    state = profile.get("state", "").strip()
    states = profile.get("states") or []

    queries = _board_site_queries(focus, year, kind="job")

    # Small open-web fallback
    if desired_role:
        queries.append(f"{desired_role} jobs hiring {year} apply")
    elif major:
        queries.append(f"{major} jobs entry level {year} apply")
    else:
        queries.append(f"entry level jobs hiring {year} apply")

    loc = location_pref.lower()
    if loc == "remote":
        queries.append(f"site:linkedin.com/jobs remote {focus} jobs {year}")
    elif loc in ("hybrid", "on-site") and (states or state):
        loc_label = (states[0] if states else state)
        queries.append(f"site:indeed.com {focus} jobs {loc_label} {year}")

    if experience:
        queries.append(f"site:glassdoor.com {experience} {focus} jobs {year}")

    return queries


# ── Shared search runner ─────────────────────────────────────────────────────

def _is_board_query(query: str) -> bool:
    return query.strip().lower().startswith("site:")


def _run_search(
    queries: list[str],
    max_results: int,
    max_total: int = MAX_TOTAL_URLS,
    *,
    board_max_results: int | None = None,
) -> list[str]:
    """
    Runs DDG searches, preferring board site: angles with deeper per-query depth,
    then ranks/dedupes so board/ATS URLs win over junk.
    """
    print(f"Running {len(queries)} search queries based on profile...")

    board_depth = board_max_results if board_max_results is not None else max(max_results, BOARD_MAX_RESULTS)
    open_depth = min(max_results, OPEN_WEB_MAX_RESULTS)

    # Process board queries first so they fill the budget before open-web noise.
    ordered = sorted(queries, key=lambda q: (0 if _is_board_query(q) else 1))

    raw_urls: list[str] = []
    seen = set()
    failed = 0
    board_hits = 0

    for query in ordered:
        depth = board_depth if _is_board_query(query) else open_depth
        print(f"  Searching: '{query}' (max {depth})")
        try:
            results = DDGS().text(query, max_results=depth)
            for r in results:
                href = r.get("href") or ""
                if not href or href in seen:
                    continue
                seen.add(href)
                raw_urls.append(href)
                if _is_board_url(href):
                    board_hits += 1
                print(f"    ↳ {r.get('title', href)}")
        except Exception as e:
            failed += 1
            print(f"  Search failed for '{query}': {e}")

    urls = _rank_and_cap_urls(raw_urls, max_total=max_total)
    kept_boards = sum(1 for u in urls if _is_board_url(u))
    print(
        f"\n✅ {len(urls)} unique URLs kept (cap {max_total}; "
        f"{kept_boards} board/ATS, {board_hits} board hits before cap)"
        f"{f' ({failed} angle(s) failed)' if failed else ''}."
    )
    return urls


def find_scholarship_urls(profile: dict, max_results: int = 5) -> list[str]:
    # Scholarship discovery stays open-web (not board-based).
    return _run_search(build_queries(profile), max_results, board_max_results=max_results)


def find_internship_urls(profile: dict, max_results: int = 5) -> list[str]:
    return _run_search(build_internship_queries(profile), max_results)


def find_job_urls(profile: dict, max_results: int = 5) -> list[str]:
    return _run_search(build_job_queries(profile), max_results)


if __name__ == "__main__":
    from pipeline import run_pipeline

    test_profile = {
        "major":        "data science",
        "state":        "NC",
        "ethnicity":    "white",
        "first_gen":    False,
        "income_based": False,
    }
    found_urls = find_scholarship_urls(test_profile, max_results=3)
    print(f"\nPassing {len(found_urls)} URLs to pipeline...\n")
    run_pipeline(found_urls, mode="scholarship")
