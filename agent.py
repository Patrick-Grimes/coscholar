from datetime import date
from ddgs import DDGS

MAX_STATES_FOR_SEARCH = 4
MAX_TOTAL_URLS = 24


def _search_year() -> int:
    return date.today().year


def _year_token() -> str:
    """Current year for queries; include next year during fall recruiting season."""
    y = _search_year()
    if date.today().month >= 9:
        return f"{y} OR {y + 1}"
    return str(y)


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
    Turns a student profile into targeted DDG search queries for internships.
    """
    queries = []
    year = _year_token()

    major         = profile.get("major", "").strip()
    state         = profile.get("state", "").strip()
    states        = profile.get("states") or []
    desired_role  = profile.get("desired_role", "").strip()
    location_pref = profile.get("location_pref", "Any").strip()
    class_year    = profile.get("class_year", "").strip()

    if desired_role:
        queries.append(f"{desired_role} internship summer {year} apply")

    if major:
        queries.append(f"{major} internship {year} undergraduate")

    if desired_role and major and desired_role.lower() != major.lower():
        queries.append(f"{major} {desired_role} internship {year}")

    if states:
        search_states = [s.strip() for s in states if s.strip()][:MAX_STATES_FOR_SEARCH]
    elif state:
        search_states = [state]
    else:
        search_states = []

    for s in search_states:
        queries.append(f"internships in {s} undergraduate {year}")

    loc = location_pref.lower()
    focus = major or desired_role or "undergraduate"
    if loc == "remote":
        queries.append(f"remote internship {focus} {year}")
    elif loc == "hybrid":
        queries.append(f"hybrid internship {focus} {year}")
    elif loc == "on-site":
        queries.append(f"on-site in-person internship {focus} {year}")

    if class_year:
        queries.append(f"{class_year} student internship {year} apply")

    queries.append(f"undergraduate internships summer {year} apply")

    return queries


# ── Job queries ───────────────────────────────────────────────────────────────

def build_job_queries(profile: dict) -> list[str]:
    """
    Turns a candidate profile into targeted DDG search queries for full-time jobs.
    """
    queries = []
    year = _year_token()

    major         = profile.get("major", "").strip()
    state         = profile.get("state", "").strip()
    states        = profile.get("states") or []
    desired_role  = profile.get("desired_role", "").strip()
    location_pref = profile.get("location_pref", "Any").strip()
    experience    = profile.get("experience_level", "").strip()

    if desired_role:
        queries.append(f"{desired_role} jobs hiring {year} apply")

    if major:
        queries.append(f"{major} jobs entry level {year} apply")

    if desired_role and major and desired_role.lower() != major.lower():
        queries.append(f"{major} {desired_role} jobs {year}")

    if states:
        search_states = [s.strip() for s in states if s.strip()][:MAX_STATES_FOR_SEARCH]
    elif state:
        search_states = [state]
    else:
        search_states = []

    for s in search_states:
        queries.append(f"jobs in {s} {desired_role or major or 'professional'} {year}")

    loc = location_pref.lower()
    focus = desired_role or major or "software"
    if loc == "remote":
        queries.append(f"remote {focus} jobs {year} apply")
    elif loc == "hybrid":
        queries.append(f"hybrid {focus} jobs {year} apply")
    elif loc == "on-site":
        queries.append(f"on-site in-person {focus} jobs {year} apply")

    if experience:
        queries.append(f"{experience} {desired_role or major or 'professional'} jobs {year} apply")

    queries.append(f"entry level jobs hiring {year} apply")

    return queries


# ── Shared search runner ─────────────────────────────────────────────────────

def _run_search(queries: list[str], max_results: int, max_total: int = MAX_TOTAL_URLS) -> list[str]:
    """Runs DDG searches for a list of queries, returns deduplicated URLs."""
    print(f"Running {len(queries)} search queries based on profile...")

    seen = set()
    urls = []
    failed = 0

    for query in queries:
        if len(urls) >= max_total:
            break
        print(f"  Searching: '{query}'")
        try:
            results = DDGS().text(query, max_results=max_results)
            for r in results:
                if len(urls) >= max_total:
                    break
                href = r.get("href") or ""
                if href and href not in seen:
                    seen.add(href)
                    urls.append(href)
                    print(f"    ↳ {r.get('title', href)}")
        except Exception as e:
            failed += 1
            print(f"  Search failed for '{query}': {e}")

    print(f"\n✅ {len(urls)} unique URLs collected across all queries"
          f"{f' ({failed} angle(s) failed)' if failed else ''}.")
    return urls


def find_scholarship_urls(profile: dict, max_results: int = 5) -> list[str]:
    return _run_search(build_queries(profile), max_results)


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
