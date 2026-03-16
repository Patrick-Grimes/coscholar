from ddgs import DDGS

MAX_STATES_FOR_SEARCH = 4


# ── Scholarship queries ──────────────────────────────────────────────────────

def build_queries(profile: dict) -> list[str]:
    """
    Turns a student profile into a list of targeted DDG search queries.
    Each query targets a different eligibility angle so we don't miss
    scholarships that have nothing to do with major.
    """
    queries = []

    major     = profile.get("major", "").strip()
    state     = profile.get("state", "").strip()
    states    = profile.get("states") or []
    ethnicity = profile.get("ethnicity", "").strip()
    first_gen = profile.get("first_gen", False)
    income    = profile.get("income_based", False)

    # Major-specific (most targeted)
    if major:
        queries.append(f"{major} scholarships 2026 apply")

    # State-based (one query per selected state, capped)
    if states:
        search_states = [s.strip() for s in states if s.strip()][:MAX_STATES_FOR_SEARCH]
    elif state:
        search_states = [state]
    else:
        search_states = []

    for s in search_states:
        queries.append(f"{s} state scholarships undergraduate 2026")

    # Demographic / identity
    if ethnicity and ethnicity.lower() not in ("", "prefer not to say"):
        queries.append(f"{ethnicity} student scholarships 2026 apply")

    # First-gen
    if first_gen:
        queries.append("first generation college student scholarships 2026")

    # Need-based
    if income:
        queries.append("need based undergraduate scholarships 2026 apply")

    # Broad catch-all — finds scholarships with no major restriction
    queries.append("undergraduate scholarships 2026 no major restriction apply")

    return queries


# ── Internship queries ───────────────────────────────────────────────────────

def build_internship_queries(profile: dict) -> list[str]:
    """
    Turns a student profile into targeted DDG search queries for internships.
    """
    queries = []

    major        = profile.get("major", "").strip()
    state        = profile.get("state", "").strip()
    states       = profile.get("states") or []
    desired_role = profile.get("desired_role", "").strip()
    location_pref = profile.get("location_pref", "Any").strip()
    class_year   = profile.get("class_year", "").strip()

    # Role-specific
    if desired_role:
        queries.append(f"{desired_role} internship summer 2026 apply")

    # Major-specific
    if major:
        queries.append(f"{major} internship 2026 undergraduate")

    # Combined major + role
    if desired_role and major and desired_role.lower() != major.lower():
        queries.append(f"{major} {desired_role} internship 2026")

    # State-based (one query per selected state, capped)
    if states:
        search_states = [s.strip() for s in states if s.strip()][:MAX_STATES_FOR_SEARCH]
    elif state:
        search_states = [state]
    else:
        search_states = []

    for s in search_states:
        queries.append(f"internships in {s} undergraduate 2026")

    # Remote preference
    if location_pref.lower() == "remote":
        queries.append(f"remote internship {major or desired_role or 'undergraduate'} 2026")

    # Class year
    if class_year:
        queries.append(f"{class_year} student internship 2026 apply")

    # Broad catch-all
    queries.append("undergraduate internships summer 2026 apply")

    return queries


# ── Shared search runner ─────────────────────────────────────────────────────

def _run_search(queries: list[str], max_results: int) -> list[str]:
    """Runs DDG searches for a list of queries, returns deduplicated URLs."""
    print(f"Running {len(queries)} search queries based on profile...")

    seen = set()
    urls = []

    for query in queries:
        print(f"  Searching: '{query}'")
        try:
            results = DDGS().text(query, max_results=max_results)
            for r in results:
                if r["href"] not in seen:
                    seen.add(r["href"])
                    urls.append(r["href"])
                    print(f"    ↳ {r['title']}")
        except Exception as e:
            print(f"  Search failed for '{query}': {e}")

    print(f"\n✅ {len(urls)} unique URLs collected across all queries.")
    return urls


def find_scholarship_urls(profile: dict, max_results: int = 5) -> list[str]:
    return _run_search(build_queries(profile), max_results)


def find_internship_urls(profile: dict, max_results: int = 5) -> list[str]:
    return _run_search(build_internship_queries(profile), max_results)


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