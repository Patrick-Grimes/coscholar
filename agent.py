from ddgs import DDGS
from pipeline import run_scholarship_pipeline


def build_queries(profile: dict) -> list[str]:
    """
    Turns a student profile into a list of targeted DDG search queries.
    Each query targets a different eligibility angle so we don't miss
    scholarships that have nothing to do with major.
    """
    queries = []

    major     = profile.get("major", "").strip()
    state     = profile.get("state", "").strip()
    ethnicity = profile.get("ethnicity", "").strip()
    first_gen = profile.get("first_gen", False)
    income    = profile.get("income_based", False)

    # Major-specific (most targeted)
    if major:
        queries.append(f"{major} scholarships 2026 apply")

    # State-based (huge category, often overlooked)
    if state:
        queries.append(f"{state} state scholarships undergraduate 2026")

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


def find_scholarship_urls(profile: dict, max_results: int = 5) -> list[str]:
    """
    Runs multiple DDG searches based on the student's full profile
    and returns a deduplicated list of URLs.
    """
    queries = build_queries(profile)
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


if __name__ == "__main__":
    test_profile = {
        "major":        "data science",
        "state":        "NC",
        "ethnicity":    "white",
        "first_gen":    False,
        "income_based": False,
    }
    found_urls = find_scholarship_urls(test_profile, max_results=3)
    print(f"\nPassing {len(found_urls)} URLs to pipeline...\n")
    run_scholarship_pipeline(found_urls)