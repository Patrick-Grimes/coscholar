import streamlit as st
import pandas as pd
import os
import datetime
import html
import pdfplumber
import io
from io import StringIO

from agent import find_scholarship_urls, find_internship_urls, find_job_urls
from pipeline import run_pipeline
from drafter import draft_application
from dotenv import load_dotenv

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CoScholar AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Mode helpers ──────────────────────────────────────────────────────────────
# Career modes first — Scholarship remains available as a third target.
MODES = ["Job", "Internship", "Scholarship"]
MODE_KEYS = {"Scholarship": "scholarship", "Internship": "internship", "Job": "job"}
MODE_NOUNS = {"scholarship": "scholarships", "internship": "internships", "job": "jobs"}
MODE_ICONS = {"Scholarship": "🎓", "Internship": "💼", "Job": "🏢"}
MODE_TITLES = {
    "Scholarship": "CoScholar **AI**",
    "Internship":  "CoIntern **AI**",
    "Job":         "CoJobs **AI**",
}
MODE_SUBTITLES = {
    "Scholarship": "Profile → Find scholarships → Draft letters.",
    "Internship":  "Profile → Find internships → Draft cover letters.",
    "Job":         "Profile → Find jobs → Draft cover letters.",
}
DEFAULT_MODE = "Job"
ROLE_STOPWORDS = {
    "the", "a", "an", "and", "or", "for", "of", "to", "in", "at", "on",
    "intern", "internship", "job", "jobs", "role", "position", "associate",
}

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
.stDeployButton { display: none; }
[data-testid="stHeader"] { background: transparent; }

[data-testid="metric-container"] {
    background: #1a1d27;
    border: 1px solid #2d3148;
    border-radius: 10px;
    padding: 1rem 1.25rem;
}
.tag {
    display: inline-block;
    font-family: monospace;
    font-size: 0.7rem;
    background: #4ade8020;
    color: #4ade80;
    border: 1px solid #4ade8040;
    border-radius: 4px;
    padding: 1px 8px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.step-title { font-size: 1.3rem; font-weight: 700; margin: 0.25rem 0 0.2rem 0; }
.step-sub   { color: #64748b; font-size: 0.875rem; margin-bottom: 1.25rem; }
.draft-body { font-size: 0.9rem; line-height: 1.7; color: #cbd5e1; white-space: pre-wrap; }
.slabel {
    font-size: 0.65rem; font-family: monospace; letter-spacing: 0.12em;
    text-transform: uppercase; color: #475569; margin: 1.1rem 0 0.3rem 0;
}
.status-loaded {
    background: #4ade8015; border: 1px solid #4ade8030; border-radius: 8px;
    padding: 0.6rem 1rem; font-size: 0.85rem; color: #4ade80; margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "min_gpa" in df.columns:
        df["min_gpa"] = pd.to_numeric(df["min_gpa"], errors="coerce").fillna(0.0)
    return df


MAJOR_ALIASES = {
    "computer science": ["cs", "comp sci", "computer science", "computing", "software"],
    "data science":     ["data science", "data analytics", "data analysis", "analytics"],
    "engineering":      ["engineering", "engineer"],
    "business":         ["business", "finance", "accounting", "economics", "econ"],
    "biology":          ["biology", "bio", "life science", "biological"],
    "psychology":       ["psychology", "psych"],
    "nursing":          ["nursing", "nurse", "rn"],
    "math":             ["math", "mathematics", "statistics", "stats"],
}

US_STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
]

def major_terms(major: str) -> list[str]:
    m = major.lower().strip()
    for key, aliases in MAJOR_ALIASES.items():
        if m in aliases or m == key:
            # Prefer longer aliases first to reduce short-token false hits.
            return sorted(set(aliases + [key]), key=len, reverse=True)
    return [m] if m else []


def role_terms(desired_role: str) -> list[str]:
    import re
    tokens = re.findall(r"[a-z0-9]+", (desired_role or "").lower())
    return [t for t in tokens if t not in ROLE_STOPWORDS and len(t) > 1]


def _col(df, name):
    return df.get(name, pd.Series([""] * len(df))).astype(str).str.strip()


def _is_unknown(series: pd.Series) -> pd.Series:
    """Empty / missing eligibility — unknown, not unrestricted."""
    s = series.fillna("").astype(str).str.strip()
    return s.isin(["", "[]", "nan", "None", "none", "null", "NaN"])


def _is_explicit_all(series: pd.Series) -> pd.Series:
    """Page explicitly said unrestricted (extractor uses ALL / nationwide markers)."""
    s = series.fillna("").astype(str).str.lower()
    return (
        s.str.contains(r"\ball\b", regex=True)
        | s.str.contains("nationwide", regex=False)
        | s.str.contains("any major", regex=False)
        | s.str.contains("open to all", regex=False)
    )


def _field_match_mask(series: pd.Series, terms: list[str], *, allow_unknown: bool = True) -> tuple[pd.Series, pd.Series]:
    """
    Returns (hard_mask, explicit_match_mask).
    Unknown fields soft-pass when allow_unknown; explicit ALL always passes;
    otherwise require a term hit. Explicit mismatches are dropped.
    """
    unknown = _is_unknown(series)
    explicit_all = _is_explicit_all(series)
    hit = pd.Series([False] * len(series), index=series.index)
    for term in terms:
        if not term:
            continue
        hit = hit | series.str.contains(term, case=False, regex=False)
    if allow_unknown:
        # Soft-pass unknowns; drop known mismatches (content present, not ALL, no hit).
        hard = unknown | explicit_all | hit
    else:
        hard = explicit_all | hit
    return hard, hit


def _location_mask(loc_col: pd.Series, location_pref: str, states: list[str]) -> pd.Series:
    """
    Location preference + states.
    On-site = not Remote/Hybrid and has a location.
    When both pref and states are set, require AND (e.g. Remote in those states).
    """
    lower = loc_col.str.lower()
    is_remote = lower.str.contains("remote", regex=False)
    is_hybrid = lower.str.contains("hybrid", regex=False)
    unknown = _is_unknown(loc_col)

    pref = (location_pref or "Any").strip()
    if pref == "Remote":
        pref_mask = is_remote | unknown
    elif pref == "Hybrid":
        pref_mask = is_hybrid | unknown
    elif pref == "On-site":
        pref_mask = (~is_remote & ~is_hybrid & ~unknown) | unknown
        # Prefer concrete places when known; unknown soft-passes but ranks lower later.
    else:
        pref_mask = pd.Series([True] * len(loc_col), index=loc_col.index)

    if states:
        state_mask = pd.Series([False] * len(loc_col), index=loc_col.index)
        for s in states:
            s_clean = (s or "").strip()
            if not s_clean:
                continue
            # Word-ish match for 2-letter codes to reduce "AL" in "California".
            if len(s_clean) == 2:
                state_mask = state_mask | loc_col.str.contains(
                    rf"(?:^|[^A-Za-z]){s_clean}(?:[^A-Za-z]|$)", case=False, regex=True
                )
            else:
                state_mask = state_mask | loc_col.str.contains(s_clean, case=False, regex=False)
        # Remote nationwide listings often omit a state — allow unknown/pure-remote
        # only when preference is Remote and no concrete conflicting state is required.
        if pref == "Remote":
            state_mask = state_mask | (is_remote & unknown) | (lower.isin(["remote", "remote, usa", "remote - usa"]))
        if pref and pref != "Any":
            return pref_mask & (state_mask | unknown)
        return state_mask | unknown

    return pref_mask


def _gpa_mask(df: pd.DataFrame, gpa: float) -> pd.Series:
    gpa_col = pd.to_numeric(df.get("min_gpa", pd.Series(index=df.index)), errors="coerce")
    # Missing requirement → pass; explicit requirement must be <= user GPA.
    return gpa_col.isna() | (gpa_col <= gpa)


def _role_mask(df: pd.DataFrame, desired_role: str) -> tuple[pd.Series, pd.Series]:
    if not (desired_role or "").strip():
        true = pd.Series([True] * len(df), index=df.index)
        false = pd.Series([False] * len(df), index=df.index)
        return true, false
    role_col = _col(df, "role")
    terms = role_terms(desired_role)
    if not terms:
        true = pd.Series([True] * len(df), index=df.index)
        false = pd.Series([False] * len(df), index=df.index)
        return true, false
    return _field_match_mask(role_col, terms, allow_unknown=True)


def _score_matches(
    df: pd.DataFrame,
    *,
    mode: str,
    major: str,
    desired_role: str = "",
) -> pd.DataFrame:
    """Rank matches: explicit field hits beat unknowns; prefer apply_url + description."""
    if df is None or df.empty:
        return df

    out = df.copy()
    scores = pd.Series([0.0] * len(out), index=out.index)

    major_col = _col(out, "majors" if mode == "scholarship" else "preferred_majors")
    terms = major_terms(major)
    _, major_hit = _field_match_mask(major_col, terms, allow_unknown=True)
    scores = scores + major_hit.astype(float) * 3.0
    scores = scores + _is_explicit_all(major_col).astype(float) * 1.0
    # Penalize unknown majors slightly so explicit matches float up.
    scores = scores - _is_unknown(major_col).astype(float) * 1.5

    if mode in ("internship", "job") and desired_role.strip():
        _, role_hit = _role_mask(out, desired_role)
        scores = scores + role_hit.astype(float) * 5.0
        scores = scores - _is_unknown(_col(out, "role")).astype(float) * 1.0

    apply_col = _col(out, "apply_url")
    scores = scores + (~_is_unknown(apply_col) & apply_col.str.startswith("http")).astype(float) * 2.0

    desc_col = _col(out, "description")
    scores = scores + (~_is_unknown(desc_col)).astype(float) * 1.0

    gpa_col = pd.to_numeric(out.get("min_gpa", pd.Series(index=out.index)), errors="coerce")
    scores = scores + gpa_col.fillna(0).clip(lower=0, upper=4) * 0.1

    if mode == "scholarship" and "amount" in out.columns:
        amt = pd.to_numeric(out["amount"], errors="coerce").fillna(0)
        scores = scores + (amt.clip(upper=50000) / 10000.0)

    out["match_score"] = scores
    out = out.sort_values("match_score", ascending=False)
    return out


def filter_scholarship_matches(
    df: pd.DataFrame,
    major: str,
    gpa: float,
    states: list[str],
    ethnicity: str = "",
    first_gen: bool = False,
    income_based: bool = False,
) -> pd.DataFrame:
    if df is None or df.empty:
        return df.copy() if df is not None else pd.DataFrame()

    gpa_mask = _gpa_mask(df, gpa)

    major_col = _col(df, "majors")
    major_mask, _ = _field_match_mask(major_col, major_terms(major), allow_unknown=True)

    state_col = _col(df, "eligible_states")
    if states:
        state_mask = _is_unknown(state_col) | _is_explicit_all(state_col)
        for s in states:
            s_clean = (s or "").strip()
            if not s_clean:
                continue
            if len(s_clean) == 2:
                state_mask = state_mask | state_col.str.contains(
                    rf"(?:^|[^A-Za-z]){s_clean}(?:[^A-Za-z]|$)", case=False, regex=True
                )
            else:
                state_mask = state_mask | state_col.str.contains(s_clean, case=False, regex=False)
    else:
        state_mask = pd.Series([True] * len(df), index=df.index)

    eth_col = _col(df, "ethnicity")
    if ethnicity:
        eth_mask = (
            _is_unknown(eth_col)
            | _is_explicit_all(eth_col)
            | eth_col.str.contains(ethnicity, case=False, regex=False)
        )
    else:
        eth_mask = pd.Series([True] * len(df), index=df.index)

    fg_col = _col(df, "first_gen")
    first_gen_mask = (
        ~fg_col.str.lower().isin(["true", "1"])
        | pd.Series([first_gen] * len(df), index=df.index)
    )

    inc_col = _col(df, "income_based")
    income_mask = (
        ~inc_col.str.lower().isin(["true", "1"])
        | pd.Series([income_based] * len(df), index=df.index)
    )

    mask = gpa_mask & major_mask & state_mask & eth_mask & first_gen_mask & income_mask
    return _score_matches(df[mask], mode="scholarship", major=major)


def filter_internship_matches(
    df: pd.DataFrame,
    major: str,
    gpa: float,
    states: list[str],
    location_pref: str = "Any",
    class_year: str = "",
    desired_role: str = "",
) -> pd.DataFrame:
    if df is None or df.empty:
        return df.copy() if df is not None else pd.DataFrame()

    gpa_mask = _gpa_mask(df, gpa)

    major_col = _col(df, "preferred_majors")
    major_mask, _ = _field_match_mask(major_col, major_terms(major), allow_unknown=True)

    location_mask = _location_mask(_col(df, "location"), location_pref, states)

    if class_year:
        cy_col = _col(df, "class_year")
        class_year_mask, _ = _field_match_mask(cy_col, [class_year], allow_unknown=True)
    else:
        class_year_mask = pd.Series([True] * len(df), index=df.index)

    role_mask, _ = _role_mask(df, desired_role)

    mask = gpa_mask & major_mask & location_mask & class_year_mask & role_mask
    return _score_matches(df[mask], mode="internship", major=major, desired_role=desired_role)


def filter_job_matches(
    df: pd.DataFrame,
    major: str,
    gpa: float,
    states: list[str],
    location_pref: str = "Any",
    experience_level: str = "",
    desired_role: str = "",
) -> pd.DataFrame:
    if df is None or df.empty:
        return df.copy() if df is not None else pd.DataFrame()

    gpa_mask = _gpa_mask(df, gpa)

    major_col = _col(df, "preferred_majors")
    major_mask, _ = _field_match_mask(major_col, major_terms(major), allow_unknown=True)

    location_mask = _location_mask(_col(df, "location"), location_pref, states)

    if experience_level:
        exp_col = _col(df, "experience_level")
        experience_mask, _ = _field_match_mask(exp_col, [experience_level], allow_unknown=True)
    else:
        experience_mask = pd.Series([True] * len(df), index=df.index)

    role_mask, _ = _role_mask(df, desired_role)

    mask = gpa_mask & major_mask & location_mask & experience_mask & role_mask
    return _score_matches(df[mask], mode="job", major=major, desired_role=desired_role)


# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [
    ("mode",            DEFAULT_MODE),
    ("listings_df",     None),
    ("matches_df",      None),
    ("drafts",          {}),
    ("is_scouting",     False),
    ("is_drafting",     False),
    ("scout_done",      False),
]:
    if key not in st.session_state:
        st.session_state[key] = default
    # Migrate older sessions that defaulted to Scholarship-only product copy.
    if key == "mode" and st.session_state[key] not in MODES:
        st.session_state[key] = DEFAULT_MODE


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## CoScholar")
    st.caption("Find a job or internship → match your profile → draft a cover letter.")
    st.caption("Results stay in this browser session only.")
    st.divider()

    # Mode toggle — Jobs & Internships first; Scholarship still available.
    st.markdown('<div class="slabel">I want</div>', unsafe_allow_html=True)
    mode_index = MODES.index(st.session_state.mode) if st.session_state.mode in MODES else 0
    new_mode = st.radio(
        "I want",
        MODES,
        index=mode_index,
        horizontal=True,
        label_visibility="collapsed",
        help="Jobs and Internships share the career pipeline. Scholarship keeps the funding flow.",
    )
    if new_mode != st.session_state.mode:
        st.session_state.mode = new_mode
        st.session_state.listings_df = None
        st.session_state.matches_df  = None
        st.session_state.drafts      = {}
        st.session_state.scout_done  = False
        load_csv.clear()
        st.rerun()

    mode = st.session_state.mode
    mode_key = MODE_KEYS[mode]  # "scholarship", "internship", or "job"
    mode_noun = MODE_NOUNS[mode_key]

    st.divider()

    # Provider + API key
    st.markdown('<div class="slabel">AI Provider</div>', unsafe_allow_html=True)
    provider = st.selectbox(
        "AI Provider",
        ["Gemini", "Claude", "OpenAI", "Ollama"],
        label_visibility="collapsed",
    )

    _KEY_HINTS = {
        "Gemini": ("Paste your key from aistudio.google.com", "🔑 Get a free key at [aistudio.google.com](https://aistudio.google.com)"),
        "Claude": ("Paste your key from console.anthropic.com", "🔑 Get a key at [console.anthropic.com](https://console.anthropic.com)"),
        "OpenAI": ("Paste your key from platform.openai.com", "🔑 Get a key at [platform.openai.com](https://platform.openai.com/api-keys)"),
    }

    if provider == "Ollama":
        st.markdown('<div class="slabel">Ollama Host URL</div>', unsafe_allow_html=True)
        ollama_host = st.text_input(
            "Ollama Host",
            placeholder="http://localhost:11434",
            label_visibility="collapsed",
        )
        user_api_key = None
        st.caption("Ollama runs locally — no API key needed. [Get Ollama](https://ollama.com)")
    else:
        ollama_host = None
        placeholder, hint = _KEY_HINTS[provider]
        st.markdown(f'<div class="slabel">{provider} API Key</div>', unsafe_allow_html=True)
        user_api_key = st.text_input(
            f"{provider} API Key",
            type="password",
            placeholder=placeholder,
            label_visibility="collapsed",
        )
        if user_api_key:
            st.success("✅ Key set for this session")
        else:
            st.caption(hint)

    st.divider()

    # ── Profile (mode-dependent) ─────────────────────────────────────────────
    st.markdown('<div class="slabel">Your profile</div>', unsafe_allow_html=True)
    name  = st.text_input("Full Name", placeholder="Jane Smith", max_chars=100,
                          help="Used in cover-letter sign-off. Optional for search.")
    major = st.text_input("Major / Field", placeholder="Computer Science", max_chars=100)
    gpa   = st.number_input("GPA", min_value=0.0, max_value=5.0, step=0.1, format="%.1f",
                            help="Supports 4.0 and 5.0 scales.")
    states = st.multiselect("States (2-letter)", options=US_STATES, default=[], placeholder="Select one or more states")
    states = [s.upper() for s in states]
    primary_state = states[0] if states else ""

    if mode_key == "scholarship":
        ethnicity    = st.text_input("Ethnicity (optional)", placeholder="e.g. Hispanic, Black, Asian, white", max_chars=50)
        first_gen    = st.checkbox("First-generation college student")
        income_based = st.checkbox("Financial need / income-based")
        desired_role = ""
        location_pref = "Any"
        class_year = ""
        experience_level = ""
    elif mode_key == "internship":
        ethnicity = ""
        first_gen = False
        income_based = False
        experience_level = ""
        desired_role  = st.text_input("Desired Role / Title", placeholder="Software Engineering Intern", max_chars=100)
        location_pref = st.selectbox(
            "Location Preference",
            ["Any", "Remote", "Hybrid", "On-site"],
            help="On-site means not Remote/Hybrid. Remote + states requires both.",
        )
        class_year    = st.selectbox("Class Year", ["", "Freshman", "Sophomore", "Junior", "Senior"])
    else:  # job
        ethnicity = ""
        first_gen = False
        income_based = False
        class_year = ""
        desired_role  = st.text_input("Desired Role / Title", placeholder="Software Engineer", max_chars=100)
        location_pref = st.selectbox(
            "Location Preference",
            ["Any", "Remote", "Hybrid", "On-site"],
            help="On-site means not Remote/Hybrid. Remote + states requires both.",
        )
        experience_level = st.selectbox(
            "Experience Level",
            ["", "Entry-level", "New Grad", "Mid-level", "Senior"],
        )

    # Auto-filter on profile change
    if st.session_state.listings_df is not None and major.strip() and gpa > 0:
        if mode_key == "scholarship":
            st.session_state.matches_df = filter_scholarship_matches(
                st.session_state.listings_df, major, gpa, states,
                ethnicity, first_gen, income_based,
            )
        elif mode_key == "internship":
            st.session_state.matches_df = filter_internship_matches(
                st.session_state.listings_df, major, gpa, states,
                location_pref, class_year, desired_role=desired_role,
            )
        else:
            st.session_state.matches_df = filter_job_matches(
                st.session_state.listings_df, major, gpa, states,
                location_pref, experience_level, desired_role=desired_role,
            )

    st.divider()

    # Supplemental info
    st.markdown('<div class="slabel">Supplemental Information</div>', unsafe_allow_html=True)
    st.caption(
        "Upload files such as resumes, cover letters, or previous applications, "
        "and paste anything else in the text box — the more detail, the better the drafts!"
    )

    uploaded_files = st.file_uploader(
        "Upload resume(s) (.txt or .pdf)",
        type=["txt", "pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    MAX_FILE_SIZE = 5 * 1024 * 1024
    PDF_MAGIC = b"%PDF"

    if "uploaded_file_cache" not in st.session_state:
        st.session_state.uploaded_file_cache = {}

    uploaded_text = ""
    for f in uploaded_files:
        cache_key = f"{f.name}_{f.size}"

        if cache_key in st.session_state.uploaded_file_cache:
            uploaded_text += st.session_state.uploaded_file_cache[cache_key] + "\n\n"
            st.success(f"✅ {f.name}")
            continue

        if f.size > MAX_FILE_SIZE:
            st.warning(f"⚠️ {f.name} exceeds 5 MB limit — skipped.")
            continue

        raw_bytes = f.read()

        if f.type == "text/plain":
            text = raw_bytes.decode("utf-8", errors="ignore")
            st.session_state.uploaded_file_cache[cache_key] = text
            uploaded_text += text + "\n\n"
            st.success(f"✅ {f.name}")
        elif f.type == "application/pdf":
            if not raw_bytes[:4].startswith(PDF_MAGIC):
                st.warning(f"⚠️ {f.name} doesn't appear to be a valid PDF — skipped.")
                continue
            try:
                with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
                    text = "\n".join(p.extract_text() or "" for p in pdf.pages)
                st.session_state.uploaded_file_cache[cache_key] = text
                uploaded_text += text + "\n\n"
                st.success(f"✅ {f.name}")
            except Exception as e:
                st.warning(f"PDF parse error ({f.name}): {e}")

    pasted_text = st.text_area(
        "Or paste your profile here",
        placeholder=(
            "Paste any other relevant info here, including:\n"
            "• Key projects & technical skills\n"
            "• Extracurriculars, interests, goals"
        ),
        height=180,
        max_chars=10000,
        label_visibility="collapsed",
    )

    resume_text = "\n\n".join(filter(None, [uploaded_text.strip(), pasted_text.strip()]))

    st.markdown('<div class="slabel">Search settings</div>', unsafe_allow_html=True)
    max_results = st.slider(
        "Results per search angle",
        1, 10, 3,
        help="Each profile angle runs its own search. Total URLs are capped around 24.",
    )

    # Session status
    st.divider()
    if st.session_state.listings_df is not None:
        count = len(st.session_state.listings_df)
        st.markdown(
            f'<div class="status-loaded">📂 Session results<br>'
            f'<span style="opacity:0.7">{count} {mode_noun} in this browser session</span></div>',
            unsafe_allow_html=True,
        )
        if st.button("🔁 Reset Session Data", use_container_width=True, key="btn_reset_session"):
            st.session_state.listings_df = None
            st.session_state.matches_df  = None
            st.session_state.drafts      = {}
            st.session_state.scout_done  = False
            load_csv.clear()
            st.rerun()
    else:
        st.caption(f"No {mode_noun} yet. Run a Scout to build results for this session.")


# ── Header ────────────────────────────────────────────────────────────────────
title_col, pill_col = st.columns([5, 1])
with title_col:
    st.markdown(f"# {MODE_TITLES[mode]}")
    st.caption(MODE_SUBTITLES[mode])
    st.caption("① Profile → ② Scout matches → ③ Draft cover letters")
with pill_col:
    if st.session_state.listings_df is not None:
        n = len(st.session_state.listings_df)
        st.markdown(
            f'<div style="text-align:right;padding-top:1.25rem">'
            f'<span class="tag">🗄 {n} in DB</span></div>',
            unsafe_allow_html=True,
        )

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
_ENV_KEYS = {
    "Gemini": os.getenv("GEMINI_API_KEY") or os.getenv("API_KEY") or "",
    "Claude": os.getenv("ANTHROPIC_API_KEY") or "",
    "OpenAI": os.getenv("OPENAI_API_KEY") or "",
    "Ollama": "",
}
active_api_key = user_api_key or _ENV_KEYS.get(provider, "") or ""
api_key_set    = (provider == "Ollama") or bool(active_api_key)
# Name is for drafts/sign-off — not required to scout.
profile_ready  = bool(major.strip()) and gpa > 0
if mode_key in ("internship", "job"):
    profile_ready = profile_ready  # desired_role optional but improves ranking

# The name column used to identify a listing in drafts/display
_name_col = "name" if mode_key == "scholarship" else "company"
_is_career_mode = mode_key in ("internship", "job")
draft_profile = {
    "major": major,
    "gpa": gpa,
    "states": states,
    "desired_role": desired_role,
    "location_pref": location_pref,
    "class_year": class_year,
    "experience_level": experience_level,
    "ethnicity": ethnicity,
}

tab1, tab2, tab3 = st.tabs(["  🔍  Scout  ", "  📋  Matches  ", "  ✏️  Drafts  "])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — SCOUT
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="tag">Step 01</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="step-title">Scout {mode_noun.title()}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="step-sub">Searches the web across multiple dimensions, '
        'scrapes each page with AI, and saves results to your local database.</div>',
        unsafe_allow_html=True,
    )

    if not api_key_set:
        st.warning(f"👈  Paste your {provider} API key in the sidebar to get started.")
    elif not profile_ready:
        if st.session_state.listings_df is not None:
            st.info(
                f"📂 **{len(st.session_state.listings_df)} {mode_noun} already in this session.** "
                "👈 Fill in Major and GPA — matches update as you refine your profile."
            )
        else:
            st.info("👈  Fill in your Major and GPA in the sidebar to get started.")
    else:
        if mode_key == "scholarship":
            angles_preview = sum([bool(major), bool(states), bool(ethnicity), first_gen, income_based, True])
        elif mode_key == "internship":
            angles_preview = sum([bool(desired_role), bool(major), bool(states), location_pref == "Remote", bool(class_year), True])
        else:
            angles_preview = sum([bool(desired_role), bool(major), bool(states), location_pref == "Remote", bool(experience_level), True])
        est_min = max(1, round((angles_preview * max_results * 4) / 60))
        st.caption(
            f"⏱ Estimated time: **{est_min}–{est_min + 1} min** "
            f"({angles_preview} search angles × {max_results} URLs each, ~4s per site). "
            "Feel free to grab a coffee ☕"
        )

        btn_col, _ = st.columns([2, 3])
        with btn_col:
            scout_clicked = st.button(
                "🚀  Start Scout",
                type="primary",
                disabled=st.session_state.is_scouting,
                use_container_width=True,
                key="btn_scout",
            )

        if scout_clicked:
            st.session_state.is_scouting = True
            st.rerun()

        if st.session_state.is_scouting:
            with st.status("Scouting the web — this may take a few minutes...", expanded=True) as status:
                state_label = ", ".join(states)
                if mode_key == "scholarship":
                    angles = []
                    if major:        angles.append(f"major ({major})")
                    if state_label:  angles.append(f"states ({state_label})")
                    if ethnicity:    angles.append(f"ethnicity ({ethnicity})")
                    if first_gen:    angles.append("first-generation")
                    if income_based: angles.append("need-based")
                    angles.append("general")

                    profile = {
                        "major": major,
                        "state": primary_state,
                        "states": states,
                        "ethnicity": ethnicity,
                        "first_gen": first_gen,
                        "income_based": income_based,
                    }
                elif mode_key == "internship":
                    angles = []
                    if desired_role: angles.append(f"role ({desired_role})")
                    if major:        angles.append(f"major ({major})")
                    if state_label:  angles.append(f"states ({state_label})")
                    if location_pref == "Remote": angles.append("remote")
                    if class_year:   angles.append(f"class year ({class_year})")
                    angles.append("general")

                    profile = {
                        "major": major,
                        "state": primary_state,
                        "states": states,
                        "desired_role": desired_role,
                        "location_pref": location_pref,
                        "class_year": class_year,
                    }
                else:
                    angles = []
                    if desired_role: angles.append(f"role ({desired_role})")
                    if major:        angles.append(f"field ({major})")
                    if state_label:  angles.append(f"states ({state_label})")
                    if location_pref == "Remote": angles.append("remote")
                    if experience_level: angles.append(f"experience ({experience_level})")
                    angles.append("general")

                    profile = {
                        "major": major,
                        "state": primary_state,
                        "states": states,
                        "desired_role": desired_role,
                        "location_pref": location_pref,
                        "experience_level": experience_level,
                    }

                st.write(f"🔎 Running **{len(angles)} searches:** {', '.join(angles)}")

                urls = None
                try:
                    if mode_key == "scholarship":
                        urls = find_scholarship_urls(profile, max_results=max_results)
                    elif mode_key == "internship":
                        urls = find_internship_urls(profile, max_results=max_results)
                    else:
                        urls = find_job_urls(profile, max_results=max_results)
                except Exception as e:
                    st.session_state.is_scouting = False
                    status.update(label=f"❌ Search failed: {e}", state="error", expanded=False)

                if urls is not None and not urls:
                    st.session_state.is_scouting = False
                    status.update(
                        label="⚠️ No URLs found. Try increasing URLs to Scout in the sidebar.",
                        state="error", expanded=False,
                    )
                    urls = None

                if not urls:
                    st.rerun()

                st.write(f"📡 Found **{len(urls)} unique URLs**. Scraping with {provider}...")

                counter_box  = st.empty()
                progress_bar = st.progress(0)

                def ui_callback(url, site_num, total_sites, found_so_far, done=False):
                    progress_bar.progress(site_num / total_sites)
                    if done:
                        counter_box.markdown(
                            f"✅ **All {total_sites} sites processed** · "
                            f"**{found_so_far} active {mode_noun}** found"
                        )
                    else:
                        counter_box.markdown(
                            f"🌐 **Site {site_num} of {total_sites}** · "
                            f"**{found_so_far} {mode_noun}** found so far"
                        )

                df = run_pipeline(
                    urls,
                    mode=mode_key,
                    progress_callback=ui_callback,
                    provider=provider,
                    api_key=active_api_key,
                    ollama_host=ollama_host,
                )
                st.session_state.is_scouting = False

                if df is not None and not df.empty:
                    if mode_key == "scholarship":
                        matches = filter_scholarship_matches(df, major, gpa, states, ethnicity, first_gen, income_based)
                    elif mode_key == "internship":
                        matches = filter_internship_matches(
                            df, major, gpa, states, location_pref, class_year, desired_role=desired_role,
                        )
                    else:
                        matches = filter_job_matches(
                            df, major, gpa, states, location_pref, experience_level, desired_role=desired_role,
                        )

                    st.session_state.listings_df = df
                    st.session_state.matches_df  = matches
                    st.session_state.scout_done  = True

                    if len(df) == 0:
                        status.update(
                            label=f"⚠️ Sites scraped but 0 active {mode_noun} found. Try different search terms.",
                            state="error", expanded=False,
                        )
                    else:
                        status.update(
                            label=f"✅ Scout complete — {len(df)} {mode_noun} found · {len(matches)} match your profile.",
                            state="complete", expanded=False,
                        )
                else:
                    status.update(
                        label=f"⚠️ No active {mode_noun} found. Try different search terms.",
                        state="error",
                        expanded=False,
                    )

    # Database display
    if st.session_state.listings_df is not None:
        df = st.session_state.listings_df
        if mode_key == "internship" and "paid" in df.columns:
            df_display = df.drop(columns=["paid"])
        else:
            df_display = df
        st.markdown("---")

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Scraped", len(df))
        c2.metric(
            "Matching Profile",
            len(st.session_state.matches_df) if st.session_state.matches_df is not None else "—",
        )
        if mode_key == "scholarship":
            avg_amt = pd.to_numeric(df.get("amount", pd.Series()), errors="coerce").mean()
            c3.metric("Avg Award", f"${int(avg_amt):,}" if not pd.isna(avg_amt) else "—")
        elif mode_key == "internship":
            paid_col = _col(df, "paid")
            n_paid = paid_col.str.lower().isin(["true", "1"]).sum()
            pct = int(100 * n_paid / len(df)) if len(df) > 0 else 0
            c3.metric("Paid", f"{pct}%")
        else:
            ft_col = _col(df, "employment_type")
            n_ft = ft_col.str.contains("full", case=False, regex=False).sum()
            pct = int(100 * n_ft / len(df)) if len(df) > 0 else 0
            c3.metric("Full-time", f"{pct}%")

        st.markdown("##### Full Database")

        if mode_key == "scholarship":
            col_config = {
                "apply_url":  st.column_config.LinkColumn("Apply", display_text="Apply ↗"),
                "source_url": st.column_config.LinkColumn("Source", display_text="Visit ↗"),
                "amount":     st.column_config.NumberColumn("Amount ($)", format="$%d"),
                "min_gpa":    st.column_config.NumberColumn("Min GPA", format="%.1f"),
            }
        else:
            col_config = {
                "apply_url":  st.column_config.LinkColumn("Apply", display_text="Apply ↗"),
                "source_url": st.column_config.LinkColumn("Source", display_text="Visit ↗"),
                "min_gpa":    st.column_config.NumberColumn("Min GPA", format="%.1f"),
            }

        st.dataframe(df_display, use_container_width=True, column_config=col_config, hide_index=True)

        # Per-session full database download
        db_buf = StringIO()
        df.to_csv(db_buf, index=False)
        db_filename = f"{mode_key}_database_session.csv"
        st.download_button(
            "⬇️  Download Full Database CSV",
            data=db_buf.getvalue(),
            file_name=db_filename,
            mime="text/csv",
            key="btn_dl_full_db",
        )

        if st.session_state.scout_done:
            n_m = len(st.session_state.matches_df) if st.session_state.matches_df is not None else 0
            st.success(
                f"**Step 1 complete!** {len(df)} {mode_noun} found — {n_m} match your profile. "
                "👉 Click the **Matches** tab to review them."
            )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — MATCHES
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="tag">Step 02</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-title">Your Matches</div>', unsafe_allow_html=True)

    if mode_key == "scholarship":
        st.markdown(
            '<div class="step-sub">Ranked matches for your GPA, major, state, and demographics. '
            "Unknown eligibility ranks below explicit fits.</div>",
            unsafe_allow_html=True,
        )
    elif mode_key == "internship":
        st.markdown(
            '<div class="step-sub">Ranked by role fit, major, location, and class year. '
            "Best matches first.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="step-sub">Ranked by role fit, major, location, and experience level. '
            "Best matches first.</div>",
            unsafe_allow_html=True,
        )

    if st.session_state.matches_df is None:
        st.info("Run a Scout first, or fill in your profile to filter the existing database.")
    elif st.session_state.matches_df.empty:
        st.warning(
            f"No matches for your current profile. Try broadening your filters "
            "or clearing the database and running a fresh Scout."
        )
    else:
        matches = st.session_state.matches_df
        if mode_key == "internship" and "paid" in matches.columns:
            matches_display = matches.drop(columns=["paid"])
        else:
            matches_display = matches

        c1, c2, c3 = st.columns(3)
        c1.metric("Matches Found", len(matches))
        c2.metric("Your GPA",      f"{gpa:.1f}")
        c3.metric("Field",          major or "—")

        if mode_key == "scholarship":
            col_config = {
                "match_score": st.column_config.NumberColumn("Score", format="%.1f"),
                "apply_url":   st.column_config.LinkColumn("Apply", display_text="Apply ↗"),
                "source_url":  st.column_config.LinkColumn("Source", display_text="Visit ↗"),
                "amount":      st.column_config.NumberColumn("Amount ($)", format="$%d"),
                "min_gpa":     st.column_config.NumberColumn("Min GPA", format="%.1f"),
            }
        else:
            col_config = {
                "match_score": st.column_config.NumberColumn("Score", format="%.1f"),
                "apply_url":   st.column_config.LinkColumn("Apply", display_text="Apply ↗"),
                "source_url":  st.column_config.LinkColumn("Source", display_text="Visit ↗"),
                "min_gpa":     st.column_config.NumberColumn("Min GPA", format="%.1f"),
            }

        st.dataframe(matches_display, use_container_width=True, column_config=col_config, hide_index=True)

        buf = StringIO()
        matches.to_csv(buf, index=False)
        dl_name = f"my_{mode_key}_matches.csv"
        st.download_button(
            "⬇️  Download Matches CSV",
            data=buf.getvalue(),
            file_name=dl_name,
            mime="text/csv",
            key="btn_dl_matches",
        )

        st.success(
            f"**Step 2 complete!** {len(matches)} match(es), best fits first. "
            "👉 Open **Drafts**, add your resume, and generate letters for the top matches."
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — DRAFTS
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="tag">Step 03</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-title">Draft cover letters</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="step-sub">'
        "Uses the apply/detail page when available (not just the scout hit), plus extracted "
        "listing facts and your profile — including your name for the sign-off."
        "</div>",
        unsafe_allow_html=True,
    )

    no_matches = st.session_state.matches_df is None or st.session_state.matches_df.empty
    no_resume  = not resume_text.strip()

    if not api_key_set:
        st.warning(f"👈  Paste your {provider} API key in the sidebar to generate drafts.")
    elif no_matches:
        st.info("Complete Scout + Matches first so there is something to draft for.")
    elif no_resume:
        st.warning("👈  Upload a resume or paste your background in the sidebar before drafting.")
    else:
        matches = st.session_state.matches_df
        max_draftable = min(10, len(matches))
        default_n = min(3, max_draftable)
        draft_n = st.slider(
            "How many top matches to draft",
            min_value=1,
            max_value=max(1, max_draftable),
            value=default_n,
            help="Drafts the highest-ranked matches first to save time and API cost.",
        )
        if not name.strip():
            st.caption("Tip: add your Full Name in the sidebar for a correct sign-off.")

        st.write(
            f"Ready to draft **{draft_n}** letter(s) for **{name or 'you'}** "
            f"(from {len(matches)} ranked match(es))."
        )

        draft_col, _ = st.columns([2, 3])
        with draft_col:
            draft_clicked = st.button(
                f"✏️  Generate top {draft_n} draft{'s' if draft_n != 1 else ''}",
                type="primary",
                disabled=st.session_state.is_drafting,
                use_container_width=True,
                key="btn_draft",
            )

        if draft_clicked and not st.session_state.is_drafting:
            st.session_state.is_drafting = True
            st.session_state.drafts = {}
            bar = st.progress(0, text="Starting...")
            to_draft = matches.head(draft_n)

            for i, (idx, row) in enumerate(to_draft.iterrows()):
                display_name = row.get(_name_col, f"Item {i+1}")
                if _is_career_mode and row.get("role"):
                    display_name = f"{row['role']} @ {display_name}"
                display_name = f"{display_name} · #{i+1}"
                bar.progress(i / max(len(to_draft), 1), text=f"Writing: {display_name}...")
                apply_url = str(row.get("apply_url") or "").strip()
                source_url = str(row.get("source_url") or "").strip()
                link_url = apply_url if apply_url.startswith("http") else source_url
                try:
                    essay = draft_application(
                        row, resume_text, mode=mode_key,
                        provider=provider, api_key=active_api_key, ollama_host=ollama_host,
                        applicant_name=name.strip(), profile=draft_profile,
                    )
                    st.session_state.drafts[display_name] = (essay, link_url)
                except Exception as e:
                    st.session_state.drafts[display_name] = (f"Error generating draft: {e}", link_url)
            bar.progress(1.0, text="All drafts complete!")
            st.session_state.is_drafting = False

        if st.session_state.drafts:
            st.markdown("---")
            st.markdown(f"##### {len(st.session_state.drafts)} Draft(s) Ready")
            for item_name, (essay, url) in st.session_state.drafts.items():
                with st.expander(f"📄  {html.escape(str(item_name))}"):
                    if url:
                        domain = url.split("//")[-1].split("/")[0]
                        st.markdown(f"🔗 [Apply here]({url}) · `{domain}`")
                    st.markdown(
                        f'<div class="draft-body">{html.escape(essay)}</div>',
                        unsafe_allow_html=True,
                    )
                    safe = "".join(x for x in str(item_name) if x.isalnum())[:60]
                    st.download_button(
                        "⬇️  Download .txt",
                        data=essay,
                        file_name=f"Draft_{safe or 'letter'}.txt",
                        mime="text/plain",
                        key=f"btn_dl_{safe}",
                    )

            emoji = MODE_ICONS.get(mode, "✨")
            st.success(
                f"**Step 3 complete!** {len(st.session_state.drafts)} draft(s) ready. "
                f"Review and personalize each one before submitting. Good luck! {emoji}"
            )
