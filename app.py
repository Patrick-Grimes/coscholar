import streamlit as st
import pandas as pd
import os
import datetime
import html
import pdfplumber
import io
from io import StringIO

from agent import find_scholarship_urls, find_internship_urls
from pipeline import run_pipeline, DB_PATHS
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
            return aliases
    terms = [m]
    if " " in m:
        terms.append(m.split()[0])
    return terms


def _col(df, name):
    return df.get(name, pd.Series([""] * len(df))).astype(str).str.strip()

def _is_open(series):
    return series.isin(["", "[]", "nan", "None", "none"])


def filter_scholarship_matches(
    df: pd.DataFrame,
    major: str,
    gpa: float,
    states: list[str],
    ethnicity: str = "",
    first_gen: bool = False,
    income_based: bool = False,
) -> pd.DataFrame:

    gpa_mask = df["min_gpa"] <= gpa

    major_col = _col(df, "majors")
    terms = major_terms(major)
    major_match = pd.Series([False] * len(df))
    for term in terms:
        major_match = major_match | major_col.str.contains(term, case=False, regex=False)
    major_mask = _is_open(major_col) | major_match

    state_col = _col(df, "eligible_states")
    if states:
        state_mask = _is_open(state_col)
        for s in states:
            s_clean = (s or "").strip()
            if not s_clean:
                continue
            state_mask = state_mask | state_col.str.contains(s_clean, case=False, regex=False)
    else:
        state_mask = pd.Series([True] * len(df))

    eth_col = _col(df, "ethnicity")
    if ethnicity:
        eth_mask = _is_open(eth_col) | eth_col.str.contains(ethnicity, case=False, regex=False)
    else:
        eth_mask = pd.Series([True] * len(df))

    fg_col = _col(df, "first_gen")
    first_gen_mask = (
        ~fg_col.str.lower().isin(["true", "1"]) |
        pd.Series([first_gen] * len(df))
    )

    inc_col = _col(df, "income_based")
    income_mask = (
        ~inc_col.str.lower().isin(["true", "1"]) |
        pd.Series([income_based] * len(df))
    )

    mask = gpa_mask & major_mask & state_mask & eth_mask & first_gen_mask & income_mask
    return df[mask].copy()


def filter_internship_matches(
    df: pd.DataFrame,
    major: str,
    gpa: float,
    states: list[str],
    location_pref: str = "Any",
    class_year: str = "",
) -> pd.DataFrame:

    gpa_col = pd.to_numeric(df.get("min_gpa", pd.Series()), errors="coerce").fillna(0.0)
    gpa_mask = gpa_col <= gpa

    major_col = _col(df, "preferred_majors")
    terms = major_terms(major)
    major_match = pd.Series([False] * len(df))
    for term in terms:
        major_match = major_match | major_col.str.contains(term, case=False, regex=False)
    major_mask = _is_open(major_col) | major_match

    loc_col = _col(df, "location")
    if location_pref and location_pref != "Any":
        location_pref_mask = loc_col.str.contains(location_pref, case=False, regex=False)
    else:
        location_pref_mask = pd.Series([True] * len(df))

    if states:
        state_location_mask = pd.Series([False] * len(df))
        for s in states:
            s_clean = (s or "").strip()
            if not s_clean:
                continue
            state_location_mask = state_location_mask | loc_col.str.contains(s_clean, case=False, regex=False)
    else:
        state_location_mask = pd.Series([True] * len(df))

    if class_year:
        cy_col = _col(df, "class_year")
        class_year_mask = _is_open(cy_col) | cy_col.str.contains(class_year, case=False, regex=False)
    else:
        class_year_mask = pd.Series([True] * len(df))

    mask = gpa_mask & major_mask & location_pref_mask & state_location_mask & class_year_mask
    return df[mask].copy()


# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [
    ("mode",            "Scholarship"),
    ("listings_df",     None),
    ("matches_df",      None),
    ("drafts",          {}),
    ("is_scouting",     False),
    ("is_drafting",     False),
    ("scout_done",      False),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 CoScholar")
    st.caption("Autonomous Scholarship & Internship Agent")
    st.divider()

    # Mode toggle
    st.markdown('<div class="slabel">Mode</div>', unsafe_allow_html=True)
    new_mode = st.radio(
        "Mode",
        ["Scholarship", "Internship"],
        index=0 if st.session_state.mode == "Scholarship" else 1,
        horizontal=True,
        label_visibility="collapsed",
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
    mode_key = mode.lower()  # "scholarship" or "internship"
    DB_PATH = DB_PATHS[mode_key]
    mode_noun = "scholarships" if mode_key == "scholarship" else "internships"

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
    st.markdown('<div class="slabel">Profile</div>', unsafe_allow_html=True)
    name  = st.text_input("Full Name", placeholder="Jane Smith", max_chars=100)
    major = st.text_input("Major / Field", placeholder="Computer Science", max_chars=100)
    gpa   = st.number_input("GPA", min_value=0.0, max_value=4.0, step=0.1, format="%.1f")
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
    else:
        ethnicity = ""
        first_gen = False
        income_based = False
        desired_role  = st.text_input("Desired Role / Title", placeholder="Software Engineering Intern", max_chars=100)
        location_pref = st.selectbox("Location Preference", ["Any", "Remote", "Hybrid", "On-site"])
        class_year    = st.selectbox("Class Year", ["", "Freshman", "Sophomore", "Junior", "Senior"])

    # Load existing DB on first run
    if st.session_state.listings_df is None and os.path.exists(DB_PATH):
        st.session_state.listings_df = load_csv(DB_PATH)

    # Auto-filter on profile change
    if st.session_state.listings_df is not None and major.strip() and gpa > 0:
        if mode_key == "scholarship":
            st.session_state.matches_df = filter_scholarship_matches(
                st.session_state.listings_df, major, gpa, states,
                ethnicity, first_gen, income_based,
            )
        else:
            st.session_state.matches_df = filter_internship_matches(
                st.session_state.listings_df, major, gpa, states,
                location_pref, class_year,
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

    st.markdown('<div class="slabel">Search Settings</div>', unsafe_allow_html=True)
    max_results = st.slider("URLs to Scout", 1, 10, 3)

    # DB status
    st.divider()
    if os.path.exists(DB_PATH):
        ts    = datetime.datetime.fromtimestamp(os.path.getmtime(DB_PATH)).strftime("%b %d · %I:%M %p")
        count = len(st.session_state.listings_df) if st.session_state.listings_df is not None else len(pd.read_csv(DB_PATH))
        st.markdown(
            f'<div class="status-loaded">📂 Database loaded<br>'
            f'<span style="opacity:0.7">{count} {mode_noun} · {ts}</span></div>',
            unsafe_allow_html=True,
        )
        if st.button("🗑  Clear Database", use_container_width=True, key="btn_clear_db"):
            os.remove(DB_PATH)
            st.session_state.listings_df = None
            st.session_state.matches_df  = None
            st.session_state.drafts      = {}
            load_csv.clear()
            st.rerun()
    else:
        st.caption(f"No {mode_noun} database yet. Run a Scout to build one.")


# ── Header ────────────────────────────────────────────────────────────────────
_ICONS = {"Scholarship": "🎓", "Internship": "💼"}
_TITLES = {"Scholarship": "CoScholar **AI**", "Internship": "CoIntern **AI**"}
_SUBTITLES = {
    "Scholarship": "Discovery → Filtering → Drafting, end-to-end.",
    "Internship":  "Scout → Match → Draft, end-to-end.",
}

title_col, pill_col = st.columns([5, 1])
with title_col:
    st.markdown(f"# {_TITLES[mode]}")
    st.caption(_SUBTITLES[mode])
    st.caption("Switch between Scholarship and Internship mode in the sidebar.")
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
profile_ready  = all([name.strip(), major.strip(), gpa > 0])

# The name column used to identify a listing in drafts/display
_name_col = "name" if mode_key == "scholarship" else "company"

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
                f"📂 **{len(st.session_state.listings_df)} {mode_noun} already in your database.** "
                "👈 Fill in your Name, Major, and GPA in the sidebar — matches will populate instantly without re-scouting."
            )
        else:
            st.info("👈  Fill in your Name, Major, and GPA in the sidebar to get started.")
    else:
        if mode_key == "scholarship":
            angles_preview = sum([bool(major), bool(states), bool(ethnicity), first_gen, income_based, True])
        else:
            angles_preview = sum([bool(desired_role), bool(major), bool(states), location_pref == "Remote", bool(class_year), True])
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
                else:
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

                st.write(f"🔎 Running **{len(angles)} searches:** {', '.join(angles)}")

                urls = None
                try:
                    if mode_key == "scholarship":
                        urls = find_scholarship_urls(profile, max_results=max_results)
                    else:
                        urls = find_internship_urls(profile, max_results=max_results)
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

                run_pipeline(urls, mode=mode_key, progress_callback=ui_callback,
                             provider=provider, api_key=active_api_key, ollama_host=ollama_host)
                st.session_state.is_scouting = False

                if os.path.exists(DB_PATH):
                    load_csv.clear()
                    df = load_csv(DB_PATH)
                    if mode_key == "scholarship":
                        matches = filter_scholarship_matches(df, major, gpa, states, ethnicity, first_gen, income_based)
                    else:
                        matches = filter_internship_matches(df, major, gpa, states, location_pref, class_year)
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
                    status.update(label="⚠️ Pipeline ran but no data was saved.", state="error", expanded=False)

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
        else:
            paid_col = _col(df, "paid")
            n_paid = paid_col.str.lower().isin(["true", "1"]).sum()
            pct = int(100 * n_paid / len(df)) if len(df) > 0 else 0
            c3.metric("Paid", f"{pct}%")

        st.markdown("##### Full Database")

        if mode_key == "scholarship":
            col_config = {
                "source_url": st.column_config.LinkColumn("Source", display_text="Visit ↗"),
                "amount":     st.column_config.NumberColumn("Amount ($)", format="$%d"),
                "min_gpa":    st.column_config.NumberColumn("Min GPA", format="%.1f"),
            }
        else:
            col_config = {
                "source_url": st.column_config.LinkColumn("Source", display_text="Visit ↗"),
                "min_gpa":    st.column_config.NumberColumn("Min GPA", format="%.1f"),
            }

        st.dataframe(df_display, use_container_width=True, column_config=col_config, hide_index=True)

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
            '<div class="step-sub">Filtered to scholarships that fit your GPA, major, state, and demographics.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="step-sub">Filtered to internships that fit your major, GPA, location, and class year.</div>',
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
                "source_url": st.column_config.LinkColumn("Source", display_text="Visit ↗"),
                "amount":     st.column_config.NumberColumn("Amount ($)", format="$%d"),
                "min_gpa":    st.column_config.NumberColumn("Min GPA", format="%.1f"),
            }
        else:
            col_config = {
                "source_url": st.column_config.LinkColumn("Source", display_text="Visit ↗"),
                "min_gpa":    st.column_config.NumberColumn("Min GPA", format="%.1f"),
            }

        st.dataframe(matches_display, use_container_width=True, column_config=col_config, hide_index=True)

        buf = StringIO()
        matches.to_csv(buf, index=False)
        dl_name = "my_scholarship_matches.csv" if mode_key == "scholarship" else "my_internship_matches.csv"
        st.download_button(
            "⬇️  Download Matches CSV",
            data=buf.getvalue(),
            file_name=dl_name,
            mime="text/csv",
            key="btn_dl_matches",
        )

        st.success(
            f"**Step 2 complete!** {len(matches)} match(es) found. "
            "👉 Click the **Drafts** tab, paste your profile in the sidebar, "
            "and hit **Generate All Drafts**."
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — DRAFTS
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="tag">Step 03</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-title">Draft Applications</div>', unsafe_allow_html=True)

    if mode_key == "scholarship":
        st.markdown(
            '<div class="step-sub">'
            "Reads each scholarship's live page and writes a personalized cover letter "
            "grounded in your uploaded profile."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="step-sub">'
            "Reads each internship's live page and writes a personalized cover letter "
            "grounded in your uploaded profile."
            "</div>",
            unsafe_allow_html=True,
        )

    no_matches = st.session_state.matches_df is None or st.session_state.matches_df.empty
    no_resume  = not resume_text.strip()

    if not api_key_set:
        st.warning(f"👈  Paste your {provider} API key in the sidebar to generate drafts.")
    elif no_matches:
        st.info("Complete Step 1 first so there are matches to draft for.")
    elif no_resume:
        st.warning("👈  Paste your profile or upload a resume in the sidebar before drafting.")
    else:
        matches = st.session_state.matches_df
        st.write(f"Ready to draft **{len(matches)}** cover letter(s) for **{name or 'you'}**.")

        draft_col, _ = st.columns([2, 3])
        with draft_col:
            draft_clicked = st.button(
                "✏️  Generate All Drafts",
                type="primary",
                disabled=st.session_state.is_drafting,
                use_container_width=True,
                key="btn_draft",
            )

        if draft_clicked and not st.session_state.is_drafting:
            st.session_state.is_drafting = True
            st.session_state.drafts      = {}
            bar = st.progress(0, text="Starting...")

            for i, (_, row) in enumerate(matches.iterrows()):
                display_name = row.get(_name_col, f"Item {i+1}")
                if mode_key == "internship" and row.get("role"):
                    display_name = f"{row['role']} @ {display_name}"
                bar.progress(i / len(matches), text=f"Writing: {display_name}...")
                source_url = row.get("source_url", "")
                try:
                    essay = draft_application(
                        row, resume_text, mode=mode_key,
                        provider=provider, api_key=active_api_key, ollama_host=ollama_host,
                    )
                    st.session_state.drafts[display_name] = (essay, source_url)
                except Exception as e:
                    st.session_state.drafts[display_name] = (f"Error generating draft: {e}", source_url)
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
                    safe = "".join(x for x in str(item_name) if x.isalnum())
                    st.download_button(
                        "⬇️  Download .txt",
                        data=essay,
                        file_name=f"Draft_{safe}.txt",
                        mime="text/plain",
                        key=f"btn_dl_{safe}",
                    )

            emoji = "🎓" if mode_key == "scholarship" else "💼"
            st.success(
                f"**Step 3 complete!** {len(st.session_state.drafts)} draft(s) ready. "
                f"Review and personalize each one before submitting. Good luck! {emoji}"
            )
