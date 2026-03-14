import streamlit as st
import pandas as pd
import os
import datetime
import html
import pdfplumber
import io
from io import StringIO

from agent import find_scholarship_urls
from pipeline import run_scholarship_pipeline
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
    df["min_gpa"] = pd.to_numeric(df["min_gpa"], errors="coerce").fillna(0.0)
    return df


# Key terms for fuzzy major matching — maps common shorthand to full names
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

def major_terms(major: str) -> list[str]:
    """Returns a list of terms to match against for a given major input."""
    m = major.lower().strip()
    for key, aliases in MAJOR_ALIASES.items():
        if m in aliases or m == key:
            return aliases
    # Fallback: just use the input itself + first word
    terms = [m]
    if " " in m:
        terms.append(m.split()[0])
    return terms


def filter_matches(
    df: pd.DataFrame,
    major: str,
    gpa: float,
    state: str,
    ethnicity: str = "",
    first_gen: bool = False,
    income_based: bool = False,
) -> pd.DataFrame:

    def col(name):
        return df.get(name, pd.Series([""] * len(df))).astype(str).str.strip()

    def is_open(series):
        """True if the field is blank/empty/null — meaning open to all."""
        return series.isin(["", "[]", "nan", "None", "none"])

    # GPA — only filter if scholarship has a real GPA requirement
    gpa_mask = df["min_gpa"] <= gpa

    # Major — open to all OR contains any alias for the entered major
    major_col = col("majors")
    terms = major_terms(major)
    major_match = pd.Series([False] * len(df))
    for term in terms:
        major_match = major_match | major_col.str.contains(term, case=False, regex=False)
    major_mask = is_open(major_col) | major_match

    # State — open to all OR contains user's state
    state_col = col("eligible_states")
    state_mask = (
        is_open(state_col) |
        state_col.str.contains(state, case=False, regex=False)
    ) if state else pd.Series([True] * len(df))

    # Ethnicity — open to all OR matches user's ethnicity
    # Key fix: if user entered ethnicity, we keep scholarships open to all
    # AND scholarships explicitly for their ethnicity
    eth_col = col("ethnicity")
    if ethnicity:
        eth_mask = is_open(eth_col) | eth_col.str.contains(ethnicity, case=False, regex=False)
    else:
        eth_mask = pd.Series([True] * len(df))

    # First-gen — only filter OUT scholarships that require first-gen if user isn't
    fg_col = col("first_gen")
    first_gen_mask = (
        ~fg_col.str.lower().isin(["true", "1"]) |
        pd.Series([first_gen] * len(df))
    )

    # Income — only filter OUT need-based if user didn't check it
    inc_col = col("income_based")
    income_mask = (
        ~inc_col.str.lower().isin(["true", "1"]) |
        pd.Series([income_based] * len(df))
    )

    mask = gpa_mask & major_mask & state_mask & eth_mask & first_gen_mask & income_mask
    return df[mask].copy()


# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [
    ("scholarships_df", None),
    ("matches_df",      None),
    ("drafts",          {}),
    ("is_scouting",     False),
    ("is_drafting",     False),
    ("scout_done",      False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

DB_PATH = "scholarship_database.csv"
if st.session_state.scholarships_df is None and os.path.exists(DB_PATH):
    st.session_state.scholarships_df = load_csv(DB_PATH)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 CoScholar")
    st.caption("Autonomous Scholarship Agent · Powered by Gemini")
    st.divider()

    # API Key
    st.markdown('<div class="slabel">Gemini API Key</div>', unsafe_allow_html=True)
    user_api_key = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="Paste your key from aistudio.google.com",
        label_visibility="collapsed",
    )
    if user_api_key:
        st.success("✅ Key set for this session")
    else:
        st.caption("🔑 Get a free key at [aistudio.google.com](https://aistudio.google.com)")

    st.divider()

    # Profile
    st.markdown('<div class="slabel">Profile</div>', unsafe_allow_html=True)
    name         = st.text_input("Full Name",         placeholder="Jane Smith", max_chars=100)
    major        = st.text_input("Major / Field",     placeholder="Computer Science", max_chars=100)
    gpa          = st.number_input("GPA", min_value=0.0, max_value=4.0, step=0.1, format="%.1f")
    state        = st.text_input("State (2-letter)",  placeholder="NC", max_chars=2).upper()
    ethnicity    = st.text_input("Ethnicity (optional)", placeholder="e.g. Hispanic, Black, Asian, white", max_chars=50)
    first_gen    = st.checkbox("First-generation college student")
    income_based = st.checkbox("Financial need / income-based")

    # Auto-filter on profile change
    if st.session_state.scholarships_df is not None and major.strip() and gpa > 0:
        st.session_state.matches_df = filter_matches(
            st.session_state.scholarships_df, major, gpa, state,
            ethnicity, first_gen, income_based,
        )

    st.divider()

    # Applicant profile / resume
    st.markdown('<div class="slabel">Supplemental Information</div>', unsafe_allow_html=True)
    st.caption(
        "Upload files such as resumes, cover letters, or previous applications,"
        "and paste anything else in the text box - The more detail, the better the filtering and cover letters!"
    )

    # Multiple file upload
    uploaded_files = st.file_uploader(
        "Upload resume(s) (.txt or .pdf)",
        type=["txt", "pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
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
        count = len(st.session_state.scholarships_df) if st.session_state.scholarships_df is not None else len(pd.read_csv(DB_PATH))
        st.markdown(
            f'<div class="status-loaded">📂 Database loaded<br>'
            f'<span style="opacity:0.7">{count} scholarships · {ts}</span></div>',
            unsafe_allow_html=True,
        )
        if st.button("🗑  Clear Database", use_container_width=True, key="btn_clear_db"):
            os.remove(DB_PATH)
            st.session_state.scholarships_df = None
            st.session_state.matches_df      = None
            st.session_state.drafts          = {}
            load_csv.clear()
            st.rerun()
    else:
        st.caption("No database yet. Run a Scout to build one.")


# ── Header ────────────────────────────────────────────────────────────────────
title_col, pill_col = st.columns([5, 1])
with title_col:
    st.markdown("# CoScholar **AI**")
    st.caption("Discovery → Filtering → Drafting, end-to-end.")
with pill_col:
    if st.session_state.scholarships_df is not None:
        n = len(st.session_state.scholarships_df)
        st.markdown(
            f'<div style="text-align:right;padding-top:1.25rem">'
            f'<span class="tag">🗄 {n} in DB</span></div>',
            unsafe_allow_html=True,
        )

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
active_api_key = user_api_key or os.getenv("API_KEY") or ""
api_key_set    = bool(active_api_key)
profile_ready  = all([name.strip(), major.strip(), gpa > 0])

tab1, tab2, tab3 = st.tabs(["  🔍  Scout  ", "  📋  Matches  ", "  ✏️  Drafts  "])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — SCOUT
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="tag">Step 01</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-title">Scout Scholarships</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="step-sub">Searches the web across multiple eligibility dimensions, '
        'scrapes each page with Gemini, and saves results to your local database.</div>',
        unsafe_allow_html=True,
    )

    if not api_key_set:
        st.warning("👈  Paste your Gemini API key in the sidebar to get started.")
    elif not profile_ready:
        if st.session_state.scholarships_df is not None:
            st.info(
                f"📂 **{len(st.session_state.scholarships_df)} scholarships already in your database.** "
                "👈 Fill in your Name, Major, and GPA in the sidebar — matches will populate instantly without re-scouting."
            )
        else:
            st.info("👈  Fill in your Name, Major, and GPA in the sidebar to get started.")
    else:
        angles_preview = sum([bool(major), bool(state), bool(ethnicity), first_gen, income_based, True])
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
                angles = []
                if major:        angles.append(f"major ({major})")
                if state:        angles.append(f"state ({state})")
                if ethnicity:    angles.append(f"ethnicity ({ethnicity})")
                if first_gen:    angles.append("first-generation")
                if income_based: angles.append("need-based")
                angles.append("general")

                st.write(f"🔎 Running **{len(angles)} searches:** {', '.join(angles)}")

                profile = {
                    "major":        major,
                    "state":        state,
                    "ethnicity":    ethnicity,
                    "first_gen":    first_gen,
                    "income_based": income_based,
                }

                urls = None
                try:
                    urls = find_scholarship_urls(profile, max_results=max_results)
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

                st.write(f"📡 Found **{len(urls)} unique URLs**. Scraping with Gemini...")

                counter_box  = st.empty()
                progress_bar = st.progress(0)

                def ui_callback(url, site_num, total_sites, found_so_far, done=False):
                    progress_bar.progress(site_num / total_sites)
                    if done:
                        counter_box.markdown(
                            f"✅ **All {total_sites} sites processed** · "
                            f"**{found_so_far} active scholarships** found"
                        )
                    else:
                        counter_box.markdown(
                            f"🌐 **Site {site_num} of {total_sites}** · "
                            f"**{found_so_far} scholarships** found so far"
                        )

                run_scholarship_pipeline(urls, progress_callback=ui_callback, api_key=active_api_key)
                st.session_state.is_scouting = False

                if os.path.exists(DB_PATH):
                    load_csv.clear()
                    df      = load_csv(DB_PATH)
                    matches = filter_matches(df, major, gpa, state, ethnicity, first_gen, income_based)
                    st.session_state.scholarships_df = df
                    st.session_state.matches_df      = matches
                    st.session_state.scout_done      = True

                    if len(df) == 0:
                        status.update(
                            label="⚠️ Sites scraped but 0 active scholarships found. Try different search terms.",
                            state="error", expanded=False,
                        )
                    else:
                        status.update(
                            label=f"✅ Scout complete — {len(df)} scholarships found · {len(matches)} match your profile.",
                            state="complete", expanded=False,
                        )
                else:
                    status.update(label="⚠️ Pipeline ran but no data was saved.", state="error", expanded=False)

    # Database display
    if st.session_state.scholarships_df is not None:
        df = st.session_state.scholarships_df
        st.markdown("---")

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Scraped", len(df))
        c2.metric(
            "Matching Profile",
            len(st.session_state.matches_df) if st.session_state.matches_df is not None else "—",
        )
        avg_amt = pd.to_numeric(df.get("amount", pd.Series()), errors="coerce").mean()
        c3.metric("Avg Award", f"${int(avg_amt):,}" if not pd.isna(avg_amt) else "—")

        st.markdown("##### Full Database")
        st.dataframe(
            df,
            use_container_width=True,
            column_config={
                "source_url": st.column_config.LinkColumn("Source"),
                "amount":     st.column_config.NumberColumn("Amount ($)", format="$%d"),
                "min_gpa":    st.column_config.NumberColumn("Min GPA", format="%.1f"),
            },
            hide_index=True,
        )

        if st.session_state.scout_done:
            n_m = len(st.session_state.matches_df) if st.session_state.matches_df is not None else 0
            st.success(
                f"**Step 1 complete!** {len(df)} scholarships found — {n_m} match your profile. "
                "👉 Click the **Matches** tab to review them."
            )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — MATCHES
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="tag">Step 02</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-title">Your Matches</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="step-sub">Filtered to scholarships that fit your GPA, major, state, and demographics.</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.matches_df is None:
        st.info("Run a Scout first, or fill in your profile to filter the existing database.")
    elif st.session_state.matches_df.empty:
        st.warning(
            "No matches for your current profile. Try: broadening your major, "
            "lowering GPA threshold, or clearing the database and running a fresh Scout."
        )
    else:
        matches = st.session_state.matches_df

        c1, c2, c3 = st.columns(3)
        c1.metric("Matches Found", len(matches))
        c2.metric("Your GPA",      f"{gpa:.1f}")
        c3.metric("Field",          major or "—")

        st.dataframe(
            matches,
            use_container_width=True,
            column_config={
                "source_url": st.column_config.LinkColumn("Source"),
                "amount":     st.column_config.NumberColumn("Amount ($)", format="$%d"),
                "min_gpa":    st.column_config.NumberColumn("Min GPA", format="%.1f"),
            },
            hide_index=True,
        )

        buf = StringIO()
        matches.to_csv(buf, index=False)
        st.download_button(
            "⬇️  Download Matches CSV",
            data=buf.getvalue(),
            file_name="my_scholarship_matches.csv",
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
    st.markdown(
        '<div class="step-sub">'
        "Reads each scholarship's live page and writes a personalized cover letter "
        "grounded in your uploaded profile."
        "</div>",
        unsafe_allow_html=True,
    )

    no_matches = st.session_state.matches_df is None or st.session_state.matches_df.empty
    no_resume  = not resume_text.strip()

    if not api_key_set:
        st.warning("👈  Paste your Gemini API key in the sidebar to generate drafts.")
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
                bar.progress(i / len(matches), text=f"Writing: {row['name']}...")
                try:
                    st.session_state.drafts[row["name"]] = draft_application(row, resume_text, api_key=active_api_key)
                except Exception as e:
                    st.session_state.drafts[row["name"]] = f"Error generating draft: {e}"
            bar.progress(1.0, text="All drafts complete!")
            st.session_state.is_drafting = False

        if st.session_state.drafts:
            st.markdown("---")
            st.markdown(f"##### {len(st.session_state.drafts)} Draft(s) Ready")
            for scholarship_name, essay in st.session_state.drafts.items():
                with st.expander(f"📄  {html.escape(scholarship_name)}"):
                    st.markdown(
                        f'<div class="draft-body">{html.escape(essay)}</div>',
                        unsafe_allow_html=True,
                    )
                    safe = "".join(x for x in scholarship_name if x.isalnum())
                    st.download_button(
                        "⬇️  Download .txt",
                        data=essay,
                        file_name=f"Draft_{safe}.txt",
                        mime="text/plain",
                        key=f"btn_dl_{safe}",
                    )

            st.success(
                f"**Step 3 complete!** {len(st.session_state.drafts)} draft(s) ready. "
                "Review and personalize each one before submitting. Good luck! 🎓"
            )