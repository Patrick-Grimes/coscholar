import streamlit as st
import pandas as pd
import os
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

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Hide Streamlit chrome */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
.stDeployButton { display: none; }
/* Hide "Made with Streamlit" but keep the header so sidebar toggle works */
[data-testid="stHeader"] { background: transparent; }

/* Card component */
.card {
    background: #1a1d27;
    border: 1px solid #2d3148;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* Metric override */
[data-testid="metric-container"] {
    background: #1a1d27;
    border: 1px solid #2d3148;
    border-radius: 10px;
    padding: 1rem 1.25rem;
}

/* Mono tag */
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

/* Step header */
.step-title {
    font-size: 1.3rem;
    font-weight: 700;
    margin: 0.25rem 0 0.2rem 0;
}
.step-sub {
    color: #64748b;
    font-size: 0.875rem;
    margin-bottom: 1.25rem;
}

/* Draft body text */
.draft-body {
    font-size: 0.9rem;
    line-height: 1.7;
    color: #cbd5e1;
    white-space: pre-wrap;
}

/* Sidebar section label */
.slabel {
    font-size: 0.65rem;
    font-family: monospace;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #475569;
    margin: 1.1rem 0 0.3rem 0;
}

/* Database status pill */
.status-loaded {
    background: #4ade8015;
    border: 1px solid #4ade8030;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    font-size: 0.85rem;
    color: #4ade80;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    """Cached CSV loader — fast on reruns, auto-busted when we call load_csv.clear()."""
    df = pd.read_csv(path)
    df["min_gpa"] = pd.to_numeric(df["min_gpa"], errors="coerce").fillna(0.0)
    return df


def filter_matches(
    df: pd.DataFrame,
    major: str,
    gpa: float,
    state: str,
    ethnicity: str = "",
    first_gen: bool = False,
    income_based: bool = False,
) -> pd.DataFrame:
    def open_or_match(col, value):
        """True if the column is blank/empty (open to all) OR contains the value."""
        s = df.get(col, pd.Series([""] * len(df))).astype(str).str.strip()
        return s.str.contains(value, case=False, regex=False) | (s == "") | (s == "[]") | (s.str.lower() == "nan")

    gpa_mask   = df["min_gpa"] <= gpa
    major_mask = open_or_match("majors", major)
    state_mask = open_or_match("eligible_states", state) if state else pd.Series([True] * len(df))
    eth_mask   = open_or_match("ethnicity", ethnicity) if ethnicity else pd.Series([True] * len(df))

    # For boolean flags: only filter OUT scholarships that explicitly require
    # a trait the student doesn't have (not the other way around)
    first_gen_col = df.get("first_gen", pd.Series([False] * len(df)))
    first_gen_mask = (~first_gen_col.astype(str).str.lower().isin(["true", "1"])) | pd.Series([first_gen] * len(df))

    income_col = df.get("income_based", pd.Series([False] * len(df)))
    income_mask = (~income_col.astype(str).str.lower().isin(["true", "1"])) | pd.Series([income_based] * len(df))

    return df[gpa_mask & major_mask & state_mask & eth_mask & first_gen_mask & income_mask].copy()


# ── Session state init ────────────────────────────────────────────────────────
for key, default in [("scholarships_df", None), ("matches_df", None), ("drafts", {}), ("is_scouting", False), ("is_drafting", False), ("scout_done", False)]:
    if key not in st.session_state:
        st.session_state[key] = default

# AUTO-LOAD: if database exists on disk and session is fresh, load it in
# This is how we survive browser refreshes — the CSV is the persistence layer
DB_PATH = "scholarship_database.csv"
if st.session_state.scholarships_df is None and os.path.exists(DB_PATH):
    st.session_state.scholarships_df = load_csv(DB_PATH)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 CoScholar")
    st.caption("Autonomous Scholarship Agent · Powered by Gemini")
    st.divider()

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

    st.markdown('<div class="slabel">Profile</div>', unsafe_allow_html=True)
    name  = st.text_input("Full Name", placeholder="Jane Smith")
    major = st.text_input("Major / Field", placeholder="Computer Science")
    gpa   = st.number_input("GPA", min_value=0.0, max_value=4.0, step=0.1, format="%.1f")
    state     = st.text_input("State (2-letter)", placeholder="NC", max_chars=2).upper()
    ethnicity = st.text_input("Ethnicity (optional)", placeholder="e.g. Hispanic, Black, Asian, white")
    first_gen = st.checkbox("First-generation college student")
    income_based = st.checkbox("Financial need / income-based")

    # Auto-filter whenever sidebar profile changes
    if st.session_state.scholarships_df is not None and major.strip() and gpa > 0:
        st.session_state.matches_df = filter_matches(
            st.session_state.scholarships_df, major, gpa, state,
            ethnicity, first_gen, income_based,
        )

    st.markdown('<div class="slabel">Applicant Profile</div>', unsafe_allow_html=True)
    st.caption(
        "Include anything relevant: demographics, ethnicity, income level, "
        "first-gen status, GPA, state, projects, skills, and goals. "
        "The more detail, the better the filtering and cover letters."
    )

    uploaded_file = st.file_uploader(
        "Upload resume (.txt or .pdf)",
        type=["txt", "pdf"],
        label_visibility="collapsed",
    )

    uploaded_text = ""
    if uploaded_file:
        if uploaded_file.type == "text/plain":
            uploaded_text = uploaded_file.read().decode("utf-8", errors="ignore")
            st.success(f"✅ {uploaded_file.name}")
        elif uploaded_file.type == "application/pdf":
            try:
                import pdfplumber, io
                with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
                    uploaded_text = "\n".join(p.extract_text() or "" for p in pdf.pages)
                st.success(f"✅ {uploaded_file.name}")
            except Exception as e:
                st.warning(f"PDF parse error: {e}")

    pasted_text = st.text_area(
        "Or paste your profile here",
        placeholder=(
            "• Name, school, major, grad year\n"
            "• GPA, state, ethnicity, income background\n"
            "• First-generation student? (yes/no)\n"
            "• Key projects & technical skills\n"
            "• Extracurriculars, interests, goals"
        ),
        height=180,
        label_visibility="collapsed",
    )

    resume_text = "\n\n".join(filter(None, [uploaded_text.strip(), pasted_text.strip()]))

    st.markdown('<div class="slabel">Search Settings</div>', unsafe_allow_html=True)
    max_results = st.slider("URLs to Scout", 1, 10, 3)

    # Database status footer
    st.divider()
    if os.path.exists(DB_PATH):
        import datetime
        ts    = datetime.datetime.fromtimestamp(os.path.getmtime(DB_PATH)).strftime("%b %d · %I:%M %p")
        count = len(pd.read_csv(DB_PATH))
        st.markdown(
            f'<div class="status-loaded">📂 Database loaded<br>'
            f'<span style="opacity:0.7">{count} scholarships · updated {ts}</span></div>',
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


# ── Main area ─────────────────────────────────────────────────────────────────
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

tab1, tab2, tab3 = st.tabs(["  🔍  Scout  ", "  📋  Matches  ", "  ✏️  Drafts  "])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — SCOUT
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="tag">Step 01</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-title">Scout Scholarships</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="step-sub">Searches the web for scholarship listing pages, '
        'scrapes them with Gemini, and saves results to your local database.</div>',
        unsafe_allow_html=True,
    )

    api_key_set   = bool(user_api_key or os.getenv("API_KEY"))
    profile_ready = all([name.strip(), major.strip(), gpa > 0])

    if not api_key_set:
        st.warning("👈  Paste your Gemini API key in the sidebar to get started.")
    elif not profile_ready:
        st.info("👈  Fill in your Name, Major, and GPA in the sidebar to get started.")
    else:
        # Estimate runtime so user isn't surprised
        angles_preview = sum([
            bool(major), bool(state), bool(ethnicity),
            first_gen, income_based, True  # always includes catch-all
        ])
        est_minutes = max(1, round((angles_preview * max_results * 4) / 60))
        st.caption(
            f"⏱ Estimated time: **{est_minutes}–{est_minutes + 1} min** "
            f"({angles_preview} search angles × {max_results} URLs each, ~4s per site). "
            "Feel free to grab a coffee — this runs in the background."
        )

        btn_col, _ = st.columns([2, 3])
        with btn_col:
            scout_clicked = st.button(
                "🚀  Start Scout",
                key="scout_btn",
                type="primary",
                disabled=st.session_state.is_scouting,
                use_container_width=True,
            )

        if scout_clicked:
            st.session_state.is_scouting = True
            st.rerun()

        if st.session_state.is_scouting:
            with st.status("Scouting the web — grab a coffee, this takes a few minutes ☕", expanded=True) as status:
                angles = []
                if major:        angles.append(f"major ({major})")
                if state:        angles.append(f"state ({state})")
                if ethnicity:    angles.append(f"ethnicity ({ethnicity})")
                if first_gen:    angles.append("first-generation")
                if income_based: angles.append("need-based")
                angles.append("general")

                st.write(f"🔎 Running **{len(angles)} searches:** {', '.join(angles)}")

                if user_api_key:
                    os.environ["API_KEY"] = user_api_key

                profile = {
                    "major":        major,
                    "state":        state,
                    "ethnicity":    ethnicity,
                    "first_gen":    first_gen,
                    "income_based": income_based,
                }

                try:
                    urls = find_scholarship_urls(profile, max_results=max_results)
                except Exception as e:
                    st.session_state.is_scouting = False
                    status.update(label=f"❌ Search failed: {e}", state="error", expanded=False)
                    st.stop()

                if not urls:
                    st.session_state.is_scouting = False
                    status.update(
                        label="⚠️ No URLs found. Try broadening your profile or increasing URLs to Scout.",
                        state="error", expanded=False,
                    )
                    st.stop()

                st.write(f"📡 Found **{len(urls)} unique URLs**. Scraping each site with Gemini...")

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

                run_scholarship_pipeline(urls, progress_callback=ui_callback)
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
                            label=f"✅ Scout complete — {len(df)} active scholarships · {len(matches)} match your profile.",
                            state="complete", expanded=False,
                        )
                else:
                    status.update(label="⚠️ Pipeline ran but no data was saved.", state="error", expanded=False)

    if st.session_state.scholarships_df is not None:
        n = len(st.session_state.scholarships_df)
        st.markdown(
            f'<div style="text-align:right;padding-top:1.25rem">'
            f'<span class="tag">🗄 {n} in DB</span></div>',
            unsafe_allow_html=True,
        )

st.divider()



# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — SCOUT
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="tag">Step 02</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-title">Your Matches</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="step-sub">Scholarships filtered to your GPA, major, and state.</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.matches_df is None:
        st.info("Run a Scout first, or fill in your profile to filter the existing database.")
    elif st.session_state.matches_df.empty:
        st.warning("No matches for your current profile. Try broadening your major or adjusting GPA.")
    else:
        matches = st.session_state.matches_df

        c1, c2, c3 = st.columns(3)
        c1.metric("Matches Found", len(matches))
        c2.metric("Your GPA",     f"{gpa:.1f}")
        c3.metric("Field",         major or "—")

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
        )

        st.success(
            f"**Step 2 complete!** You have {len(matches)} scholarship match(es). "
            "👉 Click the **Drafts** tab above, then make sure your profile is pasted "
            "in the sidebar, and hit **Generate All Drafts**."
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
        "grounded in your profile."
        "</div>",
        unsafe_allow_html=True,
    )

    no_matches = st.session_state.matches_df is None or st.session_state.matches_df.empty
    no_resume  = not resume_text.strip()

    if no_matches:
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
            if user_api_key:
                os.environ["API_KEY"] = user_api_key
            st.session_state.is_drafting = True
            st.session_state.drafts = {}
            bar = st.progress(0, text="Starting...")
            for i, (_, row) in enumerate(matches.iterrows()):
                bar.progress(i / len(matches), text=f"Writing: {row['name']}...")
                try:
                    st.session_state.drafts[row["name"]] = draft_application(row, resume_text)
                except Exception as e:
                    st.session_state.drafts[row["name"]] = f"Error generating draft: {e}"
            bar.progress(1.0, text="All drafts complete!")
            st.session_state.is_drafting = False

        if st.session_state.drafts:
            st.markdown("---")
            st.markdown(f"##### {len(st.session_state.drafts)} Draft(s) Ready")
            for scholarship_name, essay in st.session_state.drafts.items():
                with st.expander(f"📄  {scholarship_name}"):
                    st.markdown(
                        f'<div class="draft-body">{essay}</div>',
                        unsafe_allow_html=True,
                    )
                    safe = "".join(x for x in scholarship_name if x.isalnum())
                    st.download_button(
                        "⬇️  Download .txt",
                        data=essay,
                        file_name=f"Draft_{safe}.txt",
                        mime="text/plain",
                        key=f"dl_{safe}",
                    )

            st.success(
                f"**Step 3 complete!** {len(st.session_state.drafts)} draft(s) ready. "
                "Download each one above, review and personalize before submitting. "
                "Good luck! 🎓"
            )