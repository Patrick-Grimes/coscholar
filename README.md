# CoScholar — Autonomous Scholarship Agent

An end-to-end agentic pipeline that scouts the web for scholarships, filters them against a student profile, and drafts personalized cover letters using RAG (Retrieval-Augmented Generation).

Built with Python, Gemini API, Streamlit, and BeautifulSoup.

---

## How It Works

| Step | What Happens |
|---|---|
| **Scout** | Runs multiple targeted searches (by major, state, ethnicity, first-gen status, etc.) via DuckDuckGo and deduplicates URLs across all queries |
| **Extract** | Fetches each page, strips JS/CSS to reduce tokens, sends clean HTML to Gemini for structured JSON extraction - name, amount, GPA, deadline, and eligibility fields |
| **Filter** | Multi-axis Pandas filtering: GPA, major, state, ethnicity, first-gen, income-based. Expired scholarships are dropped automatically by deadline parsing |
| **Draft** | Re-fetches each matched scholarship's live page and uses RAG to write a personalized cover letter grounded in your uploaded resume |

---

## Stack

| Layer | Tools |
|---|---|
| UI / Orchestration | Streamlit |
| LLM | Gemini 2.0 Flash (`google-genai`) |
| Web Search | DuckDuckGo (`ddgs`) |
| Scraping | BeautifulSoup4, Requests |
| Data | Pandas - CSV flat-file persistence |
| Resume Parsing | pdfplumber |
| Env Management | python-dotenv |

---

## Setup

```bash
# 1. Clone and create a virtual environment
git clone https://github.com/YOUR_USERNAME/coscholar.git
cd coscholar
python -m venv .venv
source .venv/bin/activate        # Mac/Linux
.venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your Gemini API key
echo "API_KEY=your_key_here" > .env

# 4. Run
streamlit run app.py
```

Get a free Gemini API key at [aistudio.google.com](https://aistudio.google.com).

---

## Project Structure

```
coscholar/
├── .streamlit/
│   └── config.toml      # Dark theme config
├── app.py               # Streamlit UI + filtering logic
├── agent.py             # Multi-query web discovery
├── pipeline.py          # HTML fetch, clean, and orchestration
├── ai_scraper.py        # Gemini-based structured extraction
├── drafter.py           # RAG cover letter generation
└── requirements.txt
```

---

## Notes

- Works best on static HTML pages. JavaScript-heavy SPAs (React/Angular) return limited results.
- The scholarship database persists as `scholarship_database.csv` locally and auto-loads on browser refresh.
- `time.sleep(3)` between Gemini calls is intentional — rate limit buffer for the free tier.
- Filtering logic lives in `app.py` (`filter_matches()`). It runs on scout completion and reactively as you update your profile in the sidebar.