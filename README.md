# CoScholar — Autonomous Scholarship Agent

🔗 [Live Demo](https://coscholar-ai.streamlit.app)

An end-to-end agentic pipeline that scouts the web for scholarships, filters them against a student profile, and drafts personalized cover letters using RAG (Retrieval-Augmented Generation).

Built with Python, Gemini API, Streamlit, and BeautifulSoup.

---

## How It Works

| Step | What Happens |
|---|---|
| **Scout** | Runs multiple targeted searches (by major, state, ethnicity, first-gen status, etc.) via DuckDuckGo and deduplicates URLs across all queries |
| **Extract** | Fetches each page, strips JS/CSS to reduce tokens, sends clean HTML to Gemini for structured JSON extraction — name, amount, GPA, deadline, and eligibility fields |
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
| Data | Pandas — CSV flat-file persistence |
| Resume Parsing | pdfplumber |
| Env Management | python-dotenv |

---

## API Key

You need a free **Gemini API key** to use CoScholar. Get one at [aistudio.google.com](https://aistudio.google.com).

You can provide your key in one of two ways:

**Option A — Sidebar (recommended for the live app)**
Paste your key directly into the Gemini API Key field in the sidebar. It is used only for that session and is never stored or written to disk.

**Option B — .env file (for local development)**
Create a `.env` file in the project root:
```
API_KEY=your_key_here
```
The app will automatically load it on startup. The sidebar key always takes priority over the `.env` key if both are present.

> **Note:** Never commit your `.env` file. It is gitignored by default.

---

## Setup

```bash
# 1. Clone and create a virtual environment
git clone https://github.com/Patrick-Grimes/coscholar.git
cd coscholar
python -m venv .venv
source .venv/bin/activate        # Mac/Linux
.venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Add your Gemini API key to .env
echo "API_KEY=your_key_here" > .env

# 4. Run
streamlit run app.py
```

---

## Project Structure

```
coscholar/
├── .streamlit/
│   └── config.toml      # Dark theme config
├── app.py               # Streamlit UI + filtering logic
├── agent.py             # Multi-query web discovery (DuckDuckGo)
├── pipeline.py          # HTML fetch, clean, SSRF validation, orchestration
├── ai_scraper.py        # Gemini-based structured extraction + schema validation
├── drafter.py           # RAG cover letter generation
└── requirements.txt
```

---

## Security

- API keys are passed directly to the Gemini client — they are never written to environment variables or logged
- Scraped HTML and LLM output is sanitized before rendering to prevent XSS
- Prompts use XML delimiters around untrusted content to mitigate prompt injection
- URLs are validated against private/loopback IP ranges before fetching (SSRF protection)
- Uploaded files are validated by size (5 MB limit) and magic bytes before parsing

---

## Notes

- Works best on static HTML pages. JavaScript-heavy SPAs (React/Angular) return limited results.
- The scholarship database persists as `scholarship_database.csv` locally and auto-loads on browser refresh.
- Gemini calls use exponential backoff retry (via `tenacity`) to handle rate limits gracefully.
- Filtering logic lives in `app.py` (`filter_matches()`). It runs on scout completion and reactively as you update your profile in the sidebar.