# CoScholar - Autonomous Scholarship, Internship & Job Agent

🔗 [Live Demo](https://coscholar-ai.streamlit.app)

An end-to-end agentic pipeline that scouts the web for **scholarships, internships, or jobs**, filters them against a student/candidate profile, and drafts personalized cover letters using RAG (Retrieval-Augmented Generation).

Built with Python, Streamlit, BeautifulSoup, and your choice of AI provider.

---

## How It Works

| Step | What Happens |
|---|---|
| **Scout** | Runs multiple targeted searches via DuckDuckGo based on your profile and mode, deduplicates URLs across all queries |
| **Extract** | Fetches each page, strips JS/CSS to reduce tokens, sends clean HTML to your chosen AI for structured JSON extraction |
| **Filter** | Multi-axis Pandas filtering against your profile. Expired listings are dropped automatically by deadline parsing |
| **Draft** | Re-fetches each match's live page and uses RAG to write a personalized cover letter grounded in your uploaded resume |

---

## Modes

An **I want** control at the top of the sidebar switches the entire app between **Job**, **Internship**, and **Scholarship**. Career modes come first; Scholarship remains available. Each mode adapts the full pipeline:

| | Scholarship | Internship | Job |
|---|---|---|---|
| **Branding** | CoScholar AI | CoIntern AI | CoJobs AI |
| **Profile fields** | Major, GPA, state, ethnicity, first-gen, income-based | Major, GPA, state, desired role, location preference, class year | Major, GPA, state, desired role, location preference, experience level |
| **Search queries** | Major-specific, state-based, demographic, need-based | Role-specific, major-based, location-based, class-year | Role-specific, field-based, location-based, experience-level |
| **Extraction schema** | Name, amount, deadline, GPA, majors, states, ethnicity, first-gen, income-based | Company, role, location, skills, majors, GPA, class year, paid, compensation, deadline | Company, role, location, skills, majors, GPA, experience level, employment type, compensation, deadline |
| **Matching logic** | GPA, major, state, ethnicity, first-gen, income | GPA, major, location preference, class year | GPA, major, location preference, experience level |
| **Cover letter tone** | Scholarship applicant — mission alignment, community impact | Internship candidate — skills, projects, company connection | Job candidate — impact, ownership, day-one readiness |
| **Database** | `scholarship_database.csv` | `internship_database.csv` | `job_database.csv` |

Switching modes clears the current session data so results from different targets don't mix.

### Matching & drafting quality

- Search years follow the current calendar (and next year in fall recruiting) — not a hardcoded `2026`.
- Fetches follow safe redirects and prefer `main`/`article` content with a larger HTML window.
- Extractors can return an `apply_url`; cover letters ground on that page when present, plus structured listing facts and sidebar profile (including Name for sign-off).
- Matches are scored and sorted (role fit, explicit major hits, apply URL, description). Empty eligibility is treated as **unknown** (soft-pass, lower rank), not “open to all.”
- Internship/job filters use desired role; On-site means not Remote/Hybrid; Remote + states uses AND.

---

## Stack

| Layer | Tools |
|---|---|
| UI / Orchestration | Streamlit |
| LLM | Gemini, Claude, OpenAI, or Ollama (your choice) |
| Web Search | DuckDuckGo |
| Scraping | BeautifulSoup4, Requests |
| Data | Pandas — CSV flat-file persistence |
| Resume Parsing | pdfplumber |
| Env Management | python-dotenv |

---

## AI Provider

CoScholar works with any of these providers. Select yours from the dropdown in the sidebar.

| Provider | Default Model | Get a Key |
|---|---|---|
| **Gemini** | `gemini-flash-latest` | [aistudio.google.com](https://aistudio.google.com) (free) |
| **Claude** | `claude-3-5-haiku-latest` | [console.anthropic.com](https://console.anthropic.com) |
| **OpenAI** | `gpt-4o-mini` | [platform.openai.com](https://platform.openai.com/api-keys) |
| **Ollama** | `llama3.2` | No key needed — runs locally via [ollama.com](https://ollama.com) |

Paste your key into the API key field in the sidebar. It is used only for that session and is never stored or written to disk.

**For local development**, you can also set keys in a `.env` file instead of entering them in the sidebar:

```
GEMINI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
```

The sidebar key always takes priority over `.env` values.

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

# 3. Run
streamlit run app.py
```

---

## Project Structure

```
coscholar/
├── .streamlit/
│   └── config.toml      # Dark theme config
├── app.py               # Streamlit UI, mode toggle, filtering logic (all modes)
├── agent.py             # Multi-query web discovery — scholarship, internship & job queries
├── llm.py               # Unified AI provider interface (Gemini/Claude/OpenAI/Ollama)
├── pipeline.py          # HTML fetch, clean, SSRF validation, mode-aware orchestration
├── ai_scraper.py        # AI-based structured extraction — scholarship, internship & job schemas
├── drafter.py           # RAG cover letter generation — scholarship, internship & job prompts
└── requirements.txt
```

---

## Security

- API keys are passed directly to provider clients — they are never written to environment variables or logged
- Scraped HTML and AI output is sanitized before rendering to prevent XSS
- Prompts use XML delimiters around untrusted content to mitigate prompt injection
- URLs are validated against private/loopback IP ranges before fetching (SSRF protection)
- Uploaded files are validated by size (5 MB limit) and magic bytes before parsing

---

## Notes

- Works best on static HTML pages. JavaScript-heavy SPAs (React/Angular) return limited results.
- Each mode persists its own database locally (`scholarship_database.csv` / `internship_database.csv` / `job_database.csv`) and auto-loads on browser refresh.
- AI calls use exponential backoff retry (via `tenacity`) to handle rate limits gracefully.
- Filtering logic lives in `app.py` (`filter_scholarship_matches()`, `filter_internship_matches()`, and `filter_job_matches()`). It runs on scout completion and reactively as you update your profile in the sidebar.
- Each draft includes an "Apply here" link to the original source URL so you know where to submit your application.
- The Matches and Scout tables show clickable "Visit" links for every listing's source page.
