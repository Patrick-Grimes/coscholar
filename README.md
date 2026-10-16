# CoScholar - Autonomous Scholarship, Internship & Job Agent

🔗 [Live Demo](https://coscholar-ai.streamlit.app)

An end-to-end agentic pipeline that scouts the web for **scholarships, internships, or jobs**, filters them against a student/candidate profile, and drafts personalized cover letters using RAG (Retrieval-Augmented Generation).

Built with Python, Streamlit, BeautifulSoup, and your choice of AI provider.

---

## How It Works

| Step | What Happens |
|---|---|
| **Profile** | Enter role/field (career) or eligibility (scholarship) plus resume in the main panel |
| **Find** | Board-first DuckDuckGo `site:` search for jobs/internships (LinkedIn, Indeed, Greenhouse, Lever, Ashby, Workday, Simplify, Handshake…); scholarships stay open-web |
| **Extract** | Fetches each page, coerces LLM JSON into durable listings (company + role + description/apply URL), logs drop reasons |
| **Filter** | Multi-axis Pandas filtering against your profile. Expired listings are dropped by deadline parsing |
| **Draft** | Re-fetches each match's apply/detail page and uses RAG to write a personalized cover letter grounded in your resume |

---

## Modes

**Sidebar** switches mode (**Job** / **Internship** / **Scholarship**) and holds AI provider + key plus advanced scout budget. **Main panel** is the product loop: profile → resume → Find → Matches → Drafts.

| | Scholarship | Internship | Job |
|---|---|---|---|
| **Branding** | CoScholar AI | CoIntern AI | CoJobs AI |
| **Primary drivers** | Major + GPA | Desired role + field | Desired role + field |
| **GPA** | Required | Optional | Optional |
| **Search** | Open-web eligibility angles | Board-first + small open-web fallback | Board-first + small open-web fallback |
| **Extraction** | Coerce + keep if name + description/apply | Coerce + keep if company + role + description/apply | Same as internship |
| **Matching** | GPA, major, state, ethnicity, first-gen, income | Role, field, location, class year | Role, field, location, experience |
| **Cover letter tone** | Mission / community impact | Skills, projects, company fit | Impact, ownership, day-one readiness |

Switching modes clears the current session data so results from different targets don't mix.

### Board-first career discovery

Job and Internship modes prefer `site:` queries for:

- linkedin.com/jobs, indeed.com, glassdoor.com, wellfound.com
- greenhouse.io, lever.co / jobs.lever.co, ashbyhq.com
- Public Workday (`myworkdayjobs.com` / jobs paths)
- simplify.jobs; Handshake (internships)

URLs are ranked so board/ATS domains win; Reddit/blogs/news are demoted. Total unique URLs stay capped around **24**, with budget spent on board hits first and slightly deeper depth on board angles.

### Matching & drafting quality

- Search years follow the current calendar (and next year in fall recruiting).
- Fetches follow safe redirects and prefer `main`/`article` content with a larger HTML window.
- Extractors coerce types (strings→lists, parse `min_gpa`) and **do not** require every schema field to be perfectly typed — survival needs company + role and a description or apply URL.
- Per-URL stats (`fetched` / `extracted` / `validation_dropped` / `deadline_filtered`) surface in the Find status so “0 active” is explainable.
- Extractors can return an `apply_url`; when missing, the scouted page is kept as a fallback link for drafting.
- Matches are scored and sorted (role fit, explicit major hits, apply URL, description). Empty eligibility is treated as **unknown** (soft-pass, lower rank), not “open to all.”

---

## Stack

| Layer | Tools |
|---|---|
| UI / Orchestration | Streamlit |
| LLM | Gemini, Claude, OpenAI, or Ollama (your choice) |
| Web Search | DuckDuckGo |
| Scraping | BeautifulSoup4, Requests |
| Data | Pandas — in-memory session results |
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

### Profile steps (main panel)

1. Pick **Job**, **Internship**, or **Scholarship** in the sidebar (plus AI key).
2. Fill **You**: name, role + field (career) or major + GPA (scholarship), location / experience or class year, resume upload or paste.
3. Click **Find jobs** / **Find internships** / **Find scholarships** and watch status counters if results are empty.
4. Review **Matches**, then **Drafts** for top-N cover letters.

---

## Project Structure

```
coscholar/
├── .streamlit/
│   └── config.toml      # Dark theme config
├── app.py               # Streamlit UI — sidebar controls + main profile/Find loop
├── agent.py             # Board-first job/internship discovery; open-web scholarships
├── llm.py               # Unified AI provider interface (Gemini/Claude/OpenAI/Ollama)
├── pipeline.py          # HTML fetch, clean, SSRF validation, stats-aware orchestration
├── ai_scraper.py        # Coerce + extract scholarship / internship / job schemas
├── drafter.py           # RAG cover letter generation
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
- Session results stay in browser memory for the current mode; switching modes clears them.
- AI calls use exponential backoff retry (via `tenacity`) to handle rate limits gracefully.
- Filtering logic lives in `app.py` (`filter_scholarship_matches()`, `filter_internship_matches()`, and `filter_job_matches()`).
- Each draft includes an "Apply here" link (preferring extracted `apply_url`) so you know where to submit.
- The Matches and Find tables show clickable Visit/Apply links for every listing.
