# CoScholar — Autonomous Scholarship Agent

🔗 [Live Demo](https://coscholar-ai.streamlit.app)

An end-to-end agentic pipeline that scouts the web for scholarships, filters them against a student profile, and drafts personalized cover letters using RAG (Retrieval-Augmented Generation).

Built with Python, Streamlit, BeautifulSoup, and your choice of AI provider.

---

## How It Works

| Step | What Happens |
|---|---|
| **Scout** | Runs multiple targeted searches (by major, state, ethnicity, first-gen status, etc.) via DuckDuckGo and deduplicates URLs across all queries |
| **Extract** | Fetches each page, strips JS/CSS to reduce tokens, sends clean HTML to your chosen AI for structured JSON extraction — name, amount, GPA, deadline, and eligibility fields |
| **Filter** | Multi-axis Pandas filtering: GPA, major, state, ethnicity, first-gen, income-based. Expired scholarships are dropped automatically by deadline parsing |
| **Draft** | Re-fetches each matched scholarship's live page and uses RAG to write a personalized cover letter grounded in your uploaded resume |

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
| **Gemini** | gemini-flash-latest | [aistudio.google.com](https://aistudio.google.com) (free) |
| **Claude** | claude-3-5-haiku-latest | [console.anthropic.com](https://console.anthropic.com) |
| **OpenAI** | gpt-4o-mini | [platform.openai.com](https://platform.openai.com/api-keys) |
| **Ollama** | llama3.2 | No key needed — runs locally via [ollama.com](https://ollama.com) |

Paste your key into the API key field in the sidebar. It is used only for that session and is never stored or written to disk.

**For local development**, you can also set keys in a \\.env\\ file instead of entering them in the sidebar:

\\\`n
GEMINI_API_KEY=your_key_here

ANTHROPIC_API_KEY=your_key_here

OPENAI_API_KEY=your_key_here

\\\`n
The sidebar key always takes priority over \\.env\\ values.

> **Note:** Never commit your \\.env\\ file. It is gitignored by default.

---

## Setup

\\\ash
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
\\\`n
---

## Project Structure

\\\`ncoscholar/
├── .streamlit/
│   └── config.toml      # Dark theme config
├── app.py               # Streamlit UI + filtering logic
├── agent.py             # Multi-query web discovery (DuckDuckGo)
├── llm.py               # Unified AI provider interface (Gemini/Claude/OpenAI/Ollama)
├── pipeline.py          # HTML fetch, clean, SSRF validation, orchestration
├── ai_scraper.py        # AI-based structured extraction + schema validation
├── drafter.py           # RAG cover letter generation
└── requirements.txt
\\\`n
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
- The scholarship database persists as \\scholarship_database.csv\\ locally and auto-loads on browser refresh.
- AI calls use exponential backoff retry (via \\	enacity\\) to handle rate limits gracefully.
- Filtering logic lives in \\pp.py\\ (\\ilter_matches()\\). It runs on scout completion and reactively as you update your profile in the sidebar.
