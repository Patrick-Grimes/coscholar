import os

PROVIDERS = ["Gemini", "Claude", "OpenAI", "Ollama"]

DEFAULT_MODELS = {
    "Gemini": "gemini-flash-latest",
    "Claude": "claude-3-5-haiku-latest",
    "OpenAI": "gpt-4o-mini",
    "Ollama": "llama3.2",
}


def call_llm(prompt: str, provider: str, api_key: str = None, ollama_host: str = None) -> str:
    """
    Send a prompt to the selected LLM provider and return the response text.

    Args:
        prompt:       The full prompt string.
        provider:     One of "Gemini", "Claude", "OpenAI", "Ollama".
        api_key:      API key for cloud providers. Falls back to env vars if not given.
        ollama_host:  Base URL for Ollama (default: http://localhost:11434).

    Returns:
        Response text from the model, or raises on error.
    """
    provider = provider.strip()

    if provider == "Gemini":
        return _call_gemini(prompt, api_key)
    elif provider == "Claude":
        return _call_claude(prompt, api_key)
    elif provider == "OpenAI":
        return _call_openai(prompt, api_key)
    elif provider == "Ollama":
        return _call_ollama(prompt, ollama_host)
    else:
        raise ValueError(f"Unknown provider: {provider}. Choose from {PROVIDERS}.")


def _call_gemini(prompt: str, api_key: str = None) -> str:
    from google import genai

    key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("API_KEY")
    if not key:
        raise ValueError("No Gemini API key provided. Get one free at aistudio.google.com.")
    if not key.startswith("AIza"):
        raise ValueError("Invalid Gemini API key format.")

    client = genai.Client(api_key=key)
    response = client.models.generate_content(
        model=DEFAULT_MODELS["Gemini"],
        contents=prompt,
    )
    return response.text


def _call_claude(prompt: str, api_key: str = None) -> str:
    import anthropic

    key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError("No Claude API key provided. Get one at console.anthropic.com.")

    client = anthropic.Anthropic(api_key=key)
    message = client.messages.create(
        model=DEFAULT_MODELS["Claude"],
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def _call_openai(prompt: str, api_key: str = None) -> str:
    from openai import OpenAI

    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("No OpenAI API key provided. Get one at platform.openai.com.")

    client = OpenAI(api_key=key)
    response = client.chat.completions.create(
        model=DEFAULT_MODELS["OpenAI"],
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def _call_ollama(prompt: str, ollama_host: str = None) -> str:
    import ollama as ollama_client

    host = ollama_host or os.getenv("OLLAMA_HOST") or "http://localhost:11434"
    client = ollama_client.Client(host=host)
    response = client.chat(
        model=DEFAULT_MODELS["Ollama"],
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]