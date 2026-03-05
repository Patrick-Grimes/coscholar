import pandas as pd
from google import genai
import os
from dotenv import load_dotenv

from pipeline import fetch_and_clean_html

load_dotenv()

API_KEY = os.getenv("API_KEY")
client = genai.Client(api_key=API_KEY)


def draft_application(scholarship_row, resume: str):
    """
    Drafts a personalized cover letter for a scholarship.

    Args:
        scholarship_row: A pandas Series (row) from the matches CSV.
        resume: The applicant's resume/profile as a plain text string.
    """
    scholarship_name = scholarship_row['name']
    url = scholarship_row.get('source_url', '')
    print(f"  -> Reading website: {url}...")

    website_content = ""
    if url and url.startswith("http"):
        website_content = fetch_and_clean_html(url)

    if not website_content:
        website_content = "No website content available. Rely on the scholarship name."

    prompt = f"""
    You are an expert scholarship consultant and professional writer.

    TASK:
    Write a highly specific, 300-word cover letter for the "{scholarship_name}".

    APPLICANT PROFILE:
    {resume}

    SCHOLARSHIP WEBPAGE CONTENT:
    {website_content[:10000]}

    INSTRUCTIONS:
    1. **ROLE & TONE:**
       - Write in first person as the applicant described in the profile above.
       - Tone: Professional, intellectually curious, and grounded.
       - Avoid "fluff" adjectives (e.g., "unwavering," "tapestry," "delve," "crucial").
       - Do NOT sound like a template. Vary sentence structure. Use active voice.

    2. **THE "HOOK" (Paragraph 1):**
       - Start with a direct connection to the scholarship's specific mission from the WEBPAGE CONTENT.
       - Do NOT start with "I am writing to apply for..."

    3. **THE "BRIDGE":**
       - Connect the applicant's most relevant projects/experiences to the scholarship's stated values.
       - Be specific — mention real project names, technologies, or accomplishments from the profile.

    4. **CONSTRAINT CHECK:**
       - Forbidden words: "thrilled," "esteem," "showcase," "realm."
       - Sign off with just the applicant's name.
       - Keep it under 300 words. Be punchy.

    5. **FALLBACK:**
       - If webpage content is empty or generic, focus on the applicant's most technical or impactful project.
    """

    try:
        response = client.models.generate_content(
            model="gemini-flash-latest",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Error drafting application: {e}"


# CLI entry point — prompts for resume input or loads from a file
if __name__ == "__main__":
    try:
        matches = pd.read_csv("matches_to_apply.csv")
    except FileNotFoundError:
        print("No matches found. Run main.py first!")
        exit()

    print("Paste your resume/profile below.")
    print("When done, enter a line with just 'END' and press Enter.\n")

    lines = []
    while True:
        line = input()
        if line.strip().upper() == "END":
            break
        lines.append(line)
    resume_text = "\n".join(lines)

    if not resume_text.strip():
        print("No resume provided. Exiting.")
        exit()

    print(f"\n--- DRAFTING APPLICATIONS FOR {len(matches)} SCHOLARSHIPS ---\n")

    for index, row in matches.iterrows():
        print(f"Writing draft for: {row['name']}...")

        essay = draft_application(row, resume_text)

        safe_name = "".join(x for x in row['name'] if x.isalnum())
        filename = f"Draft_{safe_name}.txt"

        with open(filename, "w", encoding="utf-8") as f:
            f.write(essay)
            f.write("\n\n" + "=" * 20 + "\n")
            f.write(f"Source URL: {row.get('source_url', 'N/A')}\n")
            f.write(f"Award Amount: {row.get('amount', 'N/A')}\n")

        print(f"  -> Saved to {filename}")

    print("\n✅ All drafts created. Good luck applying!")