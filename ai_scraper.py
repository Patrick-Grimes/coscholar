import os
from google import genai
from dotenv import load_dotenv
from google.genai import types 
import json

# Find .env file and load the variables
load_dotenv()

API_KEY = os.getenv("API_KEY")
client = genai.Client(api_key=API_KEY)

messy_html_example = """
<div class="funding-box">
    <h3>The Ada Lovelace Grant</h3>
    <p>We are offering approx $20,000 to women in STEM.</p>
    <small>Deadline: Oct 15th. Must have 3.5 GPA.</small>
    <div>Eligibility: CS or Math majors only.</div>
</div>
"""

def extract_scholarship_data(html_snippet):
    # The prompt MUST be inside the function so it can use the html_snippet
    prompt = f"""
    You are a data extraction robot.
    Analyze the following HTML and extract ALL scholarship opportunities found.

    HTML:
    {html_snippet}

    Return ONLY a raw JSON object (no markdown) with a single key "scholarships" that is a list of objects.
    For each scholarship, extract every eligibility field you can find. Use empty string or null for fields not mentioned.

    Fields to extract:
    - name: scholarship name (string)
    - amount: award as a number only, no dollar sign (number or null)
    - deadline: deadline as a string (string)
    - min_gpa: minimum GPA as a decimal (number or null)
    - majors: eligible majors, or empty list if open to all (list of strings)
    - eligible_states: eligible US states as 2-letter codes, or empty list if nationwide (list of strings)
    - ethnicity: required ethnicity/race if restricted, or "" if open to all (string)
    - first_gen: true if first-generation students only, false otherwise (boolean)
    - income_based: true if need-based/financial need required, false otherwise (boolean)
    - description: one sentence summary of the scholarship mission (string)

    Example:
    {{
        "scholarships": [
            {{
                "name": "Example Award",
                "amount": 5000,
                "deadline": "Oct 15",
                "min_gpa": 3.5,
                "majors": ["Computer Science"],
                "eligible_states": ["NC"],
                "ethnicity": "",
                "first_gen": false,
                "income_based": false,
                "description": "Supports STEM students in the Southeast."
            }}
        ]
    }}
    """
    
    try:
        # Use a specific model version
        response = client.models.generate_content(
            model="gemini-flash-latest", 
            contents=prompt
        )
        
        # Clean up the text (remove json wrapper)
        clean_json_text = response.text.replace("```json", "").replace("```", "").strip()
        
        return json.loads(clean_json_text)
        
    except Exception as e:
        print(f"Error!!!: {e}")
        return {"scholarships": []}

if __name__ == "__main__":
    print("STARTING")
    data = extract_scholarship_data(messy_html_example)

    print("\n--- EXTRACTED DATA ---")
    print(json.dumps(data, indent=2))

    if data.get('amount', 0) > 5000:
        print(f"\n[System] High Value Alert! Found a ${data.get('amount')} scholarship.")