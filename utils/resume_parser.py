import google.generativeai as genai
from PyPDF2 import PdfReader
from io import BytesIO
import json
import os

# Load key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


# -------------------------------------------
# 1. Extract text from uploaded resume (PDF)
# -------------------------------------------
def extract_text_from_pdf(pdf_bytes):
    pdf_stream = BytesIO(pdf_bytes)
    reader = PdfReader(pdf_stream)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text.strip()


# -------------------------------------------
# 2. Generate questions using Gemini 2.0 Flash
# -------------------------------------------
def generate_all_questions(resume_text, role, tech_n, hr_n):
    model = genai.GenerativeModel("gemini-2.0-flash")

    prompt = f"""
    You are an AI Interview Question generator.

    Candidate Resume:
    {resume_text}

    Job Role: {role}

    Generate:
    - {tech_n} technical questions
    - {hr_n} HR questions

    Format your output EXACTLY like this JSON:

    {{
        "technical": ["q1", "q2", "q3"],
        "resume_based": ["q1", "q2"],
        "hr": ["q1", "q2"]
    }}
    """

    response = model.generate_content(prompt)

    print("RAW GEMINI OUTPUT:", response.text)

    try:
        cleaned = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned)
    except Exception as e:
        print("JSON ERROR:", e)
        raise ValueError("Failed to generate technical questions")
