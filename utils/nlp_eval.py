# nlp_eval.py (STABLE VERSION — NO MORE 700%, 900% BUGS)
import os
import json
import re
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = "gemini-2.5-flash"
URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={API_KEY}"


def normalize(score):
    """Ensure score stays within 0–100."""
    try:
        score = float(score)
    except:
        return 0
    return max(0, min(score, 100))


def auto_scale(value):
    """
    Gemini sometimes returns:
    - 0–1
    - 0–10
    - 0–100

    Auto-detect and normalize to 0–100.
    """
    if value <= 1:      # percentages as decimals
        return value * 100
    if value <= 10:     # 0–10 scale
        return value * 10
    return value        # already in 0–100


def evaluate_all_answers(questions, answers, audio, video):

    clean_audio = [a if a else {} for a in audio]
    clean_video = [v if v else {} for v in video]

    prompt = f"""
You are an expert AI Interview Evaluator.
Evaluate answers using accuracy, communication, voice tone, and emotions based on audio & video data.

Return ONLY valid JSON in this format:

{{
  "overall_scores": {{
    "confidence": 0,
    "communication": 0,
    "accuracy": 0
  }},
  "per_question": [
    {{
      "accuracy": 0,
      "communication": 0,
      "voice_tone": "",
      "facial_expression": "",
      "comment": "",
      "score": 0
    }}
  ]
}}

QUESTIONS: {questions}
ANSWERS: {answers}
AUDIO: {clean_audio}
VIDEO: {clean_video}
"""

    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}

    try:
        response = requests.post(URL, json=payload, timeout=60).json()
        raw = response.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")

        cleaned = raw.replace("```json", "").replace("```", "")

        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(0)

        data = json.loads(cleaned)

        # --------------------------
        # FIX OVERALL SCORES
        # --------------------------
        overall = data.get("overall_scores", {})

        data["overall_scores"] = {
            "confidence": normalize(auto_scale(overall.get("confidence", 0))),
            "communication": normalize(auto_scale(overall.get("communication", 0))),
            "accuracy": normalize(auto_scale(overall.get("accuracy", 0))),
        }

        # --------------------------
        # FIX PER-QUESTION SCORES
        # --------------------------
        fixed_questions = []
        for i, q in enumerate(data.get("per_question", [])):
            fixed_questions.append({
                "accuracy": normalize(auto_scale(q.get("accuracy", 0))),
                "communication": normalize(auto_scale(q.get("communication", 0))),
                "voice_tone": q.get("voice_tone", ""),
                "facial_expression": q.get("facial_expression", ""),
                "comment": q.get("comment", ""),
                "score": normalize(auto_scale(q.get("score", 0))),
                "question": questions[i] if i < len(questions) else "",
                "answer": answers[i] if i < len(answers) else ""
            })

        data["per_question"] = fixed_questions

        # --------------------------
        # RE-CALCULATE TOTAL ACCURACY
        # --------------------------
        if len(fixed_questions) > 0:
            total = sum(q["score"] for q in fixed_questions) / len(fixed_questions)
            data["total_accuracy"] = round(normalize(total), 2)
        else:
            data["total_accuracy"] = 0

        # --------------------------
        # FIT STATUS
        # --------------------------
        acc = data["total_accuracy"]
        if acc >= 80:
            data["fit_status"] = "Strong Fit"
        elif acc >= 50:
            data["fit_status"] = "Moderate Fit"
        else:
            data["fit_status"] = "Needs Improvement"

        return data

    except Exception as e:
        print("❌ NLP Evaluation Error:", e)
        return fallback(questions)


def fallback(qs):
    return {
        "overall_scores": {"confidence": 50, "communication": 50, "accuracy": 50},
        "total_accuracy": 50,
        "fit_status": "Needs Improvement",
        "per_question": [
            {
                "accuracy": 50,
                "communication": 50,
                "voice_tone": "Neutral",
                "facial_expression": "Neutral",
                "comment": "Fallback used.",
                "score": 50
            }
            for _ in qs
        ]
    }
