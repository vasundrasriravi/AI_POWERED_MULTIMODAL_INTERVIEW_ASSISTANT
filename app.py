import os
import json
import tempfile
from datetime import datetime

from flask import Flask, render_template, request, redirect, jsonify, send_file
from dotenv import load_dotenv
import google.generativeai as genai

import cv2
import soundfile as sf
from moviepy.editor import VideoFileClip

# Local utils
from utils.audio_analysis import AudioAnalyzer
from utils.video_analysis import VideoAnalyzer
from utils.nlp_eval import evaluate_all_answers
from utils.feedback import generate_feedback

# -----------------------------------------------------------
# INITIAL SETUP
# -----------------------------------------------------------

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = Flask(__name__)
os.makedirs("static/recordings", exist_ok=True)

audio_analyzer = AudioAnalyzer()
video_analyzer = VideoAnalyzer()


# -----------------------------------------------------------
# PDF TEXT EXTRACTOR
# -----------------------------------------------------------

def extract_text_from_pdf(file_stream):
    import PyPDF2
    pdf_reader = PyPDF2.PdfReader(file_stream)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text


# -----------------------------------------------------------
# QUESTION GENERATOR
# -----------------------------------------------------------

def generate_questions(resume_text, role, tech_n, hr_n):
    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = f"""
    Generate {tech_n} technical and {hr_n} HR interview questions
    for the role: {role}.

    Rules:
    - Very short, 8–12 words max
    - Simple English, easy to speak
    - No bullets, no numbering, no lists
    - Each question in a separate line
    - Use this resume text for context:
      {resume_text}
    """

    resp = model.generate_content(prompt)
    lines = resp.text.split("\n")
    cleaned = []
    for q in lines:
        q = q.strip().lstrip("-•0123456789. ").strip()
        if len(q.split()) >= 5:
            cleaned.append(q)

    return cleaned[:tech_n + hr_n]


# -----------------------------------------------------------
# HOME PAGE
# -----------------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        pdf_file = request.files["resume"]
        role = request.form["role"]
        tech_n = int(request.form["tech_n"])
        hr_n = int(request.form["hr_n"])

        pdf_bytes = pdf_file.read()
        stream = tempfile.TemporaryFile()
        stream.write(pdf_bytes)
        stream.seek(0)   

        resume_text = extract_text_from_pdf(stream)
        questions = generate_questions(resume_text, role, tech_n, hr_n)

        session = {
            "questions": questions,
            "index": 0,
            "answers": [],
            "audio": [],
            "video": []
        }

        with open("session.json", "w") as f:
            json.dump(session, f)

        return redirect("/interview")

    return render_template("index.html")


# -----------------------------------------------------------
# INTERVIEW PAGE
# -----------------------------------------------------------

@app.route("/interview")
def interview():
    with open("session.json", "r") as f:
        session = json.load(f)

    idx = session["index"]
    if idx >= len(session["questions"]):
        return redirect("/results")

    return render_template("interview.html",
                           question=session["questions"][idx],
                           index=idx)


# -----------------------------------------------------------
# SUBMIT ANSWER + LIGHT COACHING
# -----------------------------------------------------------

@app.route("/submit_answer", methods=["POST"])
def submit_answer():

    typed = request.form.get("typed", "").strip()
    index = int(request.form.get("index"))
    video_file = request.files.get("video")

    with open("session.json", "r") as f:
        session = json.load(f)

    transcript = ""
    audio_data = {}
    coaching_tip = None

    # ---------- VIDEO + AUDIO PROCESSING ----------
    if video_file:
        video_path = f"static/recordings/answer_{index}.webm"
        video_file.save(video_path)

        # VIDEO ANALYSIS (MULTIPLE FRAMES)
        frame_result = video_analyzer.analyze_multiple_frames(video_path)
        session["video"].append(frame_result)

        # AUDIO EXTRACTION
        try:
            wav_path = f"static/recordings/answer_{index}.wav"
            clip = VideoFileClip(video_path)
            clip.audio.write_audiofile(wav_path, fps=16000)

            samples, sr = sf.read(wav_path)
            audio_data = audio_analyzer.analyze_audio(samples, sr)
            session["audio"].append(audio_data)

            transcript = audio_data.get("transcript", "")

        except Exception as e:
            print("Audio processing error:", e)
            session["audio"].append({})
    else:
        # no video/audio for this answer
        session["video"].append({})
        session["audio"].append({})

    # ---------- FINAL ANSWER (TYPED OR SPOKEN) ----------
    final_answer = typed if typed else transcript
    session["answers"].append(final_answer)

    # ---------- SIMPLE COACHING RULE ----------
    # If answer is very short or speech is too small => give coaching message
    word_count = len(final_answer.split())
    speech_duration = audio_data.get("speech_duration_sec", 0.0) if audio_data else 0.0
    silence_ratio = audio_data.get("silence_ratio", 1.0) if audio_data else 1.0

    if word_count < 5 or speech_duration < 4 or silence_ratio > 0.75:
        coaching_tip = (
            "Your answer was quite short / had many pauses. "
            "Next time, try to give more detailed explanation and speak more continuously."
        )

    # ---------- MOVE TO NEXT QUESTION ----------
    session["index"] += 1

    with open("session.json", "w") as f:
        json.dump(session, f)

    return jsonify({"status": "ok", "coaching_tip": coaching_tip})


# -----------------------------------------------------------
# RESULTS PAGE + HISTORY LOGGING
# -----------------------------------------------------------

@app.route("/results")
def results():
    with open("session.json", "r") as f:
        session = json.load(f)

    result = evaluate_all_answers(
        session["questions"],
        session["answers"],
        session["audio"],
        session["video"]
    )

    # ---------- FIX SCORE CALCULATION (NO MORE 63/10 OR 0.8/10) ----------
    for idx, q in enumerate(result.get("per_question", [])):
        acc = q.get("accuracy", 0)
        comm = q.get("communication", 0)

        # convert to 0-10 score correctly: (acc + comm)/20
        raw_score = (acc + comm) / 20
        score_10 = round(raw_score)  # round to whole number like 8/10

        q["score"] = score_10

    # compute final score from individual scores
    scores = [q.get("score", 0) for q in result.get("per_question", [])]
    if len(scores) > 0:
        total_accuracy = round((sum(scores) / len(scores)) * 10, 2)
    else:
        total_accuracy = 0

    result["total_accuracy"] = total_accuracy

    if total_accuracy >= 80:
        result["fit_status"] = "Strong Fit"
    elif total_accuracy >= 50:
        result["fit_status"] = "Moderate Fit"
    else:
        result["fit_status"] = "Needs Improvement"


    # Ensure basic structure exists
    overall_scores = result.get("overall_scores", {
        "confidence": 0,
        "communication": 0,
        "accuracy": 0
    })
    result["overall_scores"] = overall_scores

    fit_status = result.get("fit_status", "Needs Improvement")
    total_accuracy = result.get("total_accuracy", 0)

    result["fit_status"] = fit_status
    result["total_accuracy"] = total_accuracy

    # ---------- PDF REPORT ----------
    pdf_path = generate_feedback("final_report", {
        "questions": session["questions"],
        "answers": session["answers"],
        "per_question": result.get("per_question", []),
        "final_score": total_accuracy,
        "fit_status": fit_status
    })

    # ---------- SAVE ATTEMPT HISTORY ----------
    history_path = os.path.join(app.root_path, "history.json")
    history = {"attempts": []}

    if os.path.exists(history_path):
        try:
            with open(history_path, "r") as f:
                history = json.load(f)
        except Exception as e:
            print("History read error:", e)
            history = {"attempts": []}

    attempt_record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "total_accuracy": total_accuracy,
        "confidence": overall_scores.get("confidence", 0),
        "communication": overall_scores.get("communication", 0),
        "answer_accuracy": overall_scores.get("accuracy", 0),
        "fit_status": fit_status
    }
    history["attempts"].append(attempt_record)

    try:
        with open(history_path, "w") as f:
            json.dump(history, f)
    except Exception as e:
        print("History write error:", e)

    return render_template("result.html", result=result, pdf_path=pdf_path)


# -----------------------------------------------------------
# PROGRESS PAGE
# -----------------------------------------------------------

@app.route("/progress")
def progress():
    history_path = os.path.join(app.root_path, "history.json")
    history = {"attempts": []}

    if os.path.exists(history_path):
        try:
            with open(history_path, "r") as f:
                history = json.load(f)
        except Exception as e:
            print("History read error:", e)
            history = {"attempts": []}

    return render_template("progress.html", history=history)


# -----------------------------------------------------------
# DOWNLOAD PDF
# -----------------------------------------------------------

@app.route("/download")
def download():
    filename = request.args.get("file")
    final_path = os.path.join(app.root_path, filename)
    return send_file(final_path, as_attachment=True)


# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
