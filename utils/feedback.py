from fpdf import FPDF
import os

def generate_feedback(filename, session):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    questions = session["questions"]
    answers = session["answers"]
    per_question_scores = session.get("per_question", [])

    total_q = len(questions)
    total_a = len(answers)

    # -------- HEADER --------
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "AI Interview Feedback Report", ln=True)
    pdf.ln(4)

    # -------- PER-QUESTION FEEDBACK --------
    for i in range(total_q):
        pdf.set_font("Arial", "B", 12)
        pdf.multi_cell(0, 7, f"Q{i+1}: {questions[i]}")
        pdf.set_font("Arial", "", 12)

        if i < total_a:
            pdf.multi_cell(0, 7, f"A: {answers[i]}")
        else:
            pdf.multi_cell(0, 7, "A: (No answer provided)")

        if i < len(per_question_scores):
            score = per_question_scores[i].get("score", 0)
            comment = per_question_scores[i].get("comment", "")
        else:
            score = 0
            comment = "No feedback available."

        pdf.multi_cell(0, 7, f"Score: {score}")
        pdf.multi_cell(0, 7, f"Feedback: {comment}")
        pdf.ln(4)

    # -------- FINAL RECOMMENDATION (IMPORTANT: BEFORE OUTPUT) --------
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Final Recommendation", ln=True)

    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 7, f"Overall Score: {session.get('final_score', 0)}%")
    pdf.multi_cell(0, 7, f"Fit Status: {session.get('fit_status', 'Unknown')}")

    # -------- SAVE PDF --------
    path = f"static/{filename}.pdf"
    pdf.output(path)

    return path
