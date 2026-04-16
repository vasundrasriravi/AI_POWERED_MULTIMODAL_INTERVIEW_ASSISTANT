import whisper

model = whisper.load_model("base")  # or "small" if you want higher accuracy

def transcribe_audio(path):
    if not path:
        return ""

    try:
        result = model.transcribe(path)
        return result["text"].strip()
    except Exception as e:
        print("Whisper error:", e)
        return ""
