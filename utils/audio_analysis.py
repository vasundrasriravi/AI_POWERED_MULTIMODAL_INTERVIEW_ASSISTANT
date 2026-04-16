# audio_analysis.py (STABLE VERSION)
import numpy as np
import tempfile
import soundfile as sf
from faster_whisper import WhisperModel

class AudioAnalyzer:

    def __init__(self):
        # Medium multilingual = BEST for Indian English
        self.model = WhisperModel(
            "medium", 
            device="cpu", 
            compute_type="int8"
        )

    def analyze_audio(self, samples, sr=16000):

        if samples is None or len(samples) == 0:
            return self.empty_audio()

        samples = np.array(samples, dtype="float32")

        # --------------------------
        # AUDIO METRICS (STABLE)
        # --------------------------
        rms = float(np.sqrt(np.mean(samples**2)))

        # relaxed threshold
        silence_threshold = 0.015

        silence_ratio = float(np.mean(np.abs(samples) < silence_threshold))

        # soft speaking detection allowed
        speech_detected = rms > 0.01

        speech_samples = np.sum(np.abs(samples) > silence_threshold)
        speech_duration = round(speech_samples / sr, 2)

        # --------------------------
        # TRANSCRIPTION (STABLE)
        # --------------------------
        transcript = ""

        try:
            tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(tmp_wav.name, samples, sr)

            segments, info = self.model.transcribe(
                tmp_wav.name,
                language="en",
                beam_size=1,       # more stable, less hallucination
                vad_filter=False   # stop cutting first/last words
            )

            transcript = " ".join(seg.text.strip() for seg in segments).strip()

        except Exception as e:
            print("Whisper STT Error:", e)
            transcript = ""

        # fallback if whisper produced empty text
        if len(transcript) < 2 and speech_detected:
            transcript = "[audio detected but transcription unclear]"

        return {
            "transcript": transcript,
            "rms": round(rms, 3),
            "silence_ratio": round(silence_ratio, 3),
            "speech_detected": speech_detected,
            "speech_duration_sec": speech_duration
        }

    def empty_audio(self):
        return {
            "transcript": "",
            "rms": 0,
            "silence_ratio": 1.0,
            "speech_detected": False,
            "speech_duration_sec": 0
        }
