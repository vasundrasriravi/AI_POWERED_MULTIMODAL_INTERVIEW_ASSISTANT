# video_analysis.py
import cv2
import numpy as np

class VideoAnalyzer:

    def __init__(self):
        # DNN face detector (MUCH MORE STABLE than Haarcascade)
        modelFile = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_detector = cv2.CascadeClassifier(modelFile)

    def analyze_multiple_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            return self.empty_result()

        results = []
        sample_interval = max(1, total_frames // 15)       # sample 15 frames minimum

        for pos in range(0, total_frames, sample_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()

            if not ret or frame is None:
                continue

            frame = cv2.resize(frame, (640, 480))
            r = self.analyze_frame(frame)

            if r.get("face_detected"):
                results.append(r)

        cap.release()

        if len(results) == 0:
            return self.empty_result()

        # choose frame with best clarity
        best = max(results, key=lambda x: x.get("clarity", 0))
        return best

    def empty_result(self):
        return {
            "face_detected": False,
            "emotion": "Unknown",
            "confidence": 0,
            "clarity": 0,
            "brightness": 0,
            "edge_density": 0,
            "eye_contact": 0
        }

    def analyze_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )

        if len(faces) == 0:
            return {"face_detected": False}

        x, y, w, h = faces[0]

        # face ROI
        face = gray[y:y+h, x:x+w]
        if face.size == 0:
            return {"face_detected": False}

        # clarity
        clarity = float(np.std(face))

        # brightness
        brightness = float(np.mean(face))

        # edges
        edges = cv2.Canny(face, 50, 150)
        edge_density = float(np.mean(edges) / 255.0)

        # safe eye region
        h1 = max(1, h // 6)
        h2 = max(h1 + 1, h // 2)
        w1 = max(1, w // 6)
        w2 = max(w1 + 1, 5 * w // 6)

        eye_area = face[h1:h2, w1:w2]

        if eye_area.size == 0:
            eye_clarity = 0
        else:
            eye_clarity = float(np.std(eye_area))

        # emotion detection (simple rule)
        if brightness > 135:
            emotion = "Happy"
        elif edge_density > 0.12:
            emotion = "Confused"
        else:
            emotion = "Neutral"

        # normalize confidence
        confidence_raw = clarity * 0.5 + eye_clarity * 0.3 + brightness * 0.2
        confidence = round(min(100, confidence_raw / 3), 2)

        return {
            "face_detected": True,
            "emotion": emotion,
            "confidence": confidence,
            "clarity": round(clarity, 2),
            "brightness": round(brightness, 2),
            "edge_density": round(edge_density, 2),
            "eye_contact": round(eye_clarity, 2)
        }
