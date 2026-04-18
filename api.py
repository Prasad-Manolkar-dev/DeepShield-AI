from fastapi import FastAPI, UploadFile, File
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import mediapipe as mp

app = FastAPI()

# ==============================
# LOAD TRAINED MODEL (ResNet)
# ==============================

model = models.resnet18(pretrained=False)

model.fc = nn.Sequential(
    nn.Linear(512, 1),
    nn.Sigmoid()
)

model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
model.eval()

print("✅ Trained ResNet model loaded")

# ==============================
# TRANSFORM
# ==============================

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ==============================
# FACE DETECTOR
# ==============================

mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# ==============================
# VIDEO PREDICTION
# ==============================

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):

    contents = await file.read()

    with open("temp.mp4", "wb") as f:
        f.write(contents)

    cap = cv2.VideoCapture("temp.mp4")

    predictions = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        return {"error": "Invalid video"}

    # sample 20 frames
    frame_indices = set(np.linspace(0, total_frames - 1, 20, dtype=int))

    current_frame = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame not in frame_indices:
            current_frame += 1
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_detector.process(frame_rgb)

        if results.detections:
            for detection in results.detections:

                bbox = detection.location_data.relative_bounding_box

                h, w, _ = frame.shape

                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                x2 = int((bbox.xmin + bbox.width) * w)
                y2 = int((bbox.ymin + bbox.height) * h)

                # clamp
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                face = frame_rgb[y1:y2, x1:x2]

                if face.size == 0:
                    continue

                face = transform(face).unsqueeze(0)

                with torch.no_grad():
                    output = model(face)

                predictions.append(output.item())

        current_frame += 1

    cap.release()

    if len(predictions) < 5:
        return {"error": "Not enough valid face frames"}

    # ==============================
    # VOTING LOGIC (ROBUST)
    # ==============================

    fake_votes = sum(1 for p in predictions if p > 0.5)
    real_votes = len(predictions) - fake_votes

    confidence = fake_votes / len(predictions)

    return {
        "fake_votes": fake_votes,
        "real_votes": real_votes,
        "confidence": confidence,
        "result": "FAKE" if fake_votes > real_votes else "REAL"
    }