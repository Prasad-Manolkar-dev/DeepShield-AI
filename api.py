from fastapi import FastAPI, UploadFile, File
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np

app = FastAPI()

# ==============================
# MODEL
# ==============================

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(512, 1)

model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

print("✅ Model loaded")

# ==============================
# TRANSFORM
# ==============================

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ==============================
# API
# ==============================

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):

    contents = await file.read()

    with open("temp.mp4", "wb") as f:
        f.write(contents)

    cap = cv2.VideoCapture("temp.mp4")

    predictions = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = set(np.linspace(0, total_frames - 1, 25, dtype=int))

    current = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if current not in frame_indices:
            current += 1
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame = transform(frame).unsqueeze(0)

        with torch.no_grad():
            output = model(frame)
            prob = torch.sigmoid(output).item()

        print("Prediction:", prob)

        predictions.append(prob)

        current += 1

    cap.release()

    if len(predictions) == 0:
        return {"error": "No frames processed"}

    fake_votes = sum(1 for p in predictions if p < 0.5)
    real_votes = len(predictions) - fake_votes

    confidence = max(fake_votes, real_votes) / len(predictions)

    return {
        "fake_votes": fake_votes,
        "real_votes": real_votes,
        "confidence": confidence,
        "result": "FAKE" if fake_votes > real_votes else "REAL"
    }