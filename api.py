from fastapi import FastAPI, UploadFile, File
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np

app = FastAPI()

# ==============================
# LOAD MODEL
# ==============================

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(512, 1)

model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

print("✅ Model loaded")

# ==============================
# TRANSFORM (MUST MATCH TRAINING)
# ==============================

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ==============================
# PREDICTION API
# ==============================

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):

    contents = await file.read()

    with open("temp.mp4", "wb") as f:
        f.write(contents)

    cap = cv2.VideoCapture("temp.mp4")

    predictions = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = set(np.linspace(0, total_frames - 1, 30, dtype=int))

    current = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if current not in frame_indices:
            current += 1
            continue

        # 🔥 FIX: BGR → RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame = transform(frame).unsqueeze(0)

        with torch.no_grad():
            output = model(frame)
            prob = torch.sigmoid(output).item()

        print("Prediction:", prob)

        # 🔥 Ignore weak predictions
        if 0.4 < prob < 0.6:
            current += 1
            continue

        predictions.append(prob)

        current += 1

    cap.release()

    # ==============================
    # SAFETY CHECK
    # ==============================

    if len(predictions) == 0:
        return {"error": "No confident frames detected"}

    # ==============================
    # CALIBRATED DECISION (FINAL FIX)
    # ==============================

    avg_prob = sum(predictions) / len(predictions)

    print("Average probability:", avg_prob)

    # 🔥 IMPORTANT: calibrated threshold
    threshold = 0.97

    result = "REAL" if avg_prob > threshold else "FAKE"

    confidence = abs(avg_prob - threshold)

    return {
        "average_probability": avg_prob,
        "threshold": threshold,
        "confidence": confidence,
        "result": result
    }