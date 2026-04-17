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

resnet = models.resnet18(pretrained=True)
resnet = nn.Sequential(*list(resnet.children())[:-1])

class DeepFakeModel(nn.Module):
    def __init__(self):
        super(DeepFakeModel, self).__init__()

        self.feature_extractor = resnet

        # freeze all layers
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # unfreeze last layer
        for param in list(self.feature_extractor.children())[-1].parameters():
            param.requires_grad = True

        self.fc = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

model = DeepFakeModel()

# LOAD TRAINED WEIGHTS
model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
model.eval()

print("✅ Trained model loaded")

# ==============================
# TRANSFORM
# ==============================

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ==============================
# VIDEO PREDICTION ENDPOINT
# ==============================

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):

    contents = await file.read()

    # Save uploaded video temporarily
    with open("temp.mp4", "wb") as f:
        f.write(contents)

    cap = cv2.VideoCapture("temp.mp4")

    predictions = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = transform(frame).unsqueeze(0)

        with torch.no_grad():
            output = model(frame)

        predictions.append(output.item())

    cap.release()

    if len(predictions) == 0:
        return {"error": "No frames processed"}

    avg_prediction = sum(predictions) / len(predictions)

    return {
        "average_prediction": avg_prediction,
        "result": "FAKE" if avg_prediction > 0.5 else "REAL"
    }