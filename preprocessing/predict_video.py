import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms

# ==============================
# LOAD MODEL (same as training)
# ==============================

resnet = models.resnet18(pretrained=True)
resnet = nn.Sequential(*list(resnet.children())[:-1])

class DeepFakeModel(nn.Module):
    def __init__(self):
        super(DeepFakeModel, self).__init__()
        self.feature_extractor = resnet
        self.fc = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

model = DeepFakeModel()
model.eval()

print("✅ Model loaded")

# ==============================
# VIDEO PATH
# ==============================

video_path = r"C:\Users\Prasad\Desktop\DeepShield-AI\dataset\videos\sample\videoplayback.mp4"

cap = cv2.VideoCapture(video_path)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

predictions = []

# ==============================
# PROCESS VIDEO
# ==============================

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

# ==============================
# FINAL DECISION
# ==============================

avg_prediction = sum(predictions) / len(predictions)

print("Average prediction:", avg_prediction)

if avg_prediction > 0.5:
    print("🔴 FAKE VIDEO")
else:
    print("🟢 REAL VIDEO")