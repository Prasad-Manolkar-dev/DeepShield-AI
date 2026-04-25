import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import random

# ==============================
# LOAD MODEL
# ==============================

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(512, 1)

model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

print("✅ Model loaded")

# ==============================
# TRANSFORM (same as training)
# ==============================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ==============================
# PATHS
# ==============================

real_path = r"C:\Users\Prasad\Desktop\DeepShield-AI\dataset\train\Real"
fake_path = r"C:\Users\Prasad\Desktop\DeepShield-AI\dataset\train\Fake"

# ==============================
# TEST FUNCTION
# ==============================

def test_folder(path, label_name):
    files = os.listdir(path)
    samples = random.sample(files, 10)

    print(f"\n🔍 Testing {label_name} images:\n")

    for file in samples:
        img_path = os.path.join(path, file)

        img = Image.open(img_path).convert("RGB")
        img = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = model(img)
            prob = torch.sigmoid(output).item()

        print(f"{file[:20]}... → {prob:.4f}")

# ==============================
# RUN TEST
# ==============================

test_folder(real_path, "REAL")
test_folder(fake_path, "FAKE")