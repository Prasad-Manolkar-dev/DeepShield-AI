import os
import cv2
import torch
import numpy as np
from torchvision import models, transforms

# ==============================
# LOAD MODEL
# ==============================

model = models.resnet18(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()

print("✅ ResNet loaded")

# ==============================
# PATHS
# ==============================

input_folder = r"C:\Users\Prasad\Desktop\DeepShield-AI\dataset\faces\sample_video"
output_file = r"C:\Users\Prasad\Desktop\DeepShield-AI\dataset\features.npy"

files = os.listdir(input_folder)

print("Total face images:", len(files))

# ==============================
# TRANSFORM
# ==============================

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ==============================
# FEATURE STORAGE
# ==============================

all_features = []

# ==============================
# PROCESS ALL IMAGES
# ==============================

for file in files:
    path = os.path.join(input_folder, file)

    img = cv2.imread(path)

    if img is None:
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = transform(img)
    img = img.unsqueeze(0)

    with torch.no_grad():
        features = model(img)

    features = features.view(-1).numpy()

    all_features.append(features)

print("Total features extracted:", len(all_features))

# ==============================
# SAVE FEATURES
# ==============================

all_features = np.array(all_features)

np.save(output_file, all_features)

print("✅ Features saved to file")
print("Final shape:", all_features.shape)