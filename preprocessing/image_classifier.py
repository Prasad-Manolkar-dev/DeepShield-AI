from torchvision import transforms
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import TensorDataset, DataLoader

print(" SCRIPT STARTED")

# ==============================
# PATHS
# ==============================

real_path = r"C:\Users\Prasad\Desktop\DeepShield-AI\dataset\train\real"
fake_path = r"C:\Users\Prasad\Desktop\DeepShield-AI\dataset\train\fake"

data = []
labels = []

# ==============================
# LOAD REAL IMAGES (label = 0)
# ==============================

print("Loading real images...")
files = os.listdir(real_path)
files = files[:1000]   # limit for performance

for file in files:
    path = os.path.join(real_path, file)

    img = cv2.imread(path)

    if img is None:
        continue

    img = cv2.resize(img, (224, 224))

    data.append(img)
    labels.append(0)

# ==============================
# LOAD FAKE IMAGES (label = 1)
# ==============================

print("Loading fake images...")
files = os.listdir(fake_path)
files = files[:1000]

for file in files:
    path = os.path.join(fake_path, file)

    img = cv2.imread(path)

    if img is None:
        continue

    img = cv2.resize(img, (224, 224))

    data.append(img)
    labels.append(1)

# ==============================
# NUMPY → TENSOR
# ==============================

data = np.array(data)
labels = np.array(labels)

print("Total images:", data.shape)
print("Total labels:", labels.shape)

# Normalize
data = data / 255.0

# Convert to tensor
data = torch.tensor(data, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.float32)

# Change shape (HWC → CHW)
data = data.permute(0, 3, 1, 2)

print("Tensor data shape:", data.shape)
print("Tensor labels shape:", labels.shape)

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

print("Train shape:", X_train.shape)
print("Validation shape:", X_val.shape)
# ==============================
# MODEL (ResNet + Classifier)
# ==============================

import torch.nn as nn
from torchvision import models

resnet = models.resnet18(pretrained=True)
resnet = nn.Sequential(*list(resnet.children())[:-1])

class DeepFakeModel(nn.Module):
    def __init__(self):
        super(DeepFakeModel, self).__init__()

        # Feature extractor
        self.feature_extractor = resnet

        # Freeze all layers
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # Unfreeze last layer
        for param in list(self.feature_extractor.children())[-1].parameters():
            param.requires_grad = True

        # Dropout + Classifier
       
        self.fc = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)

        
        x = self.fc(x)
        x = self.sigmoid(x)

        return x


model = DeepFakeModel()

print("Model ready")
# ==============================
# DATA LOADERS (CORRECT SPLIT)
# ==============================

from torch.utils.data import TensorDataset, DataLoader

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# ==============================
# LOSS + OPTIMIZER
# ==============================

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ==============================
# TRAINING LOOP (CORRECT)
# ==============================

epochs = 15

print(" Training started...")

for epoch in range(epochs):

    total_loss = 0

    for batch_data, batch_labels in train_loader:

        batch_labels = batch_labels.unsqueeze(1)

        optimizer.zero_grad()

        outputs = model(batch_data)

        loss = criterion(outputs, batch_labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {total_loss:.4f}")
# ==============================
# VALIDATION (REAL ACCURACY)
# ==============================
correct = 0
total = 0

with torch.no_grad():

    for batch_data, batch_labels in val_loader:

        batch_labels = batch_labels.unsqueeze(1)

        outputs = model(batch_data)

        predicted = (outputs > 0.5).float()

        correct += (predicted == batch_labels).sum().item()
        total += batch_labels.size(0)

accuracy = (correct / total) * 100

print(f"Validation Accuracy: {accuracy:.2f}%")

torch.save(model.state_dict(), "model.pth")
print(" Model saved")