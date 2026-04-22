import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
import random

# ==============================
# SETTINGS
# ==============================

train_path = "dataset/Train"   # 🔥 FULL FRAMES (IMPORTANT)
subset_size = 10000
epochs = 8
random.seed(42)

# ==============================
# TRANSFORM
# ==============================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomGrayscale(p=0.2),  # 🔥 helps detect fake tricks
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ==============================
# DATASET
# ==============================

full_dataset = ImageFolder(train_path, transform=transform)

print("Total images:", len(full_dataset))
print("Classes:", full_dataset.classes)

# Balanced sampling
real_indices = []
fake_indices = []

for idx, (path, label) in enumerate(full_dataset.samples):
    if label == 0:
        fake_indices.append(idx)
    else:
        real_indices.append(idx)

min_class_size = min(len(fake_indices), len(real_indices))
samples_per_class = min(subset_size // 2, min_class_size)

balanced_indices = random.sample(fake_indices, samples_per_class) + \
                   random.sample(real_indices, samples_per_class)

random.shuffle(balanced_indices)

dataset = Subset(full_dataset, balanced_indices)

loader = DataLoader(dataset, batch_size=16, shuffle=True)

# ==============================
# MODEL
# ==============================

model = models.resnet18(pretrained=True)

# Unfreeze last layers
for param in model.parameters():
    param.requires_grad = False

for param in model.layer4.parameters():
    param.requires_grad = True

model.fc = nn.Linear(512, 1)  # ❌ NO SIGMOID

# ==============================
# TRAINING SETUP
# ==============================

criterion = nn.BCEWithLogitsLoss()  # 🔥 KEY CHANGE
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# ==============================
# TRAINING
# ==============================

from tqdm import tqdm

for epoch in range(epochs):
    total_loss = 0

    print(f"\n🚀 Epoch {epoch+1}/{epochs}")

    for images, labels in tqdm(loader):

        labels = labels.float().unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"✅ Avg Loss: {total_loss / len(loader):.4f}")
# ==============================
# SAVE
# ==============================

torch.save(model.state_dict(), "model.pth")
print("✅ Model saved")