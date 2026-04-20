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

train_path = "dataset/faces"
subset_size = 8000
epochs = 10
random.seed(42)

# ==============================
# TRANSFORM (with normalization)
# ==============================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
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

# Get indices per class (FASTER version)
real_indices = []
fake_indices = []

for idx, (path, label) in enumerate(full_dataset.samples):
    if label == 0:
        fake_indices.append(idx)
    else:
        real_indices.append(idx)

print("Fake count:", len(fake_indices))
print("Real count:", len(real_indices))

# Find smallest class
min_class_size = min(len(fake_indices), len(real_indices))

# Safe sampling size
samples_per_class = min(subset_size // 2, min_class_size)

print("Using per class:", samples_per_class)

# Balanced sample
balanced_indices = random.sample(fake_indices, samples_per_class) + \
                   random.sample(real_indices, samples_per_class)

random.shuffle(balanced_indices)

dataset = Subset(full_dataset, balanced_indices)
print("Subset size:", len(dataset))

loader = DataLoader(dataset, batch_size=16, shuffle=True)

# ==============================
# MODEL
# ==============================

model = models.resnet18(pretrained=True)

# Freeze ALL layers first
for param in model.parameters():
    param.requires_grad = False

#  Unfreeze LAST block (important)
for param in model.layer4.parameters():
    param.requires_grad = True

# Replace classifier
model.fc = nn.Sequential(
    nn.Linear(512, 1),
    nn.Sigmoid()
)

# Train classifier
for param in model.fc.parameters():
    param.requires_grad = True
# ==============================
# TRAINING SETUP
# ==============================

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
# ==============================
# TRAINING LOOP
# ==============================

for epoch in range(epochs):
    total_loss = 0

    for i, (images, labels) in enumerate(loader):

        labels = labels.float().unsqueeze(1)

        if i % 10 == 0:
            print(f"Epoch {epoch+1} | Batch {i}/{len(loader)}")

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f}")

# ==============================
# SAVE MODEL
# ==============================

torch.save(model.state_dict(), "../model.pth")
print("✅ Model saved")