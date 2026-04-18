import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

# ==============================
# PATH
# ==============================

train_path = "../dataset/Train"

# ==============================
# TRANSFORM
# ==============================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

# ==============================
# DATASET
# ==============================

full_dataset = ImageFolder(train_path, transform=transform)

print("Total images:", len(full_dataset))
print("Classes:", full_dataset.classes)

# 🔥 LIMIT DATA (VERY IMPORTANT)
import random
random.seed(42)

subset_size = 8000
subset_indices = random.sample(range(len(full_dataset)), subset_size)

dataset = Subset(full_dataset, subset_indices)

# ==============================
# DATALOADER
# ==============================

loader = DataLoader(dataset, batch_size=16, shuffle=True)

# ==============================
# MODEL
# ==============================

model = models.resnet18(pretrained=True)

model.fc = nn.Sequential(
    nn.Linear(512, 1),
    nn.Sigmoid()
)

# ==============================
# TRAINING SETUP
# ==============================

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# ==============================
# TRAINING LOOP
# ==============================

epochs = 5

for epoch in range(epochs):
    total_loss = 0

    for i, (images, labels) in enumerate(loader):

        # fix label shape
        labels = labels.float().unsqueeze(1)

        # progress print
        if i % 10 == 0:
            print(f"Epoch {epoch+1} | Batch {i}/{len(loader)}")

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f}")

# ==============================
# SAVE MODEL
# ==============================

torch.save(model.state_dict(), "../model.pth")
print("✅ Model saved")