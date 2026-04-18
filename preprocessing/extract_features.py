import os
import cv2
import torch
import numpy as np
from torchvision import models, transforms

real_path = "../dataset/Train/real"
fake_path = "../dataset/Train/fake"

resnet = models.resnet18(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

features = []
labels = []

def process_folder(path, label):
    files = os.listdir(path)[:1000]

    for file in files:
        img_path = os.path.join(path, file)

        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = transform(img).unsqueeze(0)

        with torch.no_grad():
            feature = resnet(img)
            feature = feature.view(-1)

        features.append(feature.numpy())
        labels.append(label)

print("Processing REAL...")
process_folder(real_path, 0)

print("Processing FAKE...")
process_folder(fake_path, 1)

features = np.array(features)
labels = np.array(labels)

np.save("../dataset/features.npy", features)
np.save("../dataset/labels.npy", labels)

print("✅ Features:", features.shape)
print("✅ Labels:", labels.shape)