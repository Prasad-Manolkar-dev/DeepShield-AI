import os

# dataset base path
dataset_path = "../dataset"

train_real = os.listdir(os.path.join(dataset_path, "train", "real"))
train_fake = os.listdir(os.path.join(dataset_path, "train", "fake"))

val_real = os.listdir(os.path.join(dataset_path, "validation", "real"))
val_fake = os.listdir(os.path.join(dataset_path, "validation", "fake"))

test_real = os.listdir(os.path.join(dataset_path, "test", "real"))
test_fake = os.listdir(os.path.join(dataset_path, "test", "fake"))

print("TRAIN DATA")
print("Real images:", len(train_real))
print("Fake images:", len(train_fake))

print("\nVALIDATION DATA")
print("Real images:", len(val_real))
print("Fake images:", len(val_fake))

print("\nTEST DATA")
print("Real images:", len(test_real))
print("Fake images:", len(test_fake))

#inspecting Image Size

from PIL import Image

sample_image_path = os.path.join(dataset_path, "train", "real", train_real[0])

img = Image.open(sample_image_path)

print("\nSample image size:", img.size)

#visualize images from the dataset.

import matplotlib.pyplot as plt

plt.imshow(img)
plt.title("Sample Real Image")
plt.axis("off")
plt.show()