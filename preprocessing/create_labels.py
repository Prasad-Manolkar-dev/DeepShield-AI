import numpy as np

features = np.load("../dataset/features.npy")

num_samples = features.shape[0]

labels = np.array([0]*(num_samples//2) + [1]*(num_samples//2))

np.save("../dataset/labels.npy", labels)

print("✅ Labels created:", labels.shape)