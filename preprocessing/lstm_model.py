import numpy as np
import torch
import torch.nn as nn

# ==============================
# LOAD FEATURES
# ==============================

features = np.load(r"C:\Users\Prasad\Desktop\DeepShield-AI\dataset\features.npy")

features = np.expand_dims(features, axis=0)
features = torch.tensor(features, dtype=torch.float32)

# Label
label = torch.tensor([[1.0]])

print("Input shape:", features.shape)
print("Label:", label)

# ==============================
# MODEL
# ==============================

class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)

        last_output = lstm_out[:, -1, :]

        out = self.fc(last_output)
        out = self.sigmoid(out)

        return out

model = LSTMModel()

print("✅ Model created")

# ==============================
# LOSS
# ==============================

criterion = nn.BCELoss()

# ==============================
# FORWARD PASS
# ==============================

output = model(features)

print("Prediction:", output)

loss = criterion(output, label)

print("Loss:", loss.item())

# ==============================
# OPTIMIZER
# ==============================

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ==============================
# BACKPROPAGATION
# ==============================

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ==============================
# TRAINING LOOP
# ==============================

epochs = 10

for epoch in range(epochs):

    optimizer.zero_grad()

    output = model(features)

    loss = criterion(output, label)

    loss.backward()

    optimizer.step()

    print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")