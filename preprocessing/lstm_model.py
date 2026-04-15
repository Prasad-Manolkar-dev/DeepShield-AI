import numpy as np
import torch
import torch.nn as nn

# ==============================
# LOAD DATA
# ==============================

features = np.load(r"C:\Users\Prasad\Desktop\DeepShield-AI\dataset\features.npy")

features = np.expand_dims(features, axis=0)
features = torch.tensor(features, dtype=torch.float32)

print("Input shape:", features.shape)

# ==============================
# DEFINE LSTM MODEL
# ==============================

class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size=512,    # features per frame
            hidden_size=128,   # memory size
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(128, 1)  # output layer

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)

        # Take last timestep output
        last_output = lstm_out[:, -1, :]

        out = self.fc(last_output)
        out = self.sigmoid(out)

        return out

# ==============================
# CREATE MODEL
# ==============================

model = LSTMModel()

print("✅ LSTM model created")

# ==============================
# FORWARD PASS
# ==============================

output = model(features)

print("Model output:", output)
print("Output shape:", output.shape)