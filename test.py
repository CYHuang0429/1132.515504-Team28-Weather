import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# === CONFIGURATION ===
CSV_PATH = "merged_date_EastHsinchu.csv"
TARGET_COL = "Precipitation"
SEQ_LENGTH = 24
BATCH_SIZE = 64
EPOCHS = 30
LR = 0.001
TEST_SPLIT = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 1. LOAD DATA ===
df = pd.read_csv(CSV_PATH)

# Ensure 'rainfall' is numeric and handle missing values
df = df.select_dtypes(include=[np.number]).dropna()

# === 2. NORMALIZATION ===
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# === 3. SEQUENCE GENERATION ===
def create_sequences(data, target_idx, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length, target_idx]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

target_idx = df.columns.get_loc(TARGET_COL)
X, y = create_sequences(scaled_data, target_idx, SEQ_LENGTH)

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

dataset = TensorDataset(X_tensor, y_tensor)
test_size = int(TEST_SPLIT * len(dataset))
train_size = len(dataset) - test_size
train_ds, test_ds = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

# === 4. DEFINE MODEL ===
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # use the last time step
        out = self.fc(out)
        return out

input_size = X.shape[2]
model = LSTMModel(input_size).to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# === 5. TRAIN MODEL ===
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        output = model(xb)
        loss = criterion(output, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss/len(train_loader):.4f}")

# === 6. EVALUATE ===
model.eval()
predictions, actuals = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(DEVICE)
        preds = model(xb).cpu().numpy()
        predictions.append(preds)
        actuals.append(yb.numpy())

predictions = np.vstack(predictions)
actuals = np.vstack(actuals)

# Inverse transform to original scale
target_scaler = MinMaxScaler()
target_scaler.min_, target_scaler.scale_ = scaler.min_[target_idx], scaler.scale_[target_idx]
pred_inv = target_scaler.inverse_transform(predictions)
act_inv = target_scaler.inverse_transform(actuals)

# === 7. PLOT RESULTS ===
plt.figure(figsize=(10, 4))
plt.plot(act_inv, label="Actual Rainfall")
plt.plot(pred_inv, label="Predicted Rainfall")
plt.legend()
plt.title("Rainfall Prediction (One Step Ahead)")
plt.xlabel("Sample")
plt.ylabel("Rainfall")
plt.tight_layout()
plt.show()
