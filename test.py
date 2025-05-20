import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# 1) Settings & Device
window_size = 24
hidden_size = 64
batch_size = 64
epochs_cls = 20
epochs_reg = 50
lr = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} for training")

# 2) Load & Preprocess
df = pd.read_csv("Masters/Master_Hsinchu.csv", parse_dates=["Date"])
df.set_index("Date", inplace=True)

num_cols = [
    "AirTemperature","DewPointTemperature","Precipitation",
    "PrecipitationDuration","RelativeHumidity",
    "SeaLevelPressure","StationPressure","WindSpeed","WindDirection"
]
df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
df.dropna(inplace=True)

# Derived features
df["TempDewSpread"] = df["AirTemperature"] - df["DewPointTemperature"]
df["LCL"]           = df["TempDewSpread"] / 0.008
df["u_wind"]        = -df["WindSpeed"] * np.sin(np.radians(df["WindDirection"]))
df["v_wind"]        = -df["WindSpeed"] * np.cos(np.radians(df["WindDirection"]))
df["PressureDelta"] = df["SeaLevelPressure"].diff()
df.dropna(inplace=True)

# Raw & log-precip
prec_orig = df["Precipitation"].values
prec_log  = np.log1p(prec_orig)

# 3) Build sliding windows + labels
features = [
    "AirTemperature","DewPointTemperature","PrecipitationDuration",
    "RelativeHumidity","SeaLevelPressure","StationPressure",
    "WindSpeed","WindDirection","TempDewSpread","LCL",
    "u_wind","v_wind","PressureDelta"
]

X, y_cls, y_reg = [], [], []
for i in range(len(df) - window_size):
    seq = df[features].iloc[i : i + window_size].values
    X.append(seq)
    target_p = prec_orig[i + window_size]
    y_cls.append(1 if target_p > 0 else 0)
    y_reg.append(prec_log[i + window_size])

X      = np.array(X)         # (N, window_size, F)
y_cls  = np.array(y_cls)     # (N,)
y_reg  = np.array(y_reg)     # (N,)

# 4) Scaling
N, W, F = X.shape
X_flat = X.reshape(-1, F)
feat_scaler   = StandardScaler()
X_scaled_flat = feat_scaler.fit_transform(X_flat)
X_scaled      = X_scaled_flat.reshape(N, W, F)

reg_scaler = StandardScaler()
y_reg_scaled = reg_scaler.fit_transform(y_reg.reshape(-1, 1)).flatten()

# 5) Train/Test split (time‑series)
split = int(0.8 * N)
X_tr, X_te = X_scaled[:split], X_scaled[split:]
ycls_tr, ycls_te = y_cls[:split], y_cls[split:]
yreg_tr, yreg_te = y_reg_scaled[:split], y_reg_scaled[split:]

# 6) Dataset & DataLoader
class SeqDataset(Dataset):
    def __init__(self, X, yc, yr):
        self.X, self.yc, self.yr = map(
            lambda arr: torch.tensor(arr, dtype=torch.float32),
            (X, yc.reshape(-1,1), yr.reshape(-1,1))
        )
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.yc[i], self.yr[i]

train_ds = SeqDataset(X_tr, ycls_tr, yreg_tr)
test_ds  = SeqDataset(X_te, ycls_te, yreg_te)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

# 7) Model Definition
class LSTMPredictor(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.lstm = nn.LSTM(
            in_size, hid_size, num_layers=2,
            batch_first=True, dropout=0.2
        )
        self.fc = nn.Linear(hid_size, out_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

clf = LSTMPredictor(F, hidden_size, 1).to(device)
reg = LSTMPredictor(F, hidden_size, 1).to(device)

# 8) Losses & Optimizers
# Weighted BCE for classification
pos = (ycls_tr == 1).sum()
neg = (ycls_tr == 0).sum()
pos_weight = torch.tensor([neg/pos], device=device)
criterion_cls = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
opt_cls = torch.optim.Adam(clf.parameters(), lr=lr)

# Smooth L1 for regression
criterion_reg = nn.SmoothL1Loss()
opt_reg       = torch.optim.Adam(reg.parameters(), lr=lr)

# 9) Training Stage 1: Classifier
for ep in range(epochs_cls):
    clf.train()
    running_loss = 0.0
    for Xb, yb_c, _ in train_loader:
        Xb, yb_c = Xb.to(device), yb_c.to(device)
        opt_cls.zero_grad()
        logits = clf(Xb)
        loss = criterion_cls(logits, yb_c)
        loss.backward()
        opt_cls.step()
        running_loss += loss.item() * Xb.size(0)
    print(f"[CLS] Epoch {ep+1}/{epochs_cls}, Loss: {running_loss/len(train_ds):.4f}")

# 10) Training Stage 2: Regressor (rain-only)
mask = ycls_tr == 1
X_rain = X_tr[mask]; y_rain = yreg_tr[mask]
rain_ds = SeqDataset(X_rain, np.ones(len(y_rain)), y_rain)
rain_loader = DataLoader(rain_ds, batch_size=batch_size, shuffle=True)

for ep in range(epochs_reg):
    reg.train()
    running_loss = 0.0
    for Xb, _, yb_r in rain_loader:
        Xb, yb_r = Xb.to(device), yb_r.to(device)
        opt_reg.zero_grad()
        preds = reg(Xb)
        loss = criterion_reg(preds, yb_r)
        loss.backward()
        opt_reg.step()
        running_loss += loss.item() * Xb.size(0)
    print(f"[REG] Epoch {ep+1}/{epochs_reg}, Loss: {running_loss/len(rain_ds):.4f}")

# 11) Evaluation on Test Set
clf.eval(); reg.eval()
all_logits, all_true_c = [], []
all_preds_r, all_true_r = [], []

with torch.no_grad():
    for Xb, yb_c, yb_r in DataLoader(test_ds, batch_size=batch_size):
        Xb = Xb.to(device)
        lg = clf(Xb).cpu().numpy().flatten()
        all_logits.extend(lg)
        all_true_c.extend(yb_c.numpy().flatten())
        # Only regress where classifier predicts rain
        mask = lg > 0
        if mask.any():
            pr = reg(Xb).cpu().numpy().flatten()[mask]
            gt = yb_r.numpy().flatten()[mask]
            all_preds_r.extend(pr); all_true_r.extend(gt)

# Inverse‑scale & expm1
all_preds_r = reg_scaler.inverse_transform(
    np.array(all_preds_r).reshape(-1,1)
).flatten()
all_preds_r = np.expm1(all_preds_r)

all_true_r = reg_scaler.inverse_transform(
    np.array(all_true_r).reshape(-1,1)
).flatten()
all_true_r = np.expm1(all_true_r)

# Metrics
acc  = accuracy_score(all_true_c, np.array(all_logits)>0)
mae  = mean_absolute_error(all_true_r, all_preds_r)
rmse = mean_squared_error(all_true_r, all_preds_r, squared=False)
r2   = r2_score(all_true_r, all_preds_r)

print(f"\nTest Class Accuracy: {acc:.3f}")
print(f"Test Rain MAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}")
