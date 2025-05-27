
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report, confusion_matrix, precision_recall_curve

warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using \"{device}\" to train the model.")

weather = pd.read_csv("Masters/Master_Hsinchu.csv")
# weather = weather[["Month", "Date", "Hour", "AirTemperature", "Precipitation", "RelativeHumidity", "StationPressure", "WindSpeed", "WindDirection"]]
weather = weather[["Month", "Date", "Hour", "AirTemperature", "DewPointTemperature", "Precipitation", "PrecipitationDuration", "RelativeHumidity", "SeaLevelPressure", "StationPressure", "WindSpeed", "WindDirection"]]

#weather = weather.rename(columns={'temp': 'AirTemperature', 'datetime': 'Date'})
weather['Date'] = pd.to_datetime(weather['Date'])
weather.set_index('Date', inplace=True)
# print(weather.head(5))
weather[["AirTemperature", "DewPointTemperature", "Precipitation", "PrecipitationDuration", "RelativeHumidity", "SeaLevelPressure", "StationPressure", "WindSpeed", "WindDirection"]] = weather[["AirTemperature", "DewPointTemperature", "Precipitation", "PrecipitationDuration", "RelativeHumidity", "SeaLevelPressure", "StationPressure", "WindSpeed", "WindDirection"]].apply(pd.to_numeric, errors='coerce')

weather["Precipitation"] = np.log1p(weather["Precipitation"])  # log1p轉換
weather["RainBinary"] = (weather["Precipitation"] > 0).astype(int)  # 二元化降雨量
# 基本特徵
weather["Temp_DewDiff"] = weather["AirTemperature"] - weather["DewPointTemperature"]
weather["Delta_StationPressure"] = weather["StationPressure"] - weather["StationPressure"].shift(1)
# weather["HighHumidity"] = (weather["RelativeHumidity"] >= 90).astype(int)
# weather["RainBinary_t-1"] = weather["RainBinary"].shift(1)
# weather["IsRainingContinuously"] = ((weather["RainBinary"] == 1) & (weather["RainBinary_t-1"] == 1)).astype(int)
# weather["RH_roll3"] = weather["RelativeHumidity"].rolling(3).mean()
# weather["Precipitation_t-2"] = weather["Precipitation"].shift(2)
# weather["Pressure_drop3h"] = weather["StationPressure"] - weather["StationPressure"].shift(3)

# 延遲特徵（可根據需求加更多 lag）
for col in ["Precipitation", "RelativeHumidity", "WindSpeed", "Temp_DewDiff"]:
    weather[f"{col}_t-1"] = weather[col].shift(1)

# 處理缺失值
weather = weather.dropna()

# 計算每個特徵的MSE
target = ["AirTemperature", "Precipitation", "WindSpeed"]
#target = ["Precipitation"]
binarrTarget = ["RainBinary"]
resultList = []
for col in target:
    temp_df = weather[[col]].copy(deep=True)
    temp_df['prev'] = temp_df[col].shift(1)
    temp_df.dropna(inplace=True)
    temp_df[col]=pd.to_numeric(temp_df[col], errors='coerce')
    temp_df['prev'] = pd.to_numeric(temp_df['prev'], errors='coerce')

    temp_df['difference'] = temp_df[col] - temp_df['prev']
    temp_df['square_error'] = temp_df['difference'] ** 2
    mse = temp_df['square_error'].mean()
    resultList.append({'Feature': col, 'NativeMSE': mse})

result_df = pd.DataFrame(resultList)
print(result_df)

# 標準化
featureCols = [
    "AirTemperature", "DewPointTemperature", "Precipitation", "PrecipitationDuration",
    "RelativeHumidity", "SeaLevelPressure", "StationPressure", "WindSpeed", "WindDirection",
    "Temp_DewDiff", "Delta_StationPressure", 
    "Precipitation_t-1", "RelativeHumidity_t-1",
    "WindSpeed_t-1", "Temp_DewDiff_t-1", 
    
    
]

featureScaler = StandardScaler()
featureScaled = featureScaler.fit_transform(weather[featureCols])
featureScaled_df = pd.DataFrame(featureScaled, columns=featureCols, index=weather.index)

targetScaler = StandardScaler()
targetScaled = targetScaler.fit_transform(weather[target])
targetScaled_df = pd.DataFrame(targetScaled, columns=target, index=weather.index)

# Sliding Window 保留時間序列
windowSize = 24
Xall = featureScaled_df.values
Yall = targetScaled_df.values
Ybin = weather[binarrTarget].values
X = []
Y = []
Yc = []
for i in range(len(Xall) - windowSize):
    Xwindow = Xall[i:i + windowSize]
    Ywindow = Yall[i + windowSize]
    Ycwindow = Ybin[i + windowSize]
    X.append(Xwindow)
    Y.append(Ywindow)
    Yc.append(Ycwindow)
X = np.array(X)
Y = np.array(Y)
Yc = np.array(Yc)
# 自定義Dataset
class WeatherDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        

    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
class WeatherMultiOutputDataset(Dataset):
    def __init__(self, X, Y_reg, Y_cls):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y_reg = torch.tensor(Y_reg, dtype=torch.float32)  # 回歸目標
        self.Y_cls = torch.tensor(Y_cls, dtype=torch.float32)  # 分類目標（二值）

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y_reg[idx], self.Y_cls[idx]

# train_test_split
XtrainFull, Xtest, YtrainFull, Ytest, YctrainFull, Yctest = train_test_split(X, Y, Yc,test_size=0.1, shuffle=False)
Xtrain, Xval, Ytrain, Yval, Yctrain, Ycval = train_test_split(XtrainFull, YtrainFull, YctrainFull, test_size=0.1111, shuffle=False)

batch_size = 64
'''
train_dataset = WeatherDataset(Xtrain, Ytrain)
val_dataset = WeatherDataset(Xval, Yval)
test_dataset = WeatherDataset(Xtest, Ytest)

train_dataset_c = WeatherDataset(Xtrain, Yctrain)
val_dataset_c = WeatherDataset(Xval, Ycval)
test_dataset_c = WeatherDataset(Xtest, Yctest)
'''
train_dataset_multi = WeatherMultiOutputDataset(Xtrain, Ytrain, Yctrain)
val_dataset_multi = WeatherMultiOutputDataset(Xval, Yval, Ycval)
test_dataset_multi = WeatherMultiOutputDataset(Xtest, Ytest, Yctest)

train_loader = DataLoader(train_dataset_multi, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset_multi, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset_multi, batch_size=batch_size, shuffle=False)
train_dataset_reg = WeatherDataset(Xtrain, Ytrain)
val_dataset_reg   = WeatherDataset(Xval,   Yval)
train_loader_reg  = DataLoader(train_dataset_reg, batch_size=batch_size, shuffle=True)
val_loader_reg    = DataLoader(val_dataset_reg,   batch_size=batch_size, shuffle=False)


for Xbatch, Yreg_batch, Ycls_batch in train_loader:
    print("X_batch shape:",    Xbatch.shape)
    print("Y_reg_batch shape:", Yreg_batch.shape)
    print("Y_cls_batch shape:", Ycls_batch.shape)
    break
    
class MultiTaskLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size_reg):
        super(MultiTaskLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=3, batch_first=True, dropout=0.3, bidirectional=True)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size*2, num_heads=2, batch_first=True)
        self.branch_dropout = nn.Dropout(0.3)
        self.branch_bn = nn.BatchNorm1d(hidden_size*2)
        # 回歸分支：兩層 FC + ReLU + Dropout
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, output_size_reg)
        )
        # 分類分支：兩層 FC + ReLU + Dropout
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 1)
        )


    def forward(self, x):
        # LSTM + Attention
        lstm_out, _ = self.lstm(x)  
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)
        # 取最後一個時間步作為特徵
        feat = attn_out[:, -1, :]   

        # 分支前正則化
        feat = self.branch_dropout(feat)
        feat = self.branch_bn(feat)

        # 各自輸出
        reg_out = self.reg_head(feat)
        cls_out = self.cls_head(feat)
        return reg_out, cls_out


inputSize = len(featureCols)
hiddenSize = 64
outputSize = len(target)
model = MultiTaskLSTM(inputSize, hiddenSize, outputSize).to(device)
 
 # 損失函數：回歸 + 分類
criterion_reg = nn.MSELoss()
pos_weight   = torch.tensor([8.0], device=device)
criterion_cls = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# 優化器：一次更新所有參數
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# （原本緊接在這裡的，應該就是你現在看到的兩段 for epoch in range(...)）
# ========================

numEpochs     = 100
best_val_loss_mt = float('inf')    # multi-task

wait_mt          = 0
patience_mt      = 5

for epoch in range(numEpochs):
    model.train()
    total_train_loss = 0.0
    for Xb, Yreg_b, Ycls_b in train_loader:
        Xb       = Xb.to(device)
        Yreg_b   = Yreg_b.to(device)
        Ycls_b   = Ycls_b.view(-1,1).to(device)

        optimizer.zero_grad()
        pred_reg, pred_cls = model(Xb)

        loss_reg = criterion_reg(pred_reg, Yreg_b)
        loss_cls = criterion_cls(pred_cls, Ycls_b)
        loss     = loss_reg + loss_cls

        loss.backward()
        optimizer.step()
        total_train_loss += loss.item() * Xb.size(0)

    avg_train = total_train_loss / len(train_loader.dataset)

    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for Xb, Yreg_b, Ycls_b in val_loader:
            Xb       = Xb.to(device)
            Yreg_b   = Yreg_b.to(device)
            Ycls_b   = Ycls_b.view(-1,1).to(device)

            pred_reg, pred_cls = model(Xb)
            loss_reg = criterion_reg(pred_reg, Yreg_b)
            loss_cls = criterion_cls(pred_cls, Ycls_b)
            total_val_loss += (loss_reg + loss_cls).item() * Xb.size(0)

    avg_val = total_val_loss / len(val_loader.dataset)

    print(f"Epoch {epoch+1}/{numEpochs} — train: {avg_train:.4f}, val: {avg_val:.4f}")

    if avg_val < best_val_loss_mt:
        best_val_loss_mt = avg_val
        wait_mt = 0
        torch.save(model.state_dict(), "best_multitask_model.pt")
        print("  ✅ saved best model")
    else:
        wait_mt += 1
        if wait_mt >= patience_mt:
            print("  🛑 early stopping")
            break
# 回歸專用訓練迴圈 (Regression-Only Training)
numEpochs = 100
best_val_loss_reg = float('inf')    # best validation loss for regression
wait_reg = 0
patience_reg = 5

for epoch in range(numEpochs):
    # 訓練階段
    model.train()
    train_loss_reg = 0.0
    for Xb, Yb in train_loader_reg:            # train_loader_reg: 只輸出 (X, Y_reg)
        Xb = Xb.to(device)
        Yb = Yb.to(device)

        optimizer.zero_grad()
        pred_reg, _ = model(Xb)                # 多任務模型回傳 (regression, classification)
        loss_reg = criterion_reg(pred_reg, Yb) # criterion_reg = nn.MSELoss()
        loss_reg.backward()
        optimizer.step()

        train_loss_reg += loss_reg.item() * Xb.size(0)

    train_loss_reg /= len(train_loader_reg.dataset)

    # 驗證階段
    model.eval()
    val_loss_reg = 0.0
    with torch.no_grad():
        for Xb, Yb in val_loader_reg:          # val_loader_reg: 只輸出 (X, Y_reg)
            Xb = Xb.to(device)
            Yb = Yb.to(device)

            pred_reg, _ = model(Xb)
            loss_reg = criterion_reg(pred_reg, Yb)
            val_loss_reg += loss_reg.item() * Xb.size(0)

    val_loss_reg /= len(val_loader_reg.dataset)

    print(f"[Epoch {epoch+1}] Reg Train Loss: {train_loss_reg:.4f}, Val Loss: {val_loss_reg:.4f}")

    # Early Stopping 檢查
    if val_loss_reg < best_val_loss_reg:
        best_val_loss_reg = val_loss_reg
        wait_reg = 0
        torch.save(model.state_dict(), "best_regression_model.pt")
        print("  ✅ Saved best regression model")
    else:
        wait_reg += 1
        if wait_reg >= patience_reg:
            print("  🛑 Early stopping regression training")
            break

model.eval()
val_probs = []
with torch.no_grad():
    for Xb, _, Ycls_b in val_loader:      # val_loader 對應驗證集
        Xb = Xb.to(device)
        _, cls_logits = model(Xb)
        val_probs.extend(torch.sigmoid(cls_logits).cpu().numpy().squeeze())

val_probs = np.array(val_probs)
Ycval_flat = Ycval.flatten()


precisions_val, recalls_val, thresholds_val = precision_recall_curve(Ycval_flat, val_probs)
thresholds_val = np.append(thresholds_val, 1.0)
f1_scores_val = 2 * (precisions_val * recalls_val) / (precisions_val + recalls_val + 1e-6)
best_idx_val = np.argmax(f1_scores_val)
best_threshold = thresholds_val[best_idx_val]

print(f"[Validation] Best Threshold: {best_threshold:.3f}  Precision: {precisions_val[best_idx_val]:.3f}  Recall: {recalls_val[best_idx_val]:.3f}  F1: {f1_scores_val[best_idx_val]:.3f}")






with torch.no_grad():
    Xtest_tensor = torch.tensor(Xtest, dtype=torch.float32).to(device)
    #logits = model(Xtest_tensor)
    #rain_probs = torch.sigmoid(logits).cpu().numpy().squeeze()
    _, cls_logits = model(Xtest_tensor)
    rain_probs = torch.sigmoid(cls_logits).cpu().numpy().squeeze()

rain_preds = (rain_probs >= best_threshold).astype(int)

print("=== Classification Report on Test Set ===")
print(classification_report(Yctest.flatten(), rain_preds, target_names=["No Rain", "Rain"]))
# 計算 precision-recall curve
precisions, recalls, thresholds = precision_recall_curve(Yctest.flatten(), rain_probs)

# 計算 F1 分數
thresholds = np.append(thresholds, 1.0)  
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

print(f"Best Threshold: {best_threshold:.2f}")
print(f"Precision at Best Threshold: {precisions[best_idx]:.2f}")
print(f"Recall at Best Threshold:    {recalls[best_idx]:.2f}")
print(f"F1 Score at Best Threshold:  {f1_scores[best_idx]:.2f}")

# 重新根據 best_threshold 產生預測
rain_preds = (rain_probs >= best_threshold).astype(int)

# 評估模型分類性能
print("=== Classification Report ===")
print(classification_report(Yctest.flatten(), rain_preds, target_names=["No Rain", "Rain"]))


# 測試模型載入
# # 測試模型載入
# model.load_state_dict(torch.load("best_classifier.pt"))
# model.eval()
# model.load_state_dict(torch.load("best_model.pt"))
# model.eval()
# 只載入多任務模型
model.load_state_dict(
    torch.load("best_regression_model.pt", map_location=device)
)
model.eval()
with torch.no_grad():
    Xtest_tensor = torch.tensor(Xtest, dtype=torch.float32).to(device)
    Y_pred_scaled, _ = model(Xtest_tensor)
    Y_pred_scaled = Y_pred_scaled.cpu().numpy()

precip_idx = target.index("Precipitation")
# 反標準化 + 還原 log1p
Y_test_real = targetScaler.inverse_transform(Ytest)
Y_pred_real = targetScaler.inverse_transform(Y_pred_scaled)

Y_test_real[:, precip_idx] = np.expm1(Y_test_real[:, precip_idx])
Y_pred_real[:, precip_idx] = np.expm1(Y_pred_real[:, precip_idx])

# 評估回歸效果
print("\n=== Multi-Target Evaluation on All Test Samples ===")
for i, var in enumerate(target):
    mae = mean_absolute_error(Y_test_real[:, i], Y_pred_real[:, i])
    rmse = np.sqrt(mean_squared_error(Y_test_real[:, i], Y_pred_real[:, i]))
    r2 = r2_score(Y_test_real[:, i], Y_pred_real[:, i])
    print(f"{var:<15} → MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")

# 畫圖：預測 vs 真實的降雨量（只針對預測為下雨的樣本）
plt.figure(figsize=(10, 5))
precip_idx = target.index("Precipitation")
plt.plot(Y_test_real[:, precip_idx], label='True Rainfall', marker='o')
plt.plot(Y_pred_real[:, precip_idx], label='Predicted Rainfall', marker='x')
plt.xlabel("Sample Index")
plt.ylabel("Precipitation (mm)")
plt.title("Predicted vs Actual Rainfall on All Test Samples")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 4))
plt.plot(rain_probs[:200], label="Predicted Probability")
plt.plot(Yctest[:200], label="Actual RainBinary")
plt.legend()
plt.title("Rain Prediction vs Actual")
plt.grid(True)
plt.tight_layout()
plt.show()

cm = confusion_matrix(Yctest.flatten(), rain_preds)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Rain", "Rain"], yticklabels=["No Rain", "Rain"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

for i, var in enumerate(target):
    plt.figure(figsize=(10, 4))
    plt.plot(Y_test_real[:, i], label=f'True {var}', marker='o')
    plt.plot(Y_pred_real[:, i], label=f'Predicted {var}', marker='x')
    plt.xlabel("Sample Index")
    plt.ylabel(var)
    plt.title(f"{var} Prediction on All Test Samples")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 整理並輸出含「預測是否下雨」欄位的結果
output_df = pd.DataFrame({
    "True_AirTemperature": Y_test_real[:, 0],
    "Pred_AirTemperature": Y_pred_real[:, 0],
    "True_Precipitation": Y_test_real[:, 1],
    "Pred_Precipitation": Y_pred_real[:, 1],
    "True_WindSpeed": Y_test_real[:, 2],
    "Pred_WindSpeed": Y_pred_real[:, 2],
    "True_RainBinary": Yctest.flatten(),
    "Pred_RainBinary": rain_preds,
})

output_df.index.name = "SampleIndex"
output_df.to_csv("RainyHour_Predictions.csv", index=True)
print("✅ 預測結果（含預測是否下雨）已儲存至 RainyHour_Predictions.csv")

#print(weather.info)
