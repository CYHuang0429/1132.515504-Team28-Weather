
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
from imblearn.over_sampling import RandomOverSampler

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
weather = weather.dropna().reset_index(drop=True)

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
    "Temp_DewDiff", "Delta_StationPressure", "HighHumidity",
    "Precipitation_t-1", "RelativeHumidity_t-1",
    "WindSpeed_t-1", "Temp_DewDiff_t-1", "RainBinary",
    "RainBinary_t-1", "IsRainingContinuously",
    "RH_roll3", "Precipitation_t-2", "Pressure_drop3h"
    
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

train_dataset = WeatherDataset(Xtrain, Ytrain)
val_dataset = WeatherDataset(Xval, Yval)
test_dataset = WeatherDataset(Xtest, Ytest)

train_dataset_c = WeatherDataset(Xtrain, Yctrain)
val_dataset_c = WeatherDataset(Xval, Ycval)
test_dataset_c = WeatherDataset(Xtest, Yctest)

train_dataset_multi = WeatherMultiOutputDataset(Xtrain, Ytrain, Yctrain)
val_dataset_multi = WeatherMultiOutputDataset(Xval, Yval, Ycval)
test_dataset_multi = WeatherMultiOutputDataset(Xtest, Ytest, Yctest)

train_loader_multi = DataLoader(train_dataset_multi, batch_size=batch_size, shuffle=True)
val_loader_multi = DataLoader(val_dataset_multi, batch_size=batch_size, shuffle=False)
test_loader_multi = DataLoader(test_dataset_multi, batch_size=batch_size, shuffle=False)


train_dataset_reg = WeatherDataset(Xtrain, Ytrain)
val_dataset_reg = WeatherDataset(Xval, Yval)

train_loader = DataLoader(train_dataset_c, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset_c, batch_size=batch_size, shuffle=False)
train_loader_reg = DataLoader(train_dataset_reg, batch_size=batch_size, shuffle=False)
val_loader_reg = DataLoader(val_dataset_reg, batch_size=batch_size, shuffle=False)

for Xbatch, Ybatch in train_loader:
    print("X_batch shape:",Xbatch.shape)
    print("Y_batch shape:",Ybatch.shape)
    break

class LSTMClassifier(nn.Module):
    def __init__(self, inputSize, hiddenSize):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(inputSize, hiddenSize, num_layers=3, batch_first=True, dropout=0.3, bidirectional=True)
        self.attn = nn.MultiheadAttention(embed_dim=hiddenSize*2, num_heads=2, batch_first=True)

        self.linear = nn.Linear(hiddenSize*2, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        lstmOut, _ = self.lstm(x)
        # out = self.linear(lstmOut[:, -1, :])
        attnOut, _ = self.attn(lstmOut, lstmOut, lstmOut)
        out = self.linear(attnOut[:, -1, :])           # 用最後一個時間步
        #out = self.sigmoid(out)
        return out
    




# LSTM
class LSTM(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(inputSize, hiddenSize, num_layers=3, batch_first=True,dropout=0.3, bidirectional=True)
        self.linear = nn.Linear(hiddenSize*2, outputSize)

    def forward(self, x):
        lstmOut, _ = self.lstm(x)
        out = self.linear(lstmOut[:, -1, :])
        return out
    
class MultiTaskLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size_reg):
        super(MultiTaskLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=3, batch_first=True, dropout=0.3, bidirectional=True)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size*2, num_heads=2, batch_first=True)
        self.fc_reg = nn.Linear(hidden_size*2, output_size_reg)  # 回歸分支
        self.fc_cls = nn.Linear(hidden_size*2, 1)                # 分類分支

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)
        out = attn_out[:, -1, :]
        reg_out = self.fc_reg(out)
        cls_out = self.fc_cls(out)
        return reg_out, cls_out


inputSize = len(featureCols)
hiddenSize = 64
numLayers = 2
outputSize = len(target)

#model = LSTM(inputSize, hiddenSize, outputSize).to(device)
model = MultiTaskLSTM(inputSize, hiddenSize, outputSize).to(device)
#損失函數
criterion = nn.MSELoss()

#優化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

classification_model = LSTMClassifier(inputSize, hiddenSize).to(device)
# criterion_c = nn.BCELoss()
pos_weight = torch.tensor([8.0], device=device)  # 加強 minority class (Rain) 權重
criterion_c = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer_c = torch.optim.Adam(classification_model.parameters(), lr=0.001)

#訓練
numEpochs = 100
bestLoss = float('inf')
patience = 5
wait=0
trainLosses = []
validLosses = []

for epoch in range(numEpochs):
    classification_model.train()
    trainLoss = 0.0
    for Xbatch, Ybatch in train_loader:
        Xbatch = Xbatch.to(device)
        Ybatch = Ybatch.float().to(device)

        optimizer_c.zero_grad()
        Ypred = classification_model(Xbatch)
        loss = criterion_c(Ypred, Ybatch)
        loss.backward()
        optimizer_c.step()
        

        trainLoss += loss.item()*Xbatch.size(0)

    trainLoss /= len(train_loader.dataset)

    print(f"Epoch [{epoch + 1}/{numEpochs}], Train Loss: {trainLoss:.4f}")
    trainLosses.append(trainLoss)

    # validation
    classification_model.eval()
    valLoss = 0.0
    with torch.no_grad():
        for Xbatch, Ybatch in val_loader:
            Xbatch = Xbatch.to(device)
            Ybatch = Ybatch.float().to(device)

            Ypred = classification_model(Xbatch)
            loss = criterion(Ypred, Ybatch)
            valLoss += loss.item() * Xbatch.size(0)

    valLoss /= len(val_loader.dataset)
    validLosses.append(valLoss)
    print(f"Validation Loss: {valLoss:.4f}")
    # Early Stopping
    if valLoss < bestLoss:
        bestLoss = valLoss
        wait = 0
        torch.save(classification_model.state_dict(), "./best_classifier.pt")  # 儲存最佳模型
        print(f"Saved best model at epoch {epoch+1}")
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping")
            break


# ========== Training Regression Model ==========
numEpochs = 100
bestLoss_reg = float('inf')
wait = 0
patience = 5
trainLosses_reg = []
validLosses_reg = []

for epoch in range(numEpochs):
    model.train()
    trainLoss = 0.0
    for Xbatch, Ybatch in train_loader_reg:
        Xbatch = Xbatch.to(device)
        Ybatch = Ybatch.float().to(device)

        optimizer.zero_grad()
        Ypred,_ = model(Xbatch)
        loss = criterion(Ypred, Ybatch)
        loss.backward()
        optimizer.step()

        trainLoss += loss.item() * Xbatch.size(0)

    trainLoss /= len(train_loader_reg.dataset)
    trainLosses_reg.append(trainLoss)

    # Validation
    model.eval()
    valLoss = 0.0
    with torch.no_grad():
        for Xbatch, Ybatch in val_loader_reg:
            Xbatch = Xbatch.to(device)
            Ybatch = Ybatch.float().to(device)

            Ypred,_ = model(Xbatch)
            loss = criterion(Ypred, Ybatch)
            valLoss += loss.item() * Xbatch.size(0)

    valLoss /= len(val_loader_reg.dataset)
    validLosses_reg.append(valLoss)

    print(f"[Epoch {epoch+1}] Regressor Train Loss: {trainLoss:.4f}, Val Loss: {valLoss:.4f}")
    
    if valLoss < bestLoss_reg:
        bestLoss_reg = valLoss
        wait = 0
        torch.save(model.state_dict(), "./best_model.pt")
        print(f"✅ Saved best regression model at epoch {epoch+1}")
    else:
        wait += 1
        if wait >= patience:
            print("🛑 Early stopping regression training")
            break

'''
classification_model.load_state_dict(torch.load("best_classifier.pt"))
classification_model.eval()
'''

with torch.no_grad():
    Xtest_tensor = torch.tensor(Xtest, dtype=torch.float32).to(device)
    logits = classification_model(Xtest_tensor)
    rain_probs = torch.sigmoid(logits).cpu().numpy().squeeze()
    rain_preds = (rain_probs >= 0.3).astype(int)

# 計算 precision-recall curve
precisions, recalls, thresholds = precision_recall_curve(Yctest.flatten(), rain_probs)

# 計算 F1 分數
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

# test
rain_indices = np.where(rain_preds == 1)[0]

X_rain = Xtest[rain_indices]
Y_rain_true = Ytest[rain_indices]

# 測試模型載入
classification_model.load_state_dict(torch.load("best_classifier.pt"))
classification_model.eval()
with torch.no_grad():
    X_rain_tensor = torch.tensor(X_rain, dtype=torch.float32).to(device)
    #Y_rain_pred_scaled = model(X_rain_tensor).cpu().numpy()
    Y_rain_pred_scaled, _ = model(X_rain_tensor)
    Y_rain_pred_scaled = Y_rain_pred_scaled.cpu().numpy()


# 反標準化 + 還原 log1p
precip_idx = target.index("Precipitation")

Y_rain_true_real = targetScaler.inverse_transform(Y_rain_true)
Y_rain_pred_real = targetScaler.inverse_transform(Y_rain_pred_scaled)

Y_rain_true_real[:, precip_idx] = np.expm1(Y_rain_true_real[:, precip_idx])
Y_rain_pred_real[:, precip_idx] = np.expm1(Y_rain_pred_real[:, precip_idx])

# 評估回歸效果
print("\n=== Multi-Target Evaluation on Rainy Hours ===")
for i, var in enumerate(target):
    mae = mean_absolute_error(Y_rain_true_real[:, i], Y_rain_pred_real[:, i])
    rmse = np.sqrt(mean_squared_error(Y_rain_true_real[:, i], Y_rain_pred_real[:, i]))
    r2 = r2_score(Y_rain_true_real[:, i], Y_rain_pred_real[:, i])
    print(f"{var:<15} → MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")

# 畫圖：預測 vs 真實的降雨量（只針對預測為下雨的樣本）
plt.figure(figsize=(10, 5))
precip_idx = target.index("Precipitation")
plt.plot(Y_rain_true_real[:, precip_idx], label='True Rainfall', marker='o')
plt.plot(Y_rain_pred_real[:, precip_idx], label='Predicted Rainfall', marker='x')
plt.xlabel("Sample Index")
plt.ylabel("Precipitation (mm)")
plt.title("Predicted vs Actual Rainfall on Rainy Hours")
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
    plt.plot(Y_rain_true_real[:, i], label=f'True {var}', marker='o')
    plt.plot(Y_rain_pred_real[:, i], label=f'Predicted {var}', marker='x')
    plt.xlabel("Sample Index")
    plt.ylabel(var)
    plt.title(f"{var} Prediction on Rainy Hours")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#print(weather.info)
