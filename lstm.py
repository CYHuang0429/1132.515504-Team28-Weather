
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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error, classification_report, confusion_matrix
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
weather["HighHumidity"] = (weather["RelativeHumidity"] >= 90).astype(int)
weather["RainBinary_t-1"] = weather["RainBinary"].shift(1)
weather["IsRainingContinuously"] = ((weather["RainBinary"] == 1) & (weather["RainBinary_t-1"] == 1)).astype(int)
weather["RH_roll3"] = weather["RelativeHumidity"].rolling(3).mean()
weather["Precipitation_t-2"] = weather["Precipitation"].shift(2)
weather["Pressure_drop3h"] = weather["StationPressure"] - weather["StationPressure"].shift(3)

# 延遲特徵（可根據需求加更多 lag）
for col in ["Precipitation", "RelativeHumidity", "WindSpeed", "Temp_DewDiff"]:
    weather[f"{col}_t-1"] = weather[col].shift(1)

# 處理缺失值
weather = weather.dropna().reset_index(drop=True)

# 計算每個特徵的MSE
# target = ["AirTemperature", "Precipitation", "WindSpeed"]
target = ["Precipitation"]
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
        self.Y = torch.tensor(Y, dtype=torch.long)
        

    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
# train_test_split
XtrainFull, Xtest, YtrainFull, Ytest, YctrainFull, Yctest = train_test_split(X, Y, Yc,test_size=0.1, shuffle=False)
Xtrain, Xval, Ytrain, Yval, Yctrain, Ycval = train_test_split(XtrainFull, YtrainFull, YctrainFull, test_size=0.1111, shuffle=False)

train_dataset = WeatherDataset(Xtrain, Ytrain)
val_dataset = WeatherDataset(Xval, Yval)
test_dataset = WeatherDataset(Xtest, Ytest)

train_dataset_c = WeatherDataset(Xtrain, Yctrain)
val_dataset_c = WeatherDataset(Xval, Ycval)
test_dataset_c = WeatherDataset(Xtest, Yctest)

train_dataset_reg = WeatherDataset(Xtrain, Ytrain)
val_dataset_reg = WeatherDataset(Xval, Yval)

batch_size = 64
train_loader = DataLoader(train_dataset_c, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset_c, batch_size=batch_size, shuffle=False)
train_loader_reg = DataLoader(train_dataset_reg, batch_size=batch_size, shuffle=False)
val_loader_reg = DataLoader(train_dataset_reg, batch_size=batch_size, shuffle=False)

for Xbatch, Ybatch in train_loader:
    print("X_batch shape:",Xbatch.shape)
    print("Y_batch shape:",Ybatch.shape)
    break

class LSTMClassifier(nn.Module):
    def __init__(self, inputSize, hiddenSize):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(inputSize, hiddenSize, num_layers=3, batch_first=True, dropout=0.3, bidirectional=True)
        self.linear = nn.Linear(hiddenSize*2, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        lstmOut, _ = self.lstm(x)
        out = self.linear(lstmOut[:, -1, :])
        out = self.sigmoid(out)
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

inputSize = len(featureCols)
hiddenSize = 64
numLayers = 2
outputSize = len(target)

model = LSTM(inputSize, hiddenSize, outputSize).to(device)

#損失函數
criterion = nn.MSELoss()

#優化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

classification_model = LSTMClassifier(inputSize, hiddenSize).to(device)
criterion_c = nn.BCELoss()
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
        Ypred = model(Xbatch)
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

            Ypred = model(Xbatch)
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


classification_model.load_state_dict(torch.load("best_classifier.pt"))
classification_model.eval()

with torch.no_grad():
    Xtest_tensor = torch.tensor(Xtest, dtype=torch.float32).to(device)
    rain_probs = classification_model(Xtest_tensor).cpu().numpy().squeeze()
    rain_preds = (rain_probs >= 0.3).astype(int)

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
    Y_rain_pred_scaled = model(X_rain_tensor).cpu().numpy()

# 反標準化 + 還原 log1p
Y_rain_true_real = targetScaler.inverse_transform(Y_rain_true)
Y_rain_pred_real = targetScaler.inverse_transform(Y_rain_pred_scaled)

Y_rain_true_real[:, 0] = np.expm1(Y_rain_true_real[:, 0])
Y_rain_pred_real[:, 0] = np.expm1(Y_rain_pred_real[:, 0])

# 評估回歸效果

mae = mean_absolute_error(Y_rain_true_real[:, 0], Y_rain_pred_real[:, 0])
rmse = np.sqrt(mean_squared_error(Y_rain_true_real[:, 0], Y_rain_pred_real[:, 0]))
r2 = r2_score(Y_rain_true_real[:, 0], Y_rain_pred_real[:, 0])
print(f"Precipitation on Rainy Hours → MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")

# 畫圖：預測 vs 真實的降雨量（只針對預測為下雨的樣本）
plt.figure(figsize=(10, 5))
plt.plot(Y_rain_true_real[:, 0], label='True Rainfall', marker='o')
plt.plot(Y_rain_pred_real[:, 0], label='Predicted Rainfall', marker='x')
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

'''
model.eval()
testLoss = 0.0
with torch.no_grad():
    for Xbatch, Ybatch in test_loader:
        Xbatch = Xbatch.to(device)
        Ybatch = Ybatch.to(device)

        Ypred = model(Xbatch)
        loss = criterion(Ypred, Ybatch)

        testLoss += loss.item()*Xbatch.size(0)

    testLoss /= len(test_loader.dataset)
    print(f"Test Loss: {testLoss:.4f}")

# 反標準化
XtestTensor = torch.tensor(Xtest, dtype=torch.float32).to(device)
model.eval()
with torch.no_grad():
    YpredScaled = model(XtestTensor).cpu().numpy()  

YtrueScaled = Ytest 
YpredReal = targetScaler.inverse_transform(YpredScaled)
YtrueReal = targetScaler.inverse_transform(YtrueScaled)
# 對 Precipitation 還原 log1p 多目標的話要改
YpredReal[:, 0] = np.expm1(YpredReal[:, 0])
YtrueReal[:, 0] = np.expm1(YtrueReal[:, 0])


# for i, name in enumerate(["AirTemperature", "Precipitation", "WindSpeed"]):
for i, name in enumerate(["Precipitation"]):
    mae = mean_absolute_error(YtrueReal[:, i], YpredReal[:, i])
    rmse = root_mean_squared_error(YtrueReal[:, i], YpredReal[:, i])
    r2 = r2_score(YtrueReal[:, i], YpredReal[:, i])
    print(f"{name} → MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")


plt.figure(figsize=(10, 5))
plt.plot(trainLosses, label='Train Loss')
plt.plot(validLosses, label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train vs Validation Loss Over Epochs")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
'''


'''
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import RandomOverSampler

warnings.filterwarnings('ignore')

# 0. reproducibility
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using \"{device}\" to train the model.")

# ------------------------------------------------
# 1. Load & preprocess data
weather = pd.read_csv("Masters/Master_Hsinchu.csv")
cols = ["Month", "Date", "Hour", "AirTemperature", "DewPointTemperature",
        "Precipitation", "PrecipitationDuration", "RelativeHumidity",
        "SeaLevelPressure", "StationPressure", "WindSpeed", "WindDirection"]
weather = weather[cols]
weather['Date'] = pd.to_datetime(weather['Date'])
weather.set_index('Date', inplace=True)
weather = weather.apply(pd.to_numeric, errors='coerce')
weather.dropna(inplace=True)
weather["Precipitation"] = np.log1p(weather["Precipitation"])
weather["RainBinary"] = (weather["Precipitation"] > 0).astype(int)

featureCols = ["AirTemperature", "DewPointTemperature", "Precipitation",
               "PrecipitationDuration", "RelativeHumidity", "SeaLevelPressure",
               "StationPressure", "WindSpeed", "WindDirection"]
target = ["Precipitation"]
windowSize = 24

# ------------------------------------------------
# 2. Time-based split & scaling (fit only on train)
split_ratio = (0.8, 0.1, 0.1)  # train/val/test
n = len(weather)
train_end = int(n * split_ratio[0])
val_end = int(n * (split_ratio[0] + split_ratio[1]))

train_df = weather.iloc[:train_end]
val_df = weather.iloc[train_end:val_end]
test_df = weather.iloc[val_end:]

featureScaler = StandardScaler().fit(train_df[featureCols])
targetScaler = StandardScaler().fit(train_df[target])

def scale(df, scaler, cols):
    return pd.DataFrame(scaler.transform(df[cols]), columns=cols, index=df.index)

train_feat = scale(train_df, featureScaler, featureCols)
val_feat = scale(val_df, featureScaler, featureCols)
test_feat = scale(test_df, featureScaler, featureCols)

train_tgt = scale(train_df, targetScaler, target)
val_tgt = scale(val_df, targetScaler, target)
test_tgt = scale(test_df, targetScaler, target)

# ------------------------------------------------
# 3. Sliding window builder
def build_windows(Xdf, Ydf, Ybin, win):
    Xa, Ya, Yc = [], [], []
    for i in range(len(Xdf) - win):
        Xa.append(Xdf.iloc[i:i+win].values)
        Ya.append(Ydf.iloc[i+win].values)
        Yc.append(Ybin.iloc[i+win])
    return np.array(Xa), np.array(Ya), np.array(Yc)

Xtrain, Ytrain, Yctrain = build_windows(train_feat, train_tgt, train_df["RainBinary"], windowSize)
Xval, Yval, Ycval = build_windows(val_feat, val_tgt, val_df["RainBinary"], windowSize)
Xtest, Ytest, Yctest = build_windows(test_feat, test_tgt, test_df["RainBinary"], windowSize)

# ------------------------------------------------
# 4. Oversample minority class for classification
eros = RandomOverSampler(random_state=42)
X2d = Xtrain.reshape(Xtrain.shape[0], -1)
Xros, Ycros = eros.fit_resample(X2d, Yctrain)
Xtrain_ros = Xros.reshape(-1, windowSize, len(featureCols))
Yctrain_ros = Ycros

# ------------------------------------------------
# Dataset & DataLoader definitions
class WeatherDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.Y[idx]

batch_size = 64
train_loader = DataLoader(WeatherDataset(Xtrain_ros, Yctrain_ros), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(WeatherDataset(Xval, Ycval), batch_size=batch_size, shuffle=False)
train_loader_reg = DataLoader(WeatherDataset(Xtrain, Ytrain.squeeze()), batch_size=batch_size, shuffle=True)
val_loader_reg = DataLoader(WeatherDataset(Xval, Yval.squeeze()), batch_size=batch_size, shuffle=False)

# ------------------------------------------------
# 5. Define models
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=3,
                               batch_first=True, dropout=0.3,
                               bidirectional=True)
        self.head = nn.Linear(hidden_dim*2, 1)
    def forward(self, x):
        h, _ = self.encoder(x)
        return self.head(h[:, -1]).squeeze(1)

class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=3,
                               batch_first=True, dropout=0.3)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, out_dim)
        )
    def forward(self, x):
        h, _ = self.encoder(x)
        return self.head(h[:, -1]).squeeze(1)

hiddenSize = 64
cls_model = LSTMClassifier(len(featureCols), hiddenSize).to(device)
reg_model = LSTMRegressor(len(featureCols), hiddenSize, len(target)).to(device)

# Weighted BCE for classification
pos_weight = torch.tensor([(len(Yctrain_ros) - Yctrain_ros.sum()) / Yctrain_ros.sum()]).to(device)
criterion_c = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer_c = torch.optim.Adam(cls_model.parameters(), lr=0.001)

criterion_reg = nn.MSELoss()
optimizer_reg = torch.optim.Adam(reg_model.parameters(), lr=0.001)

# ------------------------------------------------
# 6. Training & evaluation helpers
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(xb)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb)
        loss = criterion(preds, yb)
        total_loss += loss.item() * len(xb)
    return total_loss / len(loader.dataset)

# ------------------------------------------------
# 7. Train classification model
numEpochs = 100
best_val = float('inf')
patience = 5
wait = 0
for epoch in range(numEpochs):
    tr_loss = train_epoch(cls_model, train_loader, criterion_c, optimizer_c)
    val_loss = eval_epoch(cls_model, val_loader, criterion_c)
    print(f"[CLS Epoch {epoch+1}] Train Loss: {tr_loss:.4f}, Val Loss: {val_loss:.4f}")
    if val_loss < best_val:
        best_val = val_loss
        wait = 0
        torch.save(cls_model.state_dict(), "best_classifier.pt")
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping classification")
            break

# ------------------------------------------------
# 8. Train regression model
best_val_r = float('inf')
wait = 0
for epoch in range(numEpochs):
    tr_loss = train_epoch(reg_model, train_loader_reg, criterion_reg, optimizer_reg)
    val_loss = eval_epoch(reg_model, val_loader_reg, criterion_reg)
    print(f"[REG Epoch {epoch+1}] Train Loss: {tr_loss:.4f}, Val Loss: {val_loss:.4f}")
    if val_loss < best_val_r:
        best_val_r = val_loss
        wait = 0
        torch.save(reg_model.state_dict(), "best_regressor.pt")
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping regression")
            break

# ------------------------------------------------
# 9. Test evaluation
cls_model.load_state_dict(torch.load("best_classifier.pt"))
reg_model.load_state_dict(torch.load("best_regressor.pt"))
cls_model.eval()
reg_model.eval()

# Classification metrics
Xtest_tensor = torch.tensor(Xtest, dtype=torch.float32).to(device)
with torch.no_grad():
    logits = cls_model(Xtest_tensor)
    probs = torch.sigmoid(logits).cpu().numpy()
preds = (probs >= 0.5).astype(int)
print("Precision:", precision_score(Yctest, preds))
print("Recall:   ", recall_score(Yctest, preds))
print("F1 Score: ", f1_score(Yctest, preds))

# Regression on rainy samples
rain_idx = np.where(preds == 1)[0]
X_rain = Xtest[rain_idx]
Y_rain_true = Ytest[rain_idx].squeeze()
X_rain_tensor = torch.tensor(X_rain, dtype=torch.float32).to(device)
with torch.no_grad():
    pred_sc = reg_model(X_rain_tensor).cpu().numpy()
# inverse transform and expm1
y_true = targetScaler.inverse_transform(Y_rain_true.reshape(-1,1)).squeeze()
y_pred = targetScaler.inverse_transform(pred_sc.reshape(-1,1)).squeeze()
y_true = np.expm1(y_true)
y_pred = np.expm1(y_pred)
print(f"Rainy MAE: {mean_absolute_error(y_true, y_pred):.2f}")
print(f"Rainy RMSE:{np.sqrt(mean_squared_error(y_true, y_pred)):.2f}")
print(f"Rainy R2:  {r2_score(y_true, y_pred):.4f}")

# Plot classification probabilities vs actual
plt.figure(figsize=(12,4))
plt.plot(probs[:200], label="Pred Prob")
plt.plot(Yctest[:200], label="Actual RainBinary")
plt.legend()
plt.title("Rain Prediction vs Actual")
plt.grid(True)
plt.tight_layout()
plt.show()

'''




#print(weather.info)
