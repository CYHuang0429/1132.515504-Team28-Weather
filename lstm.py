
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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
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

weather.dropna(inplace=True)
weather["Precipitation"] = np.log1p(weather["Precipitation"])  # log1p轉換
weather["RainBinary"] = (weather["Precipitation"] > 0).astype(int)  # 二元化降雨量

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
featureCols = ["AirTemperature", "DewPointTemperature", "Precipitation", "PrecipitationDuration", "RelativeHumidity", "SeaLevelPressure", "StationPressure", "WindSpeed", "WindDirection"]

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
        self.lstm = nn.LSTM(inputSize, hiddenSize, num_layers=2, batch_first=True, dropout=0.2)
        self.linear = nn.Linear(hiddenSize, 1)
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
        self.lstm = nn.LSTM(inputSize, hiddenSize, num_layers=2, batch_first=True,dropout=0.2)
        self.linear = nn.Linear(hiddenSize, outputSize)

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
        torch.save(model.state_dict(), "best_model.pt")  # 儲存最佳模型
        print(f"Saved best model at epoch {epoch+1}")
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping")
            break

classification_model.load_state_dict(torch.load("best_classifier.pt"))
classification_model.eval()

with torch.no_grad():
    Xtest_tensor = torch.tensor(Xtest, dtype=torch.float32).to(device)
    rain_probs = classification_model(Xtest_tensor).cpu().numpy().squeeze()
    rain_preds = (rain_probs >= 0.5).astype(int)
# test
rain_indices = np.where(rain_preds == 1)[0]

X_rain = Xtest[rain_indices]
Y_rain_true = Ytest[rain_indices]

# 測試模型載入
model.load_state_dict(torch.load("best_model.pt"))
model.eval()
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


plt.figure(figsize=(12, 4))
plt.plot(rain_probs[:200], label="Predicted Probability")
plt.plot(Yctest[:200], label="Actual RainBinary")
plt.legend()
plt.title("Rain Prediction vs Actual")
plt.grid(True)
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
 模型強化：加入 Dropout / 多層 LSTM / Early Stopping
如果你覺得模型表現還能提升，可以：

加入 dropout=0.2

設 num_layers=2 的多層 LSTM

用 ReduceLROnPlateau 或 early stopping 減少過擬合


'''



print(weather.info)
