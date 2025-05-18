# import torch
# import torch.nn as nn
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.preprocessing import StandardScaler
# import warnings
# warnings.filterwarnings('ignore')

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using \"{device}\" to train the model.")

# weather = pd.read_csv("Masters/split_master_EastHsinchu.csv")
# weather = weather[["Month", "Date", "Hour", "AirTemperature", "Precipitation", "RelativeHumidity", "StationPressure", "WindSpeed", "WindDirection"]]

# weather = weather.rename(columns={'temp': 'AirTemperature', 'datetime': 'Date'})
# weather['Date'] = pd.to_datetime(weather['Date'])
# weather.set_index('Date')
# # print(weather.head(5))

# weather.dropna(inplace=True)
# # weather['month'] = weather['date'].dt.month
# # weather_naive = weather[['date', 'temperature']].copy(deep=True)
# # weather_naive['prev_temperature'] = weather_naive['temperature'].shift(1)
# # weather_naive.drop([0], inplace=True)
# # weather_naive['difference'] = weather_naive['temperature'] - weather_naive['prev_temperature']
# # weather_naive['square_error'] = weather_naive['difference'] ** 2
# # weather_naive.head(2)

# # square_error = weather_naive['square_error'].mean()
# print(weather.info)

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
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using \"{device}\" to train the model.")

weather = pd.read_csv("Masters/split_master_EastHsinchu.csv")
weather = weather[["Month", "Date", "Hour", "AirTemperature", "Precipitation", "RelativeHumidity", "StationPressure", "WindSpeed", "WindDirection"]]

#weather = weather.rename(columns={'temp': 'AirTemperature', 'datetime': 'Date'})
weather['Date'] = pd.to_datetime(weather['Date'])
weather.set_index('Date', inplace=True)
# print(weather.head(5))

weather.dropna(inplace=True)

# 計算每個特徵的MSE
target = ["AirTemperature", "Precipitation", "WindSpeed"]
resultList = []
for col in target:
    temp_df = weather[[col]].copy(deep=True)
    temp_df['prev'] = temp_df[col].shift(1)
    temp_df.dropna(inplace=True)
    temp_df['difference'] = temp_df[col] - temp_df['prev']
    temp_df['square_error'] = temp_df['difference'] ** 2
    mse = temp_df['square_error'].mean()
    resultList.append({'Feature': col, 'NativeMSE': mse})

result_df = pd.DataFrame(resultList)
print(result_df)

# 標準化
featureCols = ["AirTemperature", "Precipitation", "RelativeHumidity", "StationPressure", "WindSpeed", "WindDirection"]

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
X = []
Y = []
for i in range(len(Xall) - windowSize):
    Xwindow = Xall[i:i + windowSize]
    Ywindow = Yall[i + windowSize]
    X.append(Xwindow)
    Y.append(Ywindow)
X = np.array(X)
Y = np.array(Y)

# 自定義Dataset
class WeatherDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        

    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
# train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, shuffle=False)

train_dataset = WeatherDataset(Xtrain, Ytrain)
test_dataset = WeatherDataset(Xtest, Ytest)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

for Xbatch, Ybatch in train_loader:
    print("X_batch shape:",Xbatch.shape)
    print("Y_batch shape:",Ybatch.shape)
    break

# LSTM
class LSTM(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(inputSize, hiddenSize, batchFirst=True)
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

#訓練
numEpochs = 20
for epoch in range(numEpochs):
    model.train()
    trainLoss = 0.0
    for Xbatch, Ybatch in train_loader:
        Xbatch = Xbatch.to(device)
        Ybatch = Ybatch.to(device)

        optimizer.zero_grad()
        Ypred = model(Xbatch)
        loss = criterion(Ypred, Ybatch)
        loss.backward()
        optimizer.step()

        trainLoss += loss.item()*Xbatch.size(0)

    trainLoss /= len(train_loader.dataset)

    print(f"Epoch [{epoch + 1}/{numEpochs}], Train Loss: {trainLoss:.4f}")
# test
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




# weather['month'] = weather['date'].dt.month
# weather_naive = weather[['date', 'temperature']].copy(deep=True)
# weather_naive['prev_temperature'] = weather_naive['temperature'].shift(1)
# weather_naive.drop([0], inplace=True)
# weather_naive['difference'] = weather_naive['temperature'] - weather_naive['prev_temperature']
# weather_naive['square_error'] = weather_naive['difference'] ** 2
# weather_naive.head(2)

# square_error = weather_naive['square_error'].mean()
print(weather.info)
