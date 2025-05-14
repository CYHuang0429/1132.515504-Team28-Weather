import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using \"{device}\" to train the model.")

weather = pd.read_csv("Masters/split_master_EastHsinchu.csv")
weather = weather[["Month", "Date", "Hour", "AirTemperature", "Precipitation", "RelativeHumidity", "StationPressure", "WindSpeed", "WindDirection"]]

weather = weather.rename(columns={'temp': 'AirTemperature', 'datetime': 'Date'})
weather['Date'] = pd.to_datetime(weather['Date'])
weather.set_index('Date')
# print(weather.head(5))

weather.dropna(inplace=True)
# weather['month'] = weather['date'].dt.month
# weather_naive = weather[['date', 'temperature']].copy(deep=True)
# weather_naive['prev_temperature'] = weather_naive['temperature'].shift(1)
# weather_naive.drop([0], inplace=True)
# weather_naive['difference'] = weather_naive['temperature'] - weather_naive['prev_temperature']
# weather_naive['square_error'] = weather_naive['difference'] ** 2
# weather_naive.head(2)

# square_error = weather_naive['square_error'].mean()
print(weather.info)
