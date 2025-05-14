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

weather = pd.read_csv("split_master_EastHsinchu.csv")
weather = weather[["Month", "Date", "Hour", "AirTemperature", "Precipitation", "RelativeHumidity", "StationPressure", "WindSpeed", "WindDirection"]]

# weather = weather.rename(columns={'temp': 'temperature', 'datetime': 'date'})
# weather['date'] = pd.to_datetime(weather['date'])
# weather.set_index('date')
print(weather.head(5))

print(weather.info)
