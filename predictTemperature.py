import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from Model import Temperature_LSTM as TL

class WeatherDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # shape (N, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def dataset_construct(path, test_size_p=0.2, val_size_p=0.1):
    if not os.path.isfile(path):
        raise ValueError('Floder not exist.')
    
    cols = ["AirTemperature", "DewPointTemperature","RelativeHumidity", "SeaLevelPressure", "StationPressure"]
    drop_cols = ["Precipitation","PrecipitationDuration","WindSpeed","WindDirection"]
    
    weather = pd.read_csv(path)
    weather = weather.drop(columns=drop_cols)
    weather[cols] = weather[cols].apply(pd.to_numeric, errors='coerce')
    weather[cols] = weather[cols].fillna(weather[cols].mean())
    weather['Date'] = pd.to_datetime(weather['Date'], errors='coerce')
    weather['Month'] = weather['Date'].dt.month
    weather['Date'] = weather['Date'].dt.day

    print(weather.columns.values)
    print(weather)

    feature_cols = ["DewPointTemperature", "RelativeHumidity", "SeaLevelPressure", "StationPressure", "Month", "Date"]
    target_col = "AirTemperature"

    X = weather[feature_cols].values.astype('float32')
    y = weather[target_col].values.astype('float32')

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size_p + val_size_p, random_state=42)
    relative_val_size = val_size_p / (test_size_p + val_size_p)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1 - relative_val_size, random_state=42)

    train_dataset = WeatherDataset(X_train, y_train)
    val_dataset = WeatherDataset(X_val, y_val)
    test_dataset = WeatherDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader, test_loader


def train_model(train_loader: DataLoader, val_loader: DataLoader, device, num_epochs=10):
    model = TL().to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            print(X_batch.shape,y_batch.shape)
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)

        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")


def main():
    data_path = 'Masters/Master_Hsinchu.csv'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = dataset_construct(data_path)
    train_model(train_loader,val_loader,device)


if __name__ == '__main__':
    main()


'''
refernece:
    github: https://github.com/hritik7080/Weather-Prediction-Time-Series-Forecasting

'''