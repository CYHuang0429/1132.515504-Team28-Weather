import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(inputSize, hiddenSize, num_layers=3, batch_first=True,dropout=0.3, bidirectional=True)
        self.linear = nn.Linear(hiddenSize*2, outputSize)

    def forward(self, x):
        lstmOut, _ = self.lstm(x)
        out = self.linear(lstmOut[:, -1, :])
        return out
    
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

class AddGaussianNoise(nn.Module):
    def __init__(self, mean=0.0, std=0.05):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        if self.training:
            return x + torch.randn_like(x) * self.std + self.mean
        return x
    
class CNN_LSTM(nn.Module):
    def __init__(self, input_size, lstm_hidden_size, cnn_channels1 = 64, cnn_channels2 = 128, cnn_channels3 = 256,  dropout=0.3, add_noise=True):
        super(CNN_LSTM, self).__init__()

        self.add_noise = add_noise
        if add_noise:
            self.noise = AddGaussianNoise(std=0.05)
        
        # Conv Layer
        self.conv_stack = nn.Sequential(
            nn.Conv1d(input_size, cnn_channels1, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels1),
            nn.ReLU(),
            nn.Conv1d(cnn_channels1, cnn_channels2, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels2),
            nn.ReLU(),
            nn.Conv1d(cnn_channels2, cnn_channels3, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout)
        )

        # LSTM
        self.lstm = nn.LSTM(cnn_channels3, lstm_hidden_size, num_layers=3,batch_first=True, dropout=dropout, bidirectional=True)
        # self.lstm = nn.LSTM(cnn_channels2, lstm_hidden_size, num_layers=3,batch_first=True, dropout=dropout, bidirectional=True)
        self.attn = nn.MultiheadAttention(embed_dim=lstm_hidden_size*2, num_heads=2, batch_first=True)
        self.linear = nn.Linear(lstm_hidden_size*2, 1)

    def forward(self, x):
        if self.add_noise:
            x = self.noise(x)
        # Input shape: (batch, seq_len, input_size)
        x = x.permute(0, 2, 1)  # -> (batch, input_size, seq_len)

        # CNN layers
        x = self.conv_stack(x)

        x = x.permute(0, 2, 1)  # -> (batch, seq_len, cnn_channels2)

        # LSTM
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)
        out = self.linear(attn_out[:, -1, :])  # Use last time step
        return out


