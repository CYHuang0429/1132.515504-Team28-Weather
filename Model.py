import torch
import torch.nn as nn

import torch.nn as nn

class Temperature_LSTM(nn.Module):
    def __init__(self, seq_len=1, in_channels=32):
        super(Temperature_LSTM, self).__init__()

        self.seq_len = seq_len

        # CNN layers
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=256, kernel_size=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=1)  # no-op here but kept for structure

        conv_out_len = (seq_len - 1)  # kernel_size=1 does not reduce length, so seq_len remains 1
        conv_out_len = max(conv_out_len, 1)  # to avoid zero or negative length

        self.flattened_size = 128 * conv_out_len

        # RepeatVector equivalent
        self.repeat_vector = nn.Linear(self.flattened_size, seq_len * 100)

        # LSTM layers
        self.lstm1 = nn.LSTM(input_size=100, hidden_size=100, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(input_size=100, hidden_size=100, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=100, hidden_size=100, batch_first=True)

        # Bidirectional LSTM
        self.bi_lstm = nn.LSTM(input_size=100, hidden_size=128, batch_first=True, bidirectional=True)

        # Dense layers
        self.fc1 = nn.Linear(128 * 2, 100)  # bidirectional → 2x hidden size
        self.relu_final = nn.ReLU()
        self.fc2 = nn.Linear(100, 1)  # regression output

    def forward(self, x):
        # x shape: (batch, in_channels=6, seq_len=1)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        x = self.repeat_vector(x)
        x = x.view(x.size(0), self.seq_len, 100)

        x, _ = self.lstm1(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)

        x, _ = self.bi_lstm(x)
        x = x[:, -1, :]

        x = self.relu_final(self.fc1(x))
        output = self.fc2(x)
        return output
