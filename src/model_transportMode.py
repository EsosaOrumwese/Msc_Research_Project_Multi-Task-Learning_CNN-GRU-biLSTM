import torch
import torch.nn as nn


## Single Task model for transport mode classification. Binary classification (Driving/Not_driving)
class BiLSTMNetwork(nn.Module):
      def __init__(self, input_size, hidden_size, num_layers, dropout=0.5):
            super(BiLSTMNetwork, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            # Bi-LSTM layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=0 if num_layers == 1 else dropout, batch_first=True, bidirectional=True)
            # Fully connected layer
            self.fc = nn.Linear(hidden_size * 2, 8)  # *2 for bidirection, 8 classes of transport mode
            self.relu = nn.ReLU()

      def forward(self, x):
            # Initialize hidden state with zeros
            h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device) # *2 for bidirectional
            c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device)
            
            # LSTM forward pass
            out, _ = self.lstm(x, (h0, c0))
            out = self.relu(out)
            # Decode the hidden state of the last time step
            out = self.fc(out[:, -1, :])
            return out