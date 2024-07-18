import torch
import torch.nn as nn
from torchvision import models


# Multi-Task Learning model for both driver identification and transport mode classification
class BiLSTMNetwork(nn.Module):
      def __init__(self, input_size, hidden_size, num_layers):
            super(BiLSTMNetwork, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

      def forward(self, x):
            h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.lstm(x, (h0, c0))
            return out[:, -1, :]  # Return the last time step

class ResNet50_GRU(nn.Module):
      def __init__(self, hidden_size, num_layers):
            super(ResNet50_GRU, self).__init__()
            self.resnet50 = models.resnet50(weights='DEFAULT')
            # freeze weights
            for param in self.resnet50.parameters():
                  param.requires_grad = False
            self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-2])
            self.batch_norm = nn.BatchNorm2d(2048)
            self.dropout = nn.Dropout(0.5)
            self.gru = nn.GRU(input_size=2048, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0.5)

      def forward(self, x):
            features = self.resnet50(x)
            features = self.batch_norm(features)
            features = self.dropout(features)
            features = features.view(x.size(0), 7*7, 2048)
            gru_out, _ = self.gru(features)
            return gru_out[:, -1, :]  # Return the last time step

class MultitaskModel(nn.Module):
      def __init__(self, input_size, hidden_size, num_layers):
            super(MultitaskModel, self).__init__()
            self.lstm_network = BiLSTMNetwork(input_size, hidden_size, num_layers)
            self.resnet_gru_network = ResNet50_GRU(hidden_size, num_layers)

            # share fully connected layers
            self.fc1 = nn.Linear(hidden_size * 3, hidden_size)  # Adjusted for the concatenated input size
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.relu = nn.ReLU()

            # task-specific GRUs
            self.gru_driver = nn.GRU(hidden_size, hidden_size, num_layers=2, batch_first=True, dropout=0.5)
            self.gru_transport = nn.GRU(hidden_size, hidden_size, num_layers=2, batch_first=True, dropout=0.5)

            # Task-specific fully connected layers
            self.fc_driver = nn.Linear(hidden_size, 3) # now we're looking at multiclass classification
            self.fc_transport = nn.Linear(hidden_size, 1)
            
            # sigmoid for binary classification
            self.sigmoid = nn.Sigmoid()
            # self.softmax = nn.Softmax(dim=1)  # No need since I'm using CrossEntropyLoss which applies it internally

      def forward(self, x_lstm, x_resnet): #,flags): 
            lstm_out = self.lstm_network(x_lstm)
            resnet_gru_out = self.resnet_gru_network(x_resnet)
            
            # mask the resnet_gru_out based on flags (Why? Not all datapoints are to be trained as they might not be driving data)
            #resnet_gru_out = resnet_gru_out * flags.view(-1, 1)

            # combine lstm_out and resnet_gru_out, considering masked values
            combined_features = torch.cat((lstm_out, resnet_gru_out), dim=1)

            # shared fully connected layers
            shared_out = self.fc1(combined_features)
            shared_out = self.relu(shared_out)
            shared_out = self.fc2(shared_out)
            shared_out = self.relu(shared_out)

            ## task specific branches
            # Driver identification branch
            driver_gru_out, _ = self.gru_driver(shared_out.unsqueeze(1))
            driver_out = self.fc_driver(driver_gru_out[:, -1, :])

            # Transport classification branch
            transport_gru_out, _ = self.gru_transport(shared_out.unsqueeze(1))
            transport_out = self.sigmoid(self.fc_transport(transport_gru_out[:, -1, :]))
            
            return transport_out, driver_out