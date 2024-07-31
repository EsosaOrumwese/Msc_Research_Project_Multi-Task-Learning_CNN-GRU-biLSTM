import torch
import torch.nn as nn
from torchvision import models


# Multi-Task Learning model for both driver identification and transport mode classification
class BiLSTMNetwork(nn.Module):
      def __init__(self, input_size, hidden_size, num_layers, dropout):
            super(BiLSTMNetwork, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=0 if num_layers == 1 else dropout, 
                                batch_first=True, bidirectional=True)
            self.batch_norm = nn.BatchNorm1d(hidden_size * 2)  # BiLSTM output size

      def forward(self, x):
            h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.lstm(x, (h0, c0))
            out = self.batch_norm(out[:, -1, :])  # Apply batch normalization
            return out # Return the last time step

class ResNet50_GRU(nn.Module):
      def __init__(self, hidden_size, num_layers, dropout, unfreeze_L3, unfreeze_L4):
            super(ResNet50_GRU, self).__init__()
            self.resnet50 = models.resnet50(weights='DEFAULT')
            # freeze weights
            for param in self.resnet50.parameters():
                  param.requires_grad = False

            # Optionally unfreeze some of the last layers
            for param in self.resnet50.layer4.parameters():
                  param.requires_grad = unfreeze_L4

            for param in self.resnet50.layer3.parameters():
                  param.requires_grad = unfreeze_L3
                  
            self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-2])
            self.batch_norm = nn.BatchNorm2d(2048)
            self.dropout = nn.Dropout(dropout)
            self.gru = nn.GRU(input_size=2048, hidden_size=hidden_size, num_layers=num_layers, 
                              batch_first=True, dropout=0 if num_layers == 1 else dropout)
            self.output_norm = nn.BatchNorm1d(hidden_size)  # Normalize GRU output

      def forward(self, x):
            features = self.resnet50(x)
            features = self.batch_norm(features)
            features = self.dropout(features)
            features = features.view(x.size(0), 7*7, 2048)
            gru_out, _ = self.gru(features)
            gru_out = self.output_norm(gru_out[:, -1, :])  # Apply batch normalization
            return gru_out

class MultitaskModel(nn.Module):
      def __init__(self, input_size, hidden_size, num_layers, dropout=0.5, unfreeze_L3=True, unfreeze_L4=True):
            super(MultitaskModel, self).__init__()
            self.lstm_network = BiLSTMNetwork(input_size, hidden_size, num_layers, dropout)
            self.resnet_gru_network = ResNet50_GRU(hidden_size, num_layers, dropout, unfreeze_L3, unfreeze_L4)

            # share fully connected layers
            self.fc1 = nn.Linear(hidden_size * 3, hidden_size)  # Adjusted for the concatenated input size
            self.fc1_bn = nn.BatchNorm1d(hidden_size)  # Batch normalization for shared layers
            self.fc1_dropout = nn.Dropout(dropout)  # Dropout for shared layers
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc2_bn = nn.BatchNorm1d(hidden_size)  # Batch normalization for shared layers
            self.fc2_dropout = nn.Dropout(dropout)  # Dropout for shared layers
            self.relu = nn.ReLU()

            # task-specific GRUs
            self.gru_driver = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=True, 
                                     dropout=0 if num_layers == 1 else dropout)
            self.gru_transport = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=True, 
                                        dropout=0 if num_layers == 1 else dropout)

            # Task-specific fully connected layers
            self.fc_driver = nn.Linear(hidden_size, 4) # now we're looking at multiclass classification
            self.fc_transport = nn.Linear(hidden_size, 8) # also looking at multiclass here (8 transport modes)
            

      def forward(self, x_lstm, x_resnet): #,flags): 
            lstm_out = self.lstm_network(x_lstm)
            resnet_gru_out = self.resnet_gru_network(x_resnet)

            # combine lstm_out and resnet_gru_out, considering masked values
            combined_features = torch.cat((lstm_out, resnet_gru_out), dim=1)

            # shared fully connected layers
            shared_out = self.fc1(combined_features)
            shared_out = self.fc1_bn(shared_out)  # Apply batch normalization
            shared_out = self.relu(shared_out)
            shared_out = self.fc1_dropout(shared_out)  # Apply dropout
            shared_out = self.fc2(shared_out)
            shared_out = self.fc2_bn(shared_out)  # Apply batch normalization
            shared_out = self.relu(shared_out)
            shared_out = self.fc2_dropout(shared_out)  # Apply dropout

            ## task specific branches
            # Driver identification branch
            driver_gru_out, _ = self.gru_driver(shared_out.unsqueeze(1))
            driver_out = self.fc_driver(driver_gru_out[:, -1, :])

            # Transport classification branch
            transport_gru_out, _ = self.gru_transport(shared_out.unsqueeze(1))
            transport_out = self.fc_transport(transport_gru_out[:, -1, :])
            
            return transport_out, driver_out