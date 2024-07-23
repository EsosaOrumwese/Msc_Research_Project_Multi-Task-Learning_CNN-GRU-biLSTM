import torch
import torch.nn as nn
from torchvision import models


## Single Task model for driver identification. Multiclass classification (Not_driving, Driver1, Driver2)
## `Nor_driving` class is more like a dummy label which represents data that's not associated with any driver or even driving at all 

class ResNet50_GRU(nn.Module):
      def __init__(self, hidden_size=512, num_classes=4):
            super(ResNet50_GRU, self).__init__()

            self.resnet50 = models.resnet50(weights='DEFAULT')

            # Freeze ResNet-50 layers
            for param in self.resnet50.parameters():
                  param.requires_grad = False

            # Remove the fully connected layer of ResNet-50
            self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-2])
            
            # Add Batch Normalization and Dropout layers
            self.batch_norm = nn.BatchNorm2d(2048)
            self.dropout = nn.Dropout(0.5)

            # Define a GRU layer
            self.gru = nn.GRU(input_size=2048, hidden_size=hidden_size, num_layers=2, batch_first=True, dropout=0.5)

            # Define fully connected and sigmoid layers
            self.fc = nn.Linear(hidden_size, num_classes)

      def forward(self, x):
            batch_size, C, H, W = x.size()

            # Extract features using ResNet-50
            features = self.resnet50(x)
            
            # Apply batch normalization and dropout
            features = self.batch_norm(features)
            features = self.dropout(features)
            
            # Reshape features to (batch_size, 7*7, 2048) for GRU input
            features = features.view(batch_size, 7*7, 2048)
            
            # Pass the features through GRU
            gru_out, _ = self.gru(features)
            
            # Take the output of the last time step
            gru_out = gru_out[:, -1, :]
            
            # Fully connected layer
            out = self.fc(gru_out)
            
            return out
