import torch
import torch.nn as nn
import torch.nn.functional as F


## Model to be trained on longitudinal, transversal and angular velocity signals.

class SimpleCNN(nn.Module):
      def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv1d(in_channels=1, out_channels=224, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(in_channels=224, out_channels=224, kernel_size=3, padding=1)
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(224 * 224, 4)  # Fully connected layer for classification

      def forward(self, x):
            x = F.relu(self.conv1(x))  # Apply ReLU activation after Conv1d
            x = x.unsqueeze(2) # Add dimension for Conv2d: (batch_size, channels, 1, length)
            x = F.relu(self.conv2(x))  # Apply ReLU activation after Conv2d
            x = self.flatten(x)  # Flatten the tensor
            x = self.fc(x)  # Fully connected layer
            #x = F.softmax(x, dim=1) # Apply softmax for multiclass classification
            return x