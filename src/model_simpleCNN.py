import torch
import torch.nn as nn
import torch.nn.functional as F


## Model to be trained on longitudinal, transversal and angular velocity signals.
   
class SimpleCNN(nn.Module):
      def __init__(self, l1, l2):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv1d(in_channels=1, out_channels=224, kernel_size=3, padding=1)
            self.batch_norm1 = nn.BatchNorm1d(num_features=224)
            self.conv2 = nn.Conv2d(in_channels=224, out_channels=224, kernel_size=3, padding=1)
            self.batch_norm2 = nn.BatchNorm2d(num_features=224)
            self.maxpool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(224 * 112, l1)  # 112 because of maxpool
            self.fc2 = nn.Linear(l1, l2)
            self.fc3 = nn.Linear(l2,4)

      def forward(self, x):
            x = F.relu(self.conv1(x))  # Apply ReLU activation after Conv1d
            x = self.batch_norm1(x)
            x = x.unsqueeze(2) # Add dimension for Conv2d: (batch_size, channels, 1, length)
            x = F.relu(self.conv2(x))  # Apply ReLU activation after Conv2d
            x = self.batch_norm2(x)
            x = self.maxpool(x)
            x = self.flatten(x)  # Flatten the tensor
            x = F.relu(self.fc1(x))  # Fully connected layer
            x = F.relu(self.fc2(x))
            x = self.fc3(x)

            return x