import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

class biLSTM:
      '''Class for training, evaluating and testing the biLSTM model for transport mode classification'''
      def __init__(self, model, optimizer, scheduler, criterion, device):
            self.model = model
            self.optimizer = optimizer
            self.scheduler = scheduler
            self.criterion = criterion
            self.device = device
      
      def train_validation(self, train_loader, val_loader, epochs, save_path=None):
            train_losses = []
            val_losses = []
            train_accuracies = []
            val_accuracies = []

            for epoch in range(epochs):
                  self.model.train()
                  running_loss = 0.0
                  correct = 0
                  total = 0

                  for features, labels in train_loader:
                        features, labels = features.to(self.device), labels.to(self.device)

                        self.optimizer.zero_grad()
                        outputs = self.model(features).squeeze()
                        loss = self.criterion(outputs, labels.float())
                        loss.backward()
                        self.optimizer.step()

                        running_loss += loss.item()
                        predicted = torch.sigmoid(outputs) > 0.5
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                  train_loss = running_loss / len(train_loader)
                  train_accuracy = 100 * correct / total
                  train_losses.append(train_loss)
                  train_accuracies.append(train_accuracy)

                  self.model.eval()
                  val_loss = 0.0
                  correct = 0
                  total = 0

                  with torch.no_grad():
                        for features, labels in val_loader:
                              features, labels = features.to(self.device), labels.to(self.device)
                              outputs = self.model(features).squeeze()
                              loss = self.criterion(outputs, labels.float())
                              val_loss += loss.item()
                              predicted = torch.sigmoid(outputs) > 0.5
                              total += labels.size(0)
                              correct += (predicted == labels).sum().item()

                  val_loss /= len(val_loader)
                  val_accuracy = 100 * correct / total
                  val_losses.append(val_loss)
                  val_accuracies.append(val_accuracy)

                  self.scheduler.step()

                  print(f'   Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

                  # Save checkpoint
                  if save_path:
                        torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': loss,
                        }, save_path)

            return train_losses, val_losses, train_accuracies, val_accuracies
      
      def test(self, test_loader):
            self.model.eval()
            test_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                  for features, labels in test_loader:
                        features, labels = features.to(self.device), labels.to(self.device)
                        outputs = self.model(features).squeeze()
                        loss = self.criterion(outputs, labels.float())
                        test_loss += loss.item()
                        predicted = torch.sigmoid(outputs) > 0.5
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

            test_loss /= len(test_loader)
            test_accuracy = 100 * correct / total

            print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
            return test_loss, test_accuracy
      

class ResNet_GRU:
      '''Class for training, evaluating and testing the ResNet50-GRU model for driver identification'''
      def __init__(self, model, optimizer, scheduler, criterion, device):
            self.model = model
            self.optimizer = optimizer
            self.scheduler = scheduler
            self.criterion = criterion
            self.device = device

      def train_and_evaluate():
            '''Remember to change this from binary class to multiclass'''