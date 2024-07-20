import time
import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim


###### BiLSTM ---TRANSPORT MODE CLASSIFICATION --- 
class biLSTM_engine:
      '''Class for training, evaluating and testing the biLSTM model for transport mode classification'''
      def __init__(self, model, optimizer, scheduler, criterion, device):
            self.model = model
            self.optimizer = optimizer
            self.scheduler = scheduler
            self.criterion = criterion
            self.device = device

      def safe_save_model(self, save_path, epoch, loss):
            temp_save_path = save_path + ".tmp"
            torch.save({
                  'epoch': epoch,
                  'model_state_dict': self.model.state_dict(),
                  'optimizer_state_dict': self.optimizer.state_dict(),
                  'scheduler_state_dict': self.scheduler.state_dict(),
                  'loss': loss,
            }, temp_save_path)
            os.replace(temp_save_path, save_path)

      def train(self, train_loader):
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

            return train_loss, train_accuracy
      
      def validate(self, val_loader):
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

            return val_loss, val_accuracy, loss
      
      def train_validation(self, train_loader, val_loader, epochs, save_path=None):
            train_losses = []
            val_losses = []
            train_accuracies = []
            val_accuracies = []

            for epoch in range(epochs):
                  # training
                  train_loss, train_accuracy = self.train(train_loader)
                  train_losses.append(train_loss)
                  train_accuracies.append(train_accuracy)

                  # validation
                  val_loss, val_accuracy, loss = self.validate(val_loader)
                  val_losses.append(val_loss)
                  val_accuracies.append(val_accuracy)

                  self.scheduler.step()

                  print(f'   Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

                  # Save checkpoint
                  if save_path:
                        for _ in range(5):  # Retry up to 5 times
                              try:
                                    self.safe_save_model(save_path, epoch, loss)
                                    break
                              except Exception as e:
                                    print(f"Error saving model: {e}. Retrying in 1 second.")
                                    time.sleep(1)

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


##### RESNET50-GRU MODEL ---DRIVER IDENTIFICATION----
class ResNet50_GRU_engine:
      '''Class for training, evaluating and testing the ResNet50-GRU model for driver identification'''
      def __init__(self, model, optimizer, scheduler, criterion, device):
            self.model = model
            self.optimizer = optimizer
            self.scheduler = scheduler
            self.criterion = criterion
            self.device = device

      def safe_save_model(self, save_path, epoch, loss):
            temp_save_path = save_path + ".tmp"
            torch.save({
                  'epoch': epoch,
                  'model_state_dict': self.model.state_dict(),
                  'optimizer_state_dict': self.optimizer.state_dict(),
                  'scheduler_state_dict': self.scheduler.state_dict(),
                  'loss': loss,
            }, temp_save_path)
            os.replace(temp_save_path, save_path)

      def train(self, train_loader):
            self.model.train()
            running_train_loss = 0.0
            correct_train = 0
            total_train = 0

            for inputs, labels in train_loader:
                  inputs, labels = inputs.to(self.device), labels.to(self.device)
                  self.optimizer.zero_grad()
                  outputs = self.model(inputs)
                  loss = self.criterion(outputs, labels.long())
                  loss.backward()
                  self.optimizer.step()
                  
                  running_train_loss += loss.item()
                  _, predicted = torch.max(outputs, 1)
                  correct_train += (predicted == labels).sum().item()
                  total_train += labels.size(0)

            train_loss = running_train_loss / len(train_loader)
            train_accuracy = 100 * correct_train / total_train

            return train_loss, train_accuracy
      
      def validate(self, val_loader):
            self.model.eval()
            running_val_loss = 0.0
            correct_val = 0
            total_val = 0

            with torch.no_grad():
                  for inputs, labels in val_loader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels.long())
                        running_val_loss += loss.item()
                        _, predicted = torch.max(outputs, 1)
                        correct_val += (predicted == labels).sum().item()
                        total_val += labels.size(0)

            val_loss = running_val_loss / len(val_loader)
            val_accuracy = 100 * correct_val / total_val

            return val_loss, val_accuracy, loss
      
      def train_validation(self, train_loader, val_loader, epochs, save_path=None):
            train_losses = []
            val_losses = []
            train_accuracies = []
            val_accuracies = []

            for epoch in range(epochs):
                  # training
                  train_loss, train_accuracy = self.train(train_loader)
                  train_losses.append(train_loss)
                  train_accuracies.append(train_accuracy)

                  # validation
                  val_loss, val_accuracy, loss = self.validate(val_loader)
                  val_losses.append(val_loss)
                  val_accuracies.append(val_accuracy)

                  self.scheduler.step()

                  print(f'   Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

                  # Save checkpoint
                  if save_path:
                        for _ in range(5):  # Retry up to 5 times
                              try:
                                    self.safe_save_model(save_path, epoch, loss)
                                    break
                              except Exception as e:
                                    print(f"Error saving model: {e}. Retrying in 1 second.")
                                    time.sleep(1)

            return train_losses, val_losses, train_accuracies, val_accuracies
      
      def test(self, test_loader):
            self.model.eval()
            correct_test = 0
            total_test = 0
            
            with torch.no_grad():
                  for inputs, labels in test_loader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        outputs = self.model(inputs)
                        _, predicted = torch.max(outputs, 1)
                        correct_test += (predicted == labels).sum().item()
                        total_test += labels.size(0)

            test_accuracy = correct_test / total_test
            print(f'Test Accuracy: {test_accuracy}')

            return test_accuracy


###### MULTITASKLEARNING MODEL FOR BOTH TASKS 
class MTL_engine:
      '''Class for training, evaluating and testing the MTL model for driver identification and transport 
      mode classification'''
      def __init__(self, model, optimizer, scheduler, criterion_driver, criterion_transport, device):
            self.model = model
            self.optimizer = optimizer
            self.scheduler = scheduler
            self.criterion_driver = criterion_driver
            self.criterion_transport = criterion_transport
            self.device = device

      def safe_save_model(self, save_path, epoch, loss):
            temp_save_path = save_path + ".tmp"
            torch.save({
                  'epoch': epoch,
                  'model_state_dict': self.model.state_dict(),
                  'optimizer_state_dict': self.optimizer.state_dict(),
                  'scheduler_state_dict': self.scheduler.state_dict(),
                  'loss': loss,
            }, temp_save_path)
            os.replace(temp_save_path, save_path)

      def train(self, train_loader, alpha, beta):
            self.model.train()
            running_loss = 0.0
            correct_transport_train = 0
            total_transport_train = 0
            correct_driver_train = 0
            total_driver_train = 0

            for batch in train_loader:
                  sequences = batch['sequences'].to(self.device)
                  seq_labels = batch['seq_labels'].to(self.device)
                  feature_maps = batch['feature_maps'].to(self.device)
                  fmap_labels = batch['fmap_labels'].to(self.device)

                  self.optimizer.zero_grad()
                  # Forward pass
                  transport_out, driver_out = self.model(sequences, feature_maps)
                  # Compute transport classification loss
                  loss_transport = self.criterion_transport(transport_out.flatten(), seq_labels.float())

                  # Compute driver identification loss only for flagged samples
                  loss_driver = self.criterion_driver(driver_out, fmap_labels.long())

                  # # weighted sum of losses
                  # print(loss_transport, loss_driver)
                  total_loss = alpha * loss_transport + beta * loss_driver
                  total_loss.backward()

                  # Perform optimization step
                  self.optimizer.step()

                  running_loss += total_loss.item()

                  # calculate accuracy for transport classification 
                  predicted_transport = (transport_out > 0.5).int()
                  correct_transport_train += (predicted_transport == seq_labels).sum().item()
                  total_transport_train += seq_labels.size(0)

                  # Calculate accuracy for driver identification
                  _, predicted_driver = torch.max(driver_out, 1)
                  correct_driver_train += (predicted_driver == fmap_labels).sum().item()
                  total_driver_train += fmap_labels.size(0)

            train_loss = running_loss / len(train_loader)
            train_acc_transport = 100 * correct_transport_train / total_transport_train
            train_acc_driver = 100 * correct_driver_train / total_driver_train

            return train_loss, train_acc_transport, train_acc_driver
      
      def validate(self, val_loader, alpha, beta):
            self.model.eval()
            val_loss = 0.0
            correct_transport_val = 0
            total_transport_val = 0
            correct_driver_val = 0
            total_driver_val = 0

            with torch.no_grad():
                  for batch in val_loader:
                        sequences = batch['sequences'].to(self.device)
                        seq_labels = batch['seq_labels'].to(self.device)
                        feature_maps = batch['feature_maps'].to(self.device)
                        fmap_labels = batch['fmap_labels'].to(self.device)

                        transport_out, driver_out = self.model(sequences, feature_maps)

                        loss_transport = self.criterion_transport(transport_out.flatten(), seq_labels.float())
                        loss_driver = self.criterion_driver(driver_out, fmap_labels.long())

                        total_loss = alpha * loss_transport + beta * loss_driver
                        val_loss += total_loss.item()

                        predicted_transport = (transport_out > 0.5).int()
                        correct_transport_val += (predicted_transport == seq_labels).sum().item()
                        total_transport_val += seq_labels.size(0)

                        _, predicted_driver = torch.max(driver_out, 1)
                        correct_driver_val += (predicted_driver == fmap_labels).sum().item()
                        total_driver_val += fmap_labels.size(0)

            val_loss = val_loss / len(val_loader)
            val_acc_transport = 100 * correct_transport_val / total_transport_val
            val_acc_driver = 100 * correct_driver_val / total_driver_val

            return val_loss, val_acc_transport, val_acc_driver, total_loss
      
      def train_and_evaluate(self, train_loader, val_loader, epochs, 
                             alpha=1.0, beta=1.0, save_path=None):
            train_loss_history = []
            val_loss_history = []
            train_acc_transport_history = []
            val_acc_transport_history = []
            train_acc_driver_history = []
            val_acc_driver_history = []

            for epoch in range(epochs):
                  # training
                  train_loss, train_acc_transport, train_acc_driver = self.train(train_loader, alpha, beta)
                  train_loss_history.append(train_loss)
                  train_acc_transport_history.append(train_acc_transport)
                  train_acc_driver_history.append(train_acc_driver)

                  # validation
                  val_loss, val_acc_transport, val_acc_driver, total_loss = self.validate(val_loader, alpha, beta)
                  val_loss_history.append(val_loss)
                  val_acc_transport_history.append(val_acc_transport)
                  val_acc_driver_history.append(val_acc_driver)

                  self.scheduler.step()

                  print(f'Epoch [{epoch+1}/{epochs}], '
                        f'Train Loss: {train_loss_history[-1]:.4f}, Train Transport Acc: {train_acc_transport_history[-1]:.2f}%, '
                        f'Train Driver Acc: {train_acc_driver_history[-1]:.2f}%, '
                        f'Val Loss: {val_loss_history[-1]:.4f}, Val Transport Acc: {val_acc_transport_history[-1]:.2f}%, '
                        f'Val Driver Acc: {val_acc_driver_history[-1]:.2f}%')

                  # Save checkpoint
                  if save_path:
                        for _ in range(5):  # Retry up to 5 times
                              try:
                                    self.safe_save_model(save_path, epoch, total_loss)
                                    break
                              except Exception as e:
                                    print(f"Error saving model: {e}. Retrying in 1 second.")
                                    time.sleep(1)

            return train_loss_history, val_loss_history, train_acc_transport_history, \
                  val_acc_transport_history, train_acc_driver_history, val_acc_driver_history
      
      def test_model(self, test_loader, criterion_transport, criterion_driver, alpha=1.0, beta=1.0):
            self.model.eval()
            test_loss = 0.0
            correct_test_transport = 0
            correct_test_driver = 0
            total_test_transport = 0
            total_test_driver = 0

            with torch.no_grad():
                  for batch in test_loader:
                        sequences = batch['sequences'].to(self.device)
                        seq_labels = batch['seq_labels'].to(self.device)
                        feature_maps = batch['feature_maps'].to(self.device)
                        fmap_labels = batch['fmap_labels'].to(self.device)

                        transport_out, driver_out = self.model(sequences, feature_maps)

                        # Compute losses
                        loss_transport = criterion_transport(transport_out.flatten(), seq_labels.float())
                        loss_driver = criterion_driver(driver_out, fmap_labels.long())

                        total_loss = alpha * loss_transport + beta * loss_driver
                        test_loss += total_loss.item()

                        # Calculate accuracies
                        predicted_transport = (transport_out > 0.5).int()
                        correct_test_transport += (predicted_transport == seq_labels).sum().item()
                        total_test_transport += seq_labels.size(0)

                        _, predicted_driver = torch.max(driver_out, 1)
                        correct_test_driver += (predicted_driver == fmap_labels).sum().item()
                        total_test_driver += fmap_labels.size(0)

            test_loss /= len(test_loader)
            test_accuracy_transport = 100 * (correct_test_transport / total_test_transport)
            test_accuracy_driver = 100 * (correct_test_driver / total_test_driver)

            print(f'   Test Loss: {test_loss:.4f}, Test Transport Accuracy: {test_accuracy_transport:.2f}%, '
                  f'Test Driver Accuracy: {test_accuracy_driver:.2f}%')

            return test_loss, test_accuracy_transport, test_accuracy_driver