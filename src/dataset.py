## File contains the different dataset classes for the different problems
import os
import numpy as np
import csv
import torch
from torch.utils.data import Dataset


## Dataset for simple cnn
class SignalsDataset(Dataset):
      def __init__(self, data, labels, feature_index):
            """
            Args:
                  data (numpy.ndarray): The input data of shape (num_samples, num_features, window_size).
                  labels (numpy.ndarray): The labels for the data of shape (num_samples,).
                  feature_index (int): Index of the feature to use.
            """
            self.data = data[:, feature_index, :]  # Select the specified feature
            self.labels = labels

      def __len__(self):
            return len(self.data)

      def __getitem__(self, idx):
            x = torch.tensor(self.data[idx], dtype=torch.float32).unsqueeze(0)  # Add channel dimension
            y = torch.tensor(self.labels[idx], dtype=torch.long)
            return x, y


## Dataset class for biLSTM model   
class TransportModeDataset(Dataset):
      def __init__(self, base_dir, mode):
            """
            Args:
               base_dir (str): Directory where the lstm feature data is stored.
               mode (str): Data mode ('train', 'valid', or 'test').
            """
            self.base_dir = base_dir
            self.split = mode
            self.metadata = []

            # Read the metadata CSV file to get filenames and labels
            csv_file = os.path.join(self.base_dir, self.split, 'metadata.csv')
            with open(csv_file, mode='r') as file:
                  reader = csv.reader(file)
                  next(reader)  # Skip header row
                  self.metadata = [(row[0], int(row[1])) for row in reader]

      def __len__(self):
            return len(self.metadata)

      def __getitem__(self, idx):
            filename, label = self.metadata[idx]
            file_path = os.path.join(self.base_dir, self.split, filename)
            feature_map = np.load(file_path)
            return torch.tensor(feature_map, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


## Dataset class for ResNet50-GRU model
class FeatureMapDataset(Dataset):
      def __init__(self, base_dir, mode):
            """
            Args:
               base_dir (str): Directory where the feature maps are stored.
               mode (str): Data mode ('train', 'valid', or 'test').
            """
            self.base_dir = base_dir
            self.split = mode
            self.metadata = []

            # Read the metadata CSV file to get filenames and labels
            csv_file = os.path.join(self.base_dir, self.split, 'metadata.csv')
            with open(csv_file, mode='r') as file:
                  reader = csv.reader(file)
                  next(reader)  # Skip header row
                  self.metadata = [(row[0], int(row[1])) for row in reader]

      def __len__(self):
            return len(self.metadata)

      def __getitem__(self, idx):
            filename, label = self.metadata[idx]
            file_path = os.path.join(self.base_dir, self.split, filename)
            feature_map = np.load(file_path)
            return torch.tensor(feature_map, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


## Dataset for MultiTask model
class CombinedDataset(Dataset):
      def __init__(self, fmap_base_dir, lstm_base_dir, mode):
            """
            Args:
               fmap_base_dir (str): Directory where the feature maps are stored.
               lstm_base_dir (str): Directory where the lstm feature data is stored.
               mode (str): Data mode ('train', 'valid', or 'test').
            """
            self.fmap_base_dir = fmap_base_dir
            self.lstm_base_dir = lstm_base_dir
            self.split = mode
            self.fmap_metadata = []
            self.lstm_metadata = []

            csv_file = os.path.join(self.fmap_base_dir, self.split, 'metadata.csv')
            with open(csv_file, mode='r') as file:
                  reader = csv.reader(file)
                  next(reader)  # Skip header row
                  self.fmap_metadata = [(row[0], int(row[1])) for row in reader]

            csv_file = os.path.join(self.lstm_base_dir, self.split, 'metadata.csv')
            with open(csv_file, mode='r') as file:
                  reader = csv.reader(file)
                  next(reader)  # Skip header row
                  self.lstm_metadata = [(row[0], int(row[1])) for row in reader]

      def __len__(self):
            return len(self.lstm_metadata) # both have the same length so it doesn't matter

      def __getitem__(self, idx):
            filename_f, label_f = self.fmap_metadata[idx]
            file_path_f = os.path.join(self.fmap_base_dir, self.split, filename_f)
            feature_map = np.load(file_path_f)

            filename_l, label_l = self.lstm_metadata[idx]
            file_path_l = os.path.join(self.lstm_base_dir, self.split, filename_l)
            lstm = np.load(file_path_l)
      
            return {
                  'sequences': torch.tensor(lstm, dtype=torch.float32),
                  'seq_labels': torch.tensor(label_l, dtype=torch.float32),
                  'feature_maps': torch.tensor(feature_map, dtype=torch.float32),
                  'fmap_labels': torch.tensor(label_f, dtype=torch.float32)
            }