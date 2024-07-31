## File contains the different dataset classes for the different problems
import os
import numpy as np
import csv
import torch
from torch.utils.data import Dataset
from torchvision import transforms


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
            #self.normmean = torch.tensor([0.485, 0.456, 0.406])[feature_index]
            #self.normstd = torch.tensor([0.229, 0.224, 0.225])[feature_index]

      def __len__(self):
            return len(self.data)

      def __getitem__(self, idx):
            arr = self.data[idx]
            #arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-7) # rescale to [0,1]
            x = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
            #x = (x - self.normmean)/self.normstd
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
            self.base_dir = os.path.abspath(base_dir)
            self.split = mode
            self.metadata = []

            # Read the metadata CSV file to get filenames and labels
            csv_file = os.path.join(self.base_dir, self.split, 'metadata.csv')
            with open(csv_file, mode='r') as file:
                  reader = csv.reader(file)
                  next(reader)  # Skip header row
                  self.metadata = [(row[0], int(float(row[1]))) for row in reader]

      def __len__(self):
            return len(self.metadata)

      def __getitem__(self, idx):
            filename, label = self.metadata[idx]
            file_path = os.path.join(self.base_dir, self.split, filename)
            lstm_seq = np.load(file_path)
            return torch.tensor(lstm_seq, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


## Dataset class for ResNet50-GRU model
class FeatureMapDataset(Dataset):
      def __init__(self, base_dir, mode, rescale=True):
            """
            Args:
               base_dir (str): Directory where the feature maps are stored.
               mode (str): Data mode ('train', 'valid', or 'test').
               rescale: Determines if we rescale then normalize the data or not. (Bool)
            """
            self.base_dir = os.path.abspath(base_dir)
            self.split = mode
            self.rescale = rescale
            self.metadata = []

            # Read the metadata CSV file to get filenames and labels
            csv_file = os.path.join(self.base_dir, self.split, 'metadata.csv')
            with open(csv_file, mode='r') as file:
                  reader = csv.reader(file)
                  next(reader)  # Skip header row
                  self.metadata = [(row[0], int(row[1])) for row in reader]

            # Normalization transform to ImageNet mean and std
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

      def __len__(self):
            return len(self.metadata)

      def __getitem__(self, idx):
            filename, label = self.metadata[idx]
            file_path = os.path.join(self.base_dir, self.split, filename)

            # load feature map
            feature_map = np.load(file_path)

            if self.rescale:
                  # Rescale feature map to [0, 1]
                  feature_map_min = feature_map.min(axis=(1, 2), keepdims=True)
                  feature_map_max = feature_map.max(axis=(1, 2), keepdims=True)
                  feature_map = (feature_map - feature_map_min) / (feature_map_max - feature_map_min + 1e-7)  # Add epsilon to avoid division by zero
            
            # Convert to tensor
            feature_map = torch.tensor(feature_map, dtype=torch.float32)
            
            if self.rescale:
                  # Normalize using ImageNet mean and std
                  feature_map = self.normalize(feature_map)

            return feature_map, torch.tensor(label, dtype=torch.float32)
      
# dataset class for all data augmentation
class FeatureMapDatasetALL(Dataset):
      def __init__(self, base_dir, mode, rescale=True, augment=False):
            self.base_dir = os.path.abspath(base_dir)
            self.split = mode
            self.rescale = rescale
            self.augment = augment
            self.metadata = []

            csv_file = os.path.join(self.base_dir, self.split, 'metadata.csv')
            with open(csv_file, mode='r') as file:
                  reader = csv.reader(file)
                  next(reader)
                  self.metadata = [(row[0], int(row[1])) for row in reader]

            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            if self.augment:
                  self.augmentation_transforms = transforms.Compose([
                  transforms.RandomHorizontalFlip(),
                  transforms.RandomApply([transforms.RandomRotation(degrees=5)], p=0.5),
                  transforms.RandomApply([transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)], p=0.5),
                  transforms.Resize((224, 224))
                  ])

      def __len__(self):
            return len(self.metadata)

      def __getitem__(self, idx):
            filename, label = self.metadata[idx]
            file_path = os.path.join(self.base_dir, self.split, filename)
            feature_map = np.load(file_path)

            if self.rescale:
                  feature_map_min = feature_map.min(axis=(1, 2), keepdims=True)
                  feature_map_max = feature_map.max(axis=(1, 2), keepdims=True)
                  feature_map = (feature_map - feature_map_min) / (feature_map_max - feature_map_min + 1e-7)

            feature_map = torch.tensor(feature_map, dtype=torch.float32)

            if self.augment:
                  feature_map = self._apply_augmentations(feature_map)

            if self.rescale:
                  feature_map = self.normalize(feature_map)

            return feature_map, torch.tensor(label, dtype=torch.float32)

      def _apply_augmentations(self, feature_map):
            feature_map = transforms.ToPILImage()(feature_map)
            feature_map = self.augmentation_transforms(feature_map)
            feature_map = transforms.ToTensor()(feature_map)
            return feature_map
      
# dataset for performing all data augm. apart from rotation
class FeatureMapDatasetNORoT(Dataset):
      def __init__(self, base_dir, mode, rescale=True, augment=False):
            self.base_dir = os.path.abspath(base_dir)
            self.split = mode
            self.rescale = rescale
            self.augment = augment
            self.metadata = []

            csv_file = os.path.join(self.base_dir, self.split, 'metadata.csv')
            with open(csv_file, mode='r') as file:
                  reader = csv.reader(file)
                  next(reader)
                  self.metadata = [(row[0], int(row[1])) for row in reader]

            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            if self.augment:
                  self.augmentation_transforms = transforms.Compose([
                  transforms.RandomHorizontalFlip(),
                  transforms.RandomApply([transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)], p=0.5),
                  transforms.Resize((224, 224))
                  ])

      def __len__(self):
            return len(self.metadata)

      def __getitem__(self, idx):
            filename, label = self.metadata[idx]
            file_path = os.path.join(self.base_dir, self.split, filename)
            feature_map = np.load(file_path)

            if self.rescale:
                  feature_map_min = feature_map.min(axis=(1, 2), keepdims=True)
                  feature_map_max = feature_map.max(axis=(1, 2), keepdims=True)
                  feature_map = (feature_map - feature_map_min) / (feature_map_max - feature_map_min + 1e-7)

            feature_map = torch.tensor(feature_map, dtype=torch.float32)

            if self.augment:
                  feature_map = self._apply_augmentations(feature_map)

            if self.rescale:
                  feature_map = self.normalize(feature_map)

            return feature_map, torch.tensor(label, dtype=torch.float32)

      def _apply_augmentations(self, feature_map):
            feature_map = transforms.ToPILImage()(feature_map)
            feature_map = self.augmentation_transforms(feature_map)
            feature_map = transforms.ToTensor()(feature_map)
            return feature_map
      
# dataset for performing only jitter on data
class FeatureMapDatasetJITTER(Dataset):
      def __init__(self, base_dir, mode, rescale=True, augment=False):
            self.base_dir = os.path.abspath(base_dir)
            self.split = mode
            self.rescale = rescale
            self.augment = augment
            self.metadata = []

            csv_file = os.path.join(self.base_dir, self.split, 'metadata.csv')
            with open(csv_file, mode='r') as file:
                  reader = csv.reader(file)
                  next(reader)
                  self.metadata = [(row[0], int(row[1])) for row in reader]

            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            if self.augment:
                  self.augmentation_transforms = transforms.Compose([
                  transforms.RandomApply([transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)], p=0.5),
                  transforms.Resize((224, 224))
                  ])

      def __len__(self):
            return len(self.metadata)

      def __getitem__(self, idx):
            filename, label = self.metadata[idx]
            file_path = os.path.join(self.base_dir, self.split, filename)
            feature_map = np.load(file_path)

            if self.rescale:
                  feature_map_min = feature_map.min(axis=(1, 2), keepdims=True)
                  feature_map_max = feature_map.max(axis=(1, 2), keepdims=True)
                  feature_map = (feature_map - feature_map_min) / (feature_map_max - feature_map_min + 1e-7)

            feature_map = torch.tensor(feature_map, dtype=torch.float32)

            if self.augment:
                  feature_map = self._apply_augmentations(feature_map)

            if self.rescale:
                  feature_map = self.normalize(feature_map)

            return feature_map, torch.tensor(label, dtype=torch.float32)

      def _apply_augmentations(self, feature_map):
            feature_map = transforms.ToPILImage()(feature_map)
            feature_map = self.augmentation_transforms(feature_map)
            feature_map = transforms.ToTensor()(feature_map)
            return feature_map

## Dataset for MultiTask model
class CombinedDataset(Dataset):
      def __init__(self, fmap_base_dir, lstm_base_dir, mode, rescale=False, augment=False):
            """
            Args:
               fmap_base_dir (str): Directory where the feature maps are stored.
               lstm_base_dir (str): Directory where the lstm feature data is stored.
               mode (str): Data mode ('train', 'valid', or 'test').
            """
            self.fmap_base_dir = os.path.abspath(fmap_base_dir)
            self.lstm_base_dir = os.path.abspath(lstm_base_dir)
            self.split = mode
            self.fmap_metadata = []
            self.lstm_metadata = []
            self.rescale = rescale
            self.augment = augment

            csv_file = os.path.join(self.fmap_base_dir, self.split, 'metadata.csv')
            with open(csv_file, mode='r') as file:
                  reader = csv.reader(file)
                  next(reader)  # Skip header row
                  self.fmap_metadata = [(row[0], int(row[1])) for row in reader]

            csv_file = os.path.join(self.lstm_base_dir, self.split, 'metadata.csv')
            with open(csv_file, mode='r') as file:
                  reader = csv.reader(file)
                  next(reader)  # Skip header row
                  self.lstm_metadata = [(row[0], int(float(row[1]))) for row in reader]

            # Normalization transform to ImageNet mean and std
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            # if self.augment:
            #       self.augmentation_transforms = transforms.Compose([
            #       transforms.RandomHorizontalFlip(),
            #       transforms.RandomApply([transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)], p=0.5),
            #       transforms.Resize((224, 224))
            #       ])

            ## added more transformations
            if self.augment:
                  self.augmentation_transforms = transforms.Compose([
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomApply([transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)], p=0.5),
                        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Adjust scale if necessary
                        transforms.RandomRotation(15),
                        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
                        transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))], p=0.3),
                        transforms.Resize((224, 224)),  # Ensure final size is consistent
                  ])

      def __len__(self):
            return len(self.lstm_metadata) # both have the same length so it doesn't matter

      def __getitem__(self, idx):
            ## feature maps
            filename_f, label_f = self.fmap_metadata[idx]
            file_path_f = os.path.join(self.fmap_base_dir, self.split, filename_f)
            feature_map = np.load(file_path_f)

            if self.rescale:
                  feature_map_min = feature_map.min(axis=(1, 2), keepdims=True)
                  feature_map_max = feature_map.max(axis=(1, 2), keepdims=True)
                  feature_map = (feature_map - feature_map_min) / (feature_map_max - feature_map_min + 1e-7)
            
            # Convert to tensor
            feature_map = torch.tensor(feature_map, dtype=torch.float32)
            
            if self.augment:
                  feature_map = self._apply_augmentations(feature_map)

            if self.rescale:
                  feature_map = self.normalize(feature_map)           

            ## lstm sequences
            filename_l, label_l = self.lstm_metadata[idx]
            file_path_l = os.path.join(self.lstm_base_dir, self.split, filename_l)
            lstm = np.load(file_path_l)
      
            return {
                  'sequences': torch.tensor(lstm, dtype=torch.float32),
                  'seq_labels': torch.tensor(label_l, dtype=torch.float32),
                  'feature_maps': feature_map,
                  'fmap_labels': torch.tensor(label_f, dtype=torch.float32)
            }

      def _apply_augmentations(self, feature_map):
            feature_map = transforms.ToPILImage()(feature_map)
            feature_map = self.augmentation_transforms(feature_map)
            feature_map = transforms.ToTensor()(feature_map)
            return feature_map