# dataset.py
import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

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

