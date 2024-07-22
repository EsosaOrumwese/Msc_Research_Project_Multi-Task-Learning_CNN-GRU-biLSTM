## Contains utility functions for the 3 different models
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from src.dataset import TransportModeDataset, FeatureMapDataset, CombinedDataset


class BiLSTM:
      def __init__(self) -> None:
            pass

      def create_dataloaders(self, features, labels, batch_size=16, seed=42):
            '''Create training and validation dataloaders for training'''
            
            train_features, val_features, train_labels, val_labels = train_test_split(features, labels, test_size=0.2, 
                                                                  stratify=labels, random_state=seed)
            
            dataloaders = []

            train_dataset = TransportModeDataset(train_features, train_labels)
            validation_dataset = TransportModeDataset(val_features, val_labels)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
            validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

            dataloaders.append((train_loader, validation_loader))
            
            return dataloaders

      def create_dataloaders_CV(self, features, labels, k=5, batch_size=16, seed=42):
            '''Create training and validation dataloaders for training with kfold CV'''
            
            skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
            dataloaders = []

            for train_index, val_index in skf.split(features, labels):
                  train_features, val_features = features[train_index], features[val_index]
                  train_labels, val_labels = labels[train_index], labels[val_index]

                  train_dataset = TransportModeDataset(train_features, train_labels)
                  validation_dataset = TransportModeDataset(val_features, val_labels)

                  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
                  validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

                  dataloaders.append((train_loader, validation_loader))
            
            return dataloaders
      

class ResNet50_GRU:
      def __init__(self) -> None:
            pass

      def create_dataloaders(self, features, labels, batch_size=16, seed=42):
            '''Create training and validation dataloaders for training'''
            
            train_features, val_features, train_labels, val_labels = train_test_split(features, labels, test_size=0.2, 
                                                                  stratify=labels, random_state=seed)
            
            dataloaders = []

            train_dataset = FeatureMapDataset(train_features, train_labels)
            validation_dataset = FeatureMapDataset(val_features, val_labels)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
            validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

            dataloaders.append((train_loader, validation_loader))
            
            return dataloaders

      def create_dataloaders_CV(self, features, labels, k=5, batch_size=16, seed=42):
            '''Create training and validation dataloaders for training with kfold CV'''
            
            skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
            dataloaders = []

            for train_index, val_index in skf.split(features, labels):
                  train_features, val_features = features[train_index], features[val_index]
                  train_labels, val_labels = labels[train_index], labels[val_index]

                  train_dataset = FeatureMapDataset(train_features, train_labels)
                  validation_dataset = FeatureMapDataset(val_features, val_labels)

                  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
                  validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

                  dataloaders.append((train_loader, validation_loader))
            
            return dataloaders
      

class MultiTaskModel:
      def __init__(self) -> None:
            pass

      def create_combined_dataloaders(self, sequences, seq_labels, feature_maps, fmap_labels, batch_size=16, seed=42):
            '''Create training and validation dataloaders for training'''

            # stratifying by fmap_labels because that one has 3 classes while seq_labels has 2
            train_idx, val_idx = train_test_split(np.arange(len(fmap_labels)), test_size=0.2, random_state=seed, stratify=fmap_labels)
            
            dataloaders = []

            train_sequences, val_sequences = sequences[train_idx], sequences[val_idx]
            train_seq_labels, val_seq_labels = seq_labels[train_idx], seq_labels[val_idx]
            train_feature_maps, val_feature_maps = feature_maps[train_idx], feature_maps[val_idx]
            train_fmap_labels, val_fmap_labels = fmap_labels[train_idx], fmap_labels[val_idx]

            train_dataset = CombinedDataset(train_sequences, train_seq_labels, train_feature_maps, train_fmap_labels)
            validation_dataset = CombinedDataset(val_sequences, val_seq_labels, val_feature_maps, val_fmap_labels)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4)
            validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4)

            dataloaders.append((train_loader, validation_loader))
            
            return dataloaders

      def create_combined_dataloaders_CV(self, sequences, seq_labels, feature_maps, fmap_labels, k=5, batch_size=16, random_seed=42):
            '''Create dataloaders out of the combined dataset using stratified KFOld'''
            skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_seed)
            dataloaders = []

            for train_index, val_index in skf.split(feature_maps, fmap_labels):
                  train_sequences, val_sequences = sequences[train_index], sequences[val_index]
                  train_seq_labels, val_seq_labels = seq_labels[train_index], seq_labels[val_index]
                  train_feature_maps, val_feature_maps = feature_maps[train_index], feature_maps[val_index]
                  train_fmap_labels, val_fmap_labels = fmap_labels[train_index], fmap_labels[val_index]

                  train_dataset = CombinedDataset(train_sequences, train_seq_labels, train_feature_maps, train_fmap_labels)#, train_flags)
                  validation_dataset = CombinedDataset(val_sequences, val_seq_labels, val_feature_maps, val_fmap_labels)#, val_flags)

                  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4)
                  validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4)

                  dataloaders.append((train_loader, validation_loader))

            return dataloaders
