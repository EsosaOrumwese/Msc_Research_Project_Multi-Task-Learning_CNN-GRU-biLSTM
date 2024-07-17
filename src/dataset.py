## File contains the different dataset classes for the different problems
from torch.utils.data import Dataset


## Dataset class for biLSTM model
class TransportModeDataset(Dataset):
      def __init__(self, features, labels):
            self.features = features
            self.labels = labels

      def __len__(self):
            return len(self.features)

      def __getitem__(self, idx):
            return self.features[idx], self.labels[idx]


## Dataset class for ResNet50-GRU model
class FeatureMapDataset(Dataset):
      def __init__(self, feature_maps, labels):
            self.feature_maps = feature_maps
            self.labels = labels

      def __len__(self):
            return len(self.feature_maps)

      def __getitem__(self, idx):
            return self.feature_maps[idx], self.labels[idx]


## Dataset for MultiTask model
class CombinedDataset(Dataset):
      def __init__(self, sequences, seq_labels, feature_maps, fmap_labels):
            self.sequences = sequences
            self.seq_labels = seq_labels
            self.feature_maps = feature_maps
            self.fmap_labels = fmap_labels

      def __len__(self):
            return len(self.sequences)  # Assuming all inputs are of equal length

      def __getitem__(self, idx):
            return {
                  'sequences': self.sequences[idx],
                  'seq_labels': self.seq_labels[idx],
                  'feature_maps': self.feature_maps[idx],
                  'fmap_labels': self.fmap_labels[idx],
            }
