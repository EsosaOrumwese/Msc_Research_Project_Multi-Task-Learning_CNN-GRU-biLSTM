import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_history(hist):
      '''Plots history of model. `hist` is a list which contains;
      `loss_hist_train`, `loss_hist_valid`, `accuracy_hist_train`, `accuracy_hist_valid`
      '''
      x_arr = np.arange(len(hist[0])) + 1

      fig = plt.figure(figsize=(12, 4))
      ax = fig.add_subplot(1, 2, 1)
      ax.plot(x_arr, hist[0], '-o', label='Train loss')
      ax.plot(x_arr, hist[1], '--<', label='Validation loss')
      ax.legend(fontsize=15)
      ax.set_xlabel('Epoch', size=15)
      ax.set_ylabel('Loss', size=15)

      ax = fig.add_subplot(1, 2, 2)
      ax.plot(x_arr, hist[2], '-o', label='Train acc.')
      ax.plot(x_arr, hist[3], '--<', label='Validation acc.')
      ax.legend(fontsize=15)
      ax.set_xlabel('Epoch', size=15)
      ax.set_ylabel('Accuracy', size=15)

      #plt.savefig('figures/14_17.png', dpi=300)
      plt.show()


### Plot functions for Feature Maps
# Normalize the feature maps to [0, 1] range for visualization
def normalize_feature_map(feature_map):
      '''Normalize the feature maps to [0, 1] '''
      feature_map_min = feature_map.min(axis=(2, 3), keepdims=True)
      feature_map_max = feature_map.max(axis=(2, 3), keepdims=True)
      normalized_feature_map = (feature_map - feature_map_min) / (feature_map_max - feature_map_min + 1e-8)
      return normalized_feature_map

# visualize feature map
def feature_map_heatmap(feature_map, cmap='viridis'):
      '''Plots the extracted feature map'''
      normalized_feature_map = normalize_feature_map(feature_map)

      sns.heatmap(np.squeeze(normalized_feature_map), cmap=cmap)

      plt.show()