import numpy as np
import matplotlib.pyplot as plt
from ray import init
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

def plot_endgame_history(endgame_hist, history):
      '''Plots history of model after endgame training. `endgame_hist` is a list which contains;
      `loss_hist_train`, `accuracy_hist_train`. While history is the former list which contains
      the older history. Relevant histories are concatenated and plotted.
      '''
      loss = np.concatenate([history[0], endgame_hist[0]])
      acc = np.concatenate([history[2], endgame_hist[1]])
      hist = loss, acc

      x_arr = np.arange(len(hist[0])) + 1

      fig = plt.figure(figsize=(12, 4))
      ax = fig.add_subplot(1, 2, 1)
      ax.plot(x_arr, hist[0], '-o', label='Train loss')
      ax.legend(fontsize=15)
      ax.set_xlabel('Epoch', size=15)
      ax.set_ylabel('Loss', size=15)

      ax = fig.add_subplot(1, 2, 2)
      ax.plot(x_arr, hist[1], '-o', label='Train acc.')
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

class MTL_plot:
      '''Class for specifically plot MTL output has that has two types of accuracies'''
      def __init__(self) -> None:
            pass

      def plot_history(self, hist):
            '''Plots history of model. `hist` is a list which contains;
            train_loss_history, val_loss_history, train_acc_transport_history,
            val_acc_transport_history, train_acc_driver_history, val_acc_driver_history
            '''
            x_arr = np.arange(len(hist[0])) + 1

            fig = plt.figure(figsize=(18, 4))
            ax = fig.add_subplot(1, 3, 1)
            ax.plot(x_arr, hist[0], '-o', label='Train loss')
            ax.plot(x_arr, hist[1], '--<', label='Validation loss')
            ax.legend(fontsize=15)
            ax.set_xlabel('Epoch', size=15)
            ax.set_ylabel('Loss', size=15)

            ax = fig.add_subplot(1, 3, 2)
            ax.plot(x_arr, hist[2], '-o', label='Train transp_acc.')
            ax.plot(x_arr, hist[3], '--<', label='Valid. transp_acc.')
            ax.legend(fontsize=15)
            ax.set_xlabel('Epoch', size=15)
            ax.set_ylabel('Accuracy', size=15)

            ax = fig.add_subplot(1, 3, 3)
            ax.plot(x_arr, hist[4], '-o', label='Train driv_acc.')
            ax.plot(x_arr, hist[5], '--<', label='Valid. driv_acc.')
            ax.legend(fontsize=15)
            ax.set_xlabel('Epoch', size=15)
            ax.set_ylabel('Accuracy', size=15)

            #plt.savefig('figures/14_17.png', dpi=300)
            plt.show()

      def plot_endgame_history(self, endgame_hist, history):
            '''Plots history of model after endgame training. `endgame_hist` is a list which contains;
            `loss_hist_train`, `accuracy_hist_train`. While history is the former list which contains
            the older history. Relevant histories are concatenated and plotted.
            '''
            loss = np.concatenate([history[0], endgame_hist[0]])
            tr_acc = np.concatenate([history[2], endgame_hist[1]])
            dr_acc = np.concatenate([history[4], endgame_hist[2]])

            hist = loss, tr_acc, dr_acc

            x_arr = np.arange(len(hist[0])) + 1

            fig = plt.figure(figsize=(18, 4))
            ax = fig.add_subplot(1, 3, 1)
            ax.plot(x_arr, hist[0], '-o', label='Train loss')
            ax.legend(fontsize=15)
            ax.set_xlabel('Epoch', size=15)
            ax.set_ylabel('Loss', size=15)

            ax = fig.add_subplot(1, 3, 2)
            ax.plot(x_arr, hist[1], '-o', label='Train transp_acc.')
            ax.legend(fontsize=15)
            ax.set_xlabel('Epoch', size=15)
            ax.set_ylabel('Accuracy', size=15)

            ax = fig.add_subplot(1, 3, 3)
            ax.plot(x_arr, hist[2], '-o', label='Train driv_acc.')
            ax.legend(fontsize=15)
            ax.set_xlabel('Epoch', size=15)
            ax.set_ylabel('Accuracy', size=15)

            #plt.savefig('figures/14_17.png', dpi=300)
            plt.show()

#### Metric Plots ######
def plot_precRec_curve(precision, recall, class_labels, num_classes, title="Precision-Recall Curve", figsize=(8,4)):
      '''Plot precision and curve'''
      cmap = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

      # Plot Precision-Recall curve
      plt.figure(figsize=figsize)
      for i in range(num_classes):
            plt.plot(recall[i], precision[i], lw=2, label=class_labels[i], color=cmap[i])

      plt.xlabel("Recall")
      plt.ylabel("Precision")
      plt.title(title)
      plt.legend(loc="best")
      plt.grid(True)
      plt.show()

def plot_F1rec_curve(recall, f1_scores, class_labels, num_classes, title="F1-Score vs. Recall", figsize=(8,4)):
      '''Plots F1-Recall curve'''
      cmap = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

      # Plot F1-Score curve
      plt.figure(figsize=figsize)
      for i in range(num_classes):
            plt.plot(recall[i], f1_scores[i], lw=2, label=class_labels[i], color=cmap[i])

      plt.xlabel("Recall")
      plt.ylabel("F1 Score")
      plt.title(title)
      plt.legend(loc="best")
      plt.grid(True)
      plt.show()

def plot_learning_curve(train_loss, val_loss, title='Learning Curves with Confidence Intervals', legend_text='XX', figsize=(8,4)):
      # Compute mean and standard deviation
      train_mean = np.mean(train_loss, axis=0)
      train_std = np.std(train_loss, axis=0)
      val_mean = np.mean(val_loss, axis=0)
      val_std = np.std(val_loss, axis=0)

      epochs = np.arange(1, len(train_mean)+1)
      n_seeds = train_loss.shape[0]

      # Compute confidence intervals (95% CI)
      ci_multiplier = 1.96 / np.sqrt(n_seeds)
      train_ci = ci_multiplier * train_std
      val_ci = ci_multiplier * val_std

      # Plotting
      plt.figure(figsize=figsize)
      plt.plot(epochs, train_mean, label='Training Loss', color='blue')
      plt.fill_between(epochs, train_mean - train_ci, train_mean + train_ci, color='blue', alpha=0.2)

      plt.plot(epochs, val_mean, label='Validation Loss', color='red')
      plt.fill_between(epochs, val_mean - val_ci, val_mean + val_ci, color='red', alpha=0.2)

      plt.xlabel('Epochs')
      plt.ylabel('Loss')
      plt.title(title)
      plt.legend(title=legend_text)
      plt.grid(True)
      plt.show()

def plot_accuracy(train_accuracies, val_accuracies, title='Accuracy Plot with Confidence Intervals', legend_text='XX', figsize=(8,4)):
      # Compute mean and standard deviation
      train_mean = np.mean(train_accuracies, axis=0)
      train_std = np.std(train_accuracies, axis=0)
      val_mean = np.mean(val_accuracies, axis=0)
      val_std = np.std(val_accuracies, axis=0)

      epochs = np.arange(1, len(train_mean)+1)
      n_seeds = train_accuracies.shape[0]

      # Compute confidence intervals (95% CI)
      ci_multiplier = 1.96 / np.sqrt(n_seeds)
      train_ci = ci_multiplier * train_std
      val_ci = ci_multiplier * val_std

      # Plotting
      plt.figure(figsize=figsize)
      plt.plot(epochs, train_mean, label='Training Accuracy', color='blue')
      plt.fill_between(epochs, train_mean - train_ci, train_mean + train_ci, color='blue', alpha=0.2)

      plt.plot(epochs, val_mean, label='Validation Accuracy', color='red')
      plt.fill_between(epochs, val_mean - val_ci, val_mean + val_ci, color='red', alpha=0.2)

      plt.xlabel('Epochs')
      plt.ylabel('Accuracy')
      plt.title(title)
      plt.legend(title=legend_text)
      plt.grid(True)
      plt.show()
