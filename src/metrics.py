#### Module for storing metric calculation functions
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

def calc_metrics(y_probs, binned_true_labels, num_classes):
      '''Calculates the precision, recall and f1scores for a given set
      of y_probs and binned_true_labels'''
      precision = dict()
      recall = dict()
      f1_scores = dict()
      thresh = dict()
      for i in range(num_classes):
            precision[i], recall[i], _ = precision_recall_curve(binned_true_labels[:, i], y_probs[:, i])
            f1_scores[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i] + 1e-8)  # Calculate F1 scores

      return precision, recall, f1_scores
