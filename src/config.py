## File for storing the configurations of the different model gotten after hyperparameter testing

# configuration for biLSTM model
class biLSTM_config:
      def __init__(self):
            self.data_dir = {
                  'features': './data/lstm_features_labels/lstm_features.npy',
                  'labels': './data/lstm_features_labels/lstm_labels.npy'}
            self.num_epochs = 20
            self.kfolds = 5
            self.learning_rate = 0.001
            self.learning_rate_decay = 0.95
            self.batch_size = 16
            self.input_size = 6
            self.hidden_size = 64
            self.num_layers = 2
            self.dropout = 0.5
            self.optimizer = 'adam'
            self.criterion = 'BCEWithLogitsLoss'
            self.scheduler = {
                  'type': 'lambda_lr',
                  'lr_lambda': lambda epoch: self.learning_rate * (self.learning_rate_decay ** epoch) 
            }
            self.seed = 42


# configuration for Reset50-GRU model
class ResNet50_GRU_config:
      def __init__(self):
            self.data_dir = {
                  'features': './data/feature_maps_labels/feature_maps.npy',
                  'labels': './data/feature_maps_labels/labels.npy'}
            self.num_epochs = 20
            self.kfolds = 5
            self.learning_rate = 0.001
            self.learning_rate_decay = 0.95
            self.batch_size = 16
            self.hidden_size = 64
            self.num_layers = 2
            self.dropout = 0.5
            self.optimizer = 'adam'
            self.criterion = 'CrossEntropyLoss'
            self.scheduler = {
                  'type': 'lambda_lr',
                  'lr_lambda': lambda epoch: self.learning_rate * (self.learning_rate_decay ** epoch) 
            }
            self.seed = 42


# configuration for Reset50-GRU model
class MultiTask_model_config:
      def __init__(self):
            self.data_dir = {
                  'fmaps_features': './data/feature_maps_labels/feature_maps.npy',
                  'fmaps_labels': './data/feature_maps_labels/labels.npy',
                  'seq_features': './data/lstm_features_labels/lstm_features.npy',
                  'seq_labels': './data/lstm_features_labels/lstm_labels.npy'}
            self.num_epochs = 20
            self.kfolds = 5
            self.learning_rate = 0.001
            self.learning_rate_decay = 0.95
            self.batch_size = 16
            self.input_size = 6
            self.hidden_size = 64
            self.num_layers = 2
            self.dropout = 0.5
            self.optimizer = 'adam'
            self.criterion_driver = 'CrossEntropyLoss'
            self.criterion_transport = 'BCELoss'
            self.alpha = 1.0,
            self.beta = 1.0,
            self.scheduler = {
                  'type': 'lambda_lr',
                  'lr_lambda': lambda epoch: self.learning_rate * (self.learning_rate_decay ** epoch) 
            }
            self.seed = 42