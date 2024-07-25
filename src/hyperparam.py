# Ensure the necessary imports
import os
import time
import tempfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from ray import tune, train
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler



class RayTuning:
      '''Utilizes Ray and Tune with PyTorch for hyperparameter tuning.

      Args:
            config (dict): Configuration dictionary containing hyperparameters and other settings.
            save_dir (str): Directory where model checkpoints and logs will be saved.
            criterion (torch.nn.Module or list of torch.nn.Module): Loss function(s). For 'MultiTaskModel', provide a list of loss functions.
            model (torch.nn.Module): model class
            modelType (str): Type of model to be used. Must be one of 'SimpleCNN', 'ResNet50GRU', 'BiLSTM', or 'MultiTaskModel'.
            engine: src.engine class which will be used for training and validating the model 

      Raises:
            ValueError: If an invalid `modelType` is provided.
      '''
      def __init__(self, config, save_dir, criterion, model, modelType, engine):
            self.config = config
            self.save_dir = save_dir
            self.model = model
            self.modelType = modelType
            self.engine = engine

            # Validate modelType
            valid_model_types = ['SimpleCNN', 'ResNet50GRU', 'BiLSTM', 'MultiTaskModel']
            if self.modelType not in valid_model_types:
                  raise ValueError(f"Invalid modelType '{self.modelType}'. Must be one of {valid_model_types}.")

            # Handle different criterion cases
            if self.modelType == 'MultiTaskModel':
                  if not isinstance(criterion, list) or len(criterion) != 2:
                        raise ValueError("For 'MultiTaskModel', criterion must be a list of two loss functions.")
                  self.criterion_driver = criterion[0]
                  self.criterion_transport = criterion[1]
            else:
                  self.criterion = criterion
      
      # Define a training function that integrates with Ray Tune
      def train_model(self, Config, train_datasets, valid_datasets):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # initialize model based on its type 
            if self.modelType == 'ResNet50GRU':
                  # __init__(self, hidden_size=512, num_classes=4, num_layers=2, dropout=0.5)
                  model = self.model(hidden_size=Config['hidden_size'], num_layers=Config['num_layers'], 
                                     dropout=Config['dropout']).to(device)
            elif self.modelType == 'BiLSTM':
                  # __init__(self, input_size, hidden_size, num_layers, dropout=0.5):
                  model = self.model(input_size=6, hidden_size=Config['hidden_size'], 
                                     num_layers=Config['num_layers'], dropout=Config['dropout']).to(device)
            elif self.modelType == 'MultiTaskModel':
                  # __init__(self, input_size, hidden_size, num_layers, dropout=0.5):
                  model = self.model(input_size=6, hidden_size=Config['hidden_size'], 
                                     num_layers=Config['num_layers'], dropout=Config['dropout']).to(device)
            else:
                  model = self.model(l1=Config["l1"], l2=Config["l2"]).to(device) #SimpleCNN

            if Config["optimizer"] == "adam":
                  optimizer = optim.Adam(model.parameters(), lr=Config["lr"])
            elif Config["optimizer"] == "adamw":
                  optimizer = optim.AdamW(model.parameters(), lr=Config["lr"])

            if Config["scheduler"] == "ReduceLROnPlateau":
                  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=Config["patience"], factor=Config["gamma"])
            elif Config["scheduler"] == "exp":
                  scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=Config["gamma"])
            elif Config["scheduler"] == "cos":
                  scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config["epochs"])

            # Load existing checkpoint through `get_checkpoint()` API.
            if train.get_checkpoint():
                  loaded_checkpoint = train.get_checkpoint()
                  with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
                        model_state, optimizer_state, scheduler_state = torch.load(
                        os.path.join(loaded_checkpoint_dir, "checkpoint.pt")
                        )
                        model.load_state_dict(model_state)
                        optimizer.load_state_dict(optimizer_state)
                        scheduler.load_state_dict(scheduler_state)
            
            train_loader = DataLoader(train_datasets, batch_size=int(Config["batch_size"]), 
                                    shuffle=True, num_workers=4)
            val_loader = DataLoader(valid_datasets, batch_size=int(Config["batch_size"]), 
                                    shuffle=True, num_workers=4)

            if self.modelType == 'MultiTaskModel':
                  engine = self.engine(model, optimizer, scheduler, self.criterion_driver, self.criterion_transport, device)

                  for epoch in range(Config['epochs']):
                        train_loss, train_acc_tr, train_acc_dr = engine.train(train_loader, Config["alpha"], Config["beta"])
                        val_loss, val_acc_tr, val_acc_dr,_ = engine.validate(val_loader, Config["alpha"], Config["beta"])
                        
                        if Config["scheduler"] == "exp":
                              engine.scheduler.step()
                        elif Config["scheduler"] == "ReduceLROnPlateau":
                              engine.scheduler.step(val_loss)
                        
                        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                              path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
                              torch.save(
                              (model.state_dict(), optimizer.state_dict(), scheduler.state_dict()), path
                              )
                              checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
                              train.report(
                              {"loss": val_loss, "accuracy_dr": val_acc_dr, 
                               "accuracy_tr":val_acc_tr},
                              checkpoint=checkpoint,
                              )

            else:
                  engine = self.engine(model, optimizer, scheduler, self.criterion, device)
            
                  for epoch in range(Config['epochs']):
                        train_loss, train_acc = engine.train(train_loader)
                        val_loss, val_acc, _ = engine.validate(val_loader)

                        if Config["scheduler"] == "exp":
                              engine.scheduler.step()
                        elif Config["scheduler"] == "ReduceLROnPlateau":
                              engine.scheduler.step(val_loss)
                        
                        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                              path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
                              torch.save(
                              (model.state_dict(), optimizer.state_dict(), scheduler.state_dict()), path
                              )
                              checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
                              train.report(
                              {"loss": val_loss, "accuracy": val_acc},
                              checkpoint=checkpoint,
                              )
                        

            print('Finished Training')

      def test_model(self, best_result, test_dataset):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # initialize model based on its type 
            if self.modelType == 'ResNet50GRU':
                  best_trained_model = self.model(hidden_size=best_result.config['hidden_size'], num_layers=best_result.config['num_layers'], 
                                                  dropout=best_result.config['dropout']).to(device)
            elif self.modelType == 'BiLSTM':
                  best_trained_model = self.model(input_size=6, hidden_size=best_result.config['hidden_size'], 
                                     num_layers=best_result.config['num_layers'], dropout=best_result.config['dropout']).to(device)
            elif self.modelType == 'MultiTaskModel':
                  best_trained_model = self.model(input_size=6, hidden_size=best_result.config['hidden_size'], 
                                     num_layers=best_result.config['num_layers'], dropout=best_result.config['dropout']).to(device)
            else:
                  best_trained_model = self.model(l1=best_result.config["l1"], l2=best_result.config["l2"]).to(device) #SimpleCNN

            checkpoint_path = os.path.join(best_result.checkpoint.to_directory(), "checkpoint.pt")

            if best_result.config["optimizer"] == "adam":
                  optimizer = optim.Adam(best_trained_model.parameters(), lr=best_result.config["lr"])
            elif best_result.config["optimizer"] == "adamw":
                  optimizer = optim.AdamW(best_trained_model.parameters(), lr=best_result.config["lr"])

            if best_result.config["scheduler"] == "ReduceLROnPlateau":
                  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=best_result.config["patience"], factor=best_result.config["gamma"])
            elif best_result.config["scheduler"] == "exp":
                  scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=best_result.config["gamma"])
            elif best_result.config["scheduler"] == "cos":
                  scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=best_result.config["epochs"])


            model_state, optimizer_state, scheduler_state = torch.load(checkpoint_path)
            best_trained_model.load_state_dict(model_state)


            test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)

            if self.modelType == 'MultiTaskModel':
                  engine = self.engine(best_trained_model, optimizer, scheduler, 
                                       self.criterion_driver, self.criterion_transport, device)
                  _, test_acc_tr, test_acc_dr = engine.test(test_loader, self.criterion_transport, self.criterion_driver,
                                                            best_result.config['alpha'], best_result.config['beta'])
                  print("Best trial test set accuracy: Driver{:.4f}%, Transport{:.4f}%".format(test_acc_dr, test_acc_tr))
            else:
                  engine = self.engine(best_trained_model, optimizer, scheduler, self.criterion, device)
                  test_acc = engine.test(test_loader)
                  print("Best trial test set accuracy: {:.4f}%".format(test_acc))
            

      def main(self, train_datasets, valid_datasets, test_dataset, num_samples=10, max_num_epochs=10):
            def custom_trial_dirname_creator(trial):
                  '''A  function that generates shorter directory names for the trials'''
                  return f"trial_{trial.trial_id}"

            scheduler = ASHAScheduler(
                  max_t=max_num_epochs,
                  grace_period=1,
                  reduction_factor=2)
            #train_simpl_model(config, train_datasets, valid_datasets)
            tuner = tune.Tuner(
                  tune.with_resources(
                        tune.with_parameters(self.train_model, 
                                          train_datasets=train_datasets,
                                          valid_datasets=valid_datasets),
                        resources={"cpu": 2, "gpu": 1}
                  ),
                  tune_config=tune.TuneConfig(
                        metric="accuracy",
                        mode="max",
                        scheduler=scheduler,
                        num_samples=num_samples,
                        trial_dirname_creator=custom_trial_dirname_creator
                  ),
                  run_config=train.RunConfig(storage_path=os.path.abspath("./ray_results"), name=self.save_dir),
                  param_space=self.config
            )
            results = tuner.fit()
            
            best_result = results.get_best_result("accuracy", "max")

            print("Best trial config: {}".format(best_result.config))
            print("Best trial final validation loss: {}".format(
                  best_result.metrics["loss"]))
            
            if self.modelType == 'MultiTaskModel':
                  print("Best trial final validation accuracy: Driver{:.4f}%, Transport{:.4f}%".format(best_result.metrics["accuracy_dr"], 
                                                                                                       best_result.metrics["accuracy_tr"]))
            else:
                  print("Best trial final validation accuracy: {:.4f}%".format(best_result.metrics["accuracy"]))

            self.test_model(best_result, test_dataset)