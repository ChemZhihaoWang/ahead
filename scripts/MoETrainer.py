import torch
import torch.nn as nn
import logging
import os
from scripts.base_trainer import BaseModelTrainer_MOE
from models.moe import MoEModel 
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
from utils import utils

class MoETrainer(BaseModelTrainer_MOE):
    def __init__(self, model, criterion, optimizer, scheduler, device, config, log_path):
        super(MoETrainer, self).__init__(model, criterion, optimizer, scheduler, device, config, log_path)

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.log_path = log_path
        # Setup logging
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(message)s',
                            handlers=[logging.FileHandler(log_path, mode='a'),
                                      logging.StreamHandler()])
        
        self.early_stopping = utils.EarlyStopping(patience=config['early_stopping_patience'], 
                                                verbose=True, 
                                                delta=config['early_stopping_delta'])

    def train_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0
        all_train_labels, all_train_outputs = [], []
        total_norm, num_batches = 0.0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels.unsqueeze(1))
            loss.backward()

            # Calculate gradient norm
            grad_norm = 0.0
            for param in self.model.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            total_norm += grad_norm
            num_batches += 1

            self.optimizer.step()
            running_loss += loss.item()

            all_train_labels.extend(labels.cpu().numpy())
            all_train_outputs.extend(outputs.cpu().detach().numpy())

        avg_loss = running_loss / len(train_loader)
        avg_grad_norm = total_norm / num_batches

        train_mae = mean_absolute_error(all_train_labels, all_train_outputs)
        train_mse = mean_squared_error(all_train_labels, all_train_outputs)
        train_rmse = math.sqrt(train_mse)

        return avg_loss, train_mae, train_mse, train_rmse, avg_grad_norm


    def evaluate_epoch(self, val_loader):
        self.model.eval()
        running_loss = 0.0
        all_val_labels, all_val_outputs = [], []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels.unsqueeze(1))
                running_loss += loss.item()

                all_val_labels.extend(labels.cpu().numpy())
                all_val_outputs.extend(outputs.cpu().detach().numpy())

        avg_loss = running_loss / len(val_loader)
        val_mae = mean_absolute_error(all_val_labels, all_val_outputs)
        val_mse = mean_squared_error(all_val_labels, all_val_outputs)
        val_rmse = math.sqrt(val_mse)

        return avg_loss, val_mae, val_mse, val_rmse

