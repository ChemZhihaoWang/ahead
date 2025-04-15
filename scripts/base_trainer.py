import os
import logging
import torch
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error
from utils import utils

class BaseModelTrainer:
    def __init__(self, model, criterion, optimizer, scheduler, device, config, log_path):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config

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

    def train(self, train_loader, val_loader, num_epochs, model_save_path):
        train_losses, val_losses = [], []
        train_mae_history, train_mse_history, train_rmse_history = [], [], []
        val_mae_history, val_mse_history, val_rmse_history = [], [], []

        for epoch in range(num_epochs):
            # Training step
            train_loss, train_mae, train_mse, train_rmse, avg_grad_norm = self.train_epoch(train_loader)
            train_losses.append(train_loss)
            train_mae_history.append(train_mae)
            train_mse_history.append(train_mse)
            train_rmse_history.append(train_rmse)

            # Validation step
            val_loss, val_mae, val_mse, val_rmse = self.evaluate_epoch(val_loader)
            val_losses.append(val_loss)
            val_mae_history.append(val_mae)
            val_mse_history.append(val_mse)
            val_rmse_history.append(val_rmse)

            # Scheduler step
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']

            # Log the metrics
            logging.info(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, MAE: {train_mae:.4f}, MSE: {train_mse:.4f}, RMSE: {train_rmse:.4f}')
            logging.info(f'Epoch {epoch+1}, Valid Loss: {val_loss:.4f}, MAE: {val_mae:.4f}, MSE: {val_mse:.4f}, RMSE: {val_rmse:.4f}')
            logging.info(f'Epoch {epoch+1}, Learning Rate: {current_lr:.6f}, Avg Gradient Norm: {avg_grad_norm:.6f}')

            # Early stopping
            self.early_stopping(val_loss, self.model, model_save_path)
            if self.early_stopping.early_stop:
                logging.info("Early stopping")
                break

        return {
                    'train_mae_history': train_mae_history,
                    'val_mae_history': val_mae_history,
                    'train_mse_history': train_mse_history,
                    'val_mse_history': val_mse_history,
                    'train_rmse_history': train_rmse_history,
                    'val_rmse_history': val_rmse_history,
                    'train_losses': train_losses,
                    'val_losses': val_losses
                }

class BaseModelTrainer_MOE:
    def __init__(self, model, criterion, optimizer, scheduler, device, config, log_path):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer  
        self.scheduler = scheduler  
        self.device = device
        self.config = config

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

    def train(self, train_loader, val_loader, num_epochs, model_save_path=None):
        train_losses, val_losses = [], []
        train_mae_history, train_mse_history, train_rmse_history = [], [], []
        val_mae_history, val_mse_history, val_rmse_history = [], [], []

        for epoch in range(num_epochs):
            # Training step
            train_loss, train_mae, train_mse, train_rmse, avg_grad_norm = self.train_epoch(train_loader)
            train_losses.append(train_loss)
            train_mae_history.append(train_mae)
            train_mse_history.append(train_mse)
            train_rmse_history.append(train_rmse)

            # Validation step
            val_loss, val_mae, val_mse, val_rmse = self.evaluate_epoch(val_loader)
            val_losses.append(val_loss)
            val_mae_history.append(val_mae)
            val_mse_history.append(val_mse)
            val_rmse_history.append(val_rmse)

            # Scheduler step
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']

            # Log the metrics
            logging.info(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, MAE: {train_mae:.4f}, MSE: {train_mse:.4f}, RMSE: {train_rmse:.4f}')
            logging.info(f'Epoch {epoch+1}, Valid Loss: {val_loss:.4f}, MAE: {val_mae:.4f}, MSE: {val_mse:.4f}, RMSE: {val_rmse:.4f}')
            logging.info(f'Epoch {epoch+1}, Learning Rate: {current_lr:.6f}, Avg Gradient Norm: {avg_grad_norm:.6f}')

            # Early stopping
            self.early_stopping(val_loss, self.model, model_save_path)
            if self.early_stopping.early_stop:
                logging.info("Early stopping")
                break

        return {
            'train_mae_history': train_mae_history,
            'val_mae_history': val_mae_history,
            'train_mse_history': train_mse_history,
            'val_mse_history': val_mse_history,
            'train_rmse_history': train_rmse_history,
            'val_rmse_history': val_rmse_history,
            'train_losses': train_losses,
            'val_losses': val_losses
        }
