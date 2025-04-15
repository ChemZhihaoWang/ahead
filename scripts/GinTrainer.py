from scripts.base_trainer import BaseModelTrainer
from models.gin import GIN  
import torch
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error

class GinTrainer(BaseModelTrainer):
    def __init__(self, criterion, optimizer, scheduler, device, config, log_path, model=None):
        """
        Initialize the GIN model and set up the trainer.
        """
        # Load the GIN model based on the specified configuration in config
        if model is None:
            # Define the GIN model with the configuration parameters
            hidden_dims = [
                config.get('hidden_dim_1', 64),
                config.get('hidden_dim_2', 128),
                config.get('hidden_dim_3', 256),
                config.get('hidden_dim_4', 128),
                config.get('hidden_dim_5', 32)
            ]
            dropout_rate = config.get('dropout_rate', 0.15)
            model = GIN(
                input_dim=config.get('input_dim', 3),
                hidden_dims=hidden_dims,
                output_dim=config.get('output_dim', 1),
                dropout_rate=dropout_rate
            ).to(device)

        # Initialize the BaseModelTrainer with the model, criterion, optimizer, scheduler, and device
        super().__init__(model, criterion, optimizer, scheduler, device, config, log_path)

    def train_epoch(self, train_loader):
        self.model.train()  # Set the model to training mode
        running_loss = 0.0
        all_train_labels, all_train_outputs = [], []
        total_norm, num_batches = 0.0, 0

        for batch in train_loader:
            # Move the batch to the specified device (CPU/GPU)
            batch = batch.to(self.device)

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward pass: model predictions
            outputs = self.model(batch).squeeze(dim=-1)

            # Compute loss
            loss = self.criterion(outputs, batch.y)
            
            # Backward pass and optimization
            loss.backward()

            # Calculate gradient norm for monitoring
            grad_norm = 0.0
            for param in self.model.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            total_norm += grad_norm
            num_batches += 1

            self.optimizer.step()

            # Accumulate running loss and store predictions and true labels
            running_loss += loss.item()
            all_train_labels.extend(batch.y.cpu().numpy())
            all_train_outputs.extend(outputs.cpu().detach().numpy())

        avg_loss = running_loss / len(train_loader)
        avg_grad_norm = total_norm / num_batches

        # Calculate metrics
        train_mae = mean_absolute_error(all_train_labels, all_train_outputs)
        train_mse = mean_squared_error(all_train_labels, all_train_outputs)
        train_rmse = math.sqrt(train_mse)

        return avg_loss, train_mae, train_mse, train_rmse, avg_grad_norm

    def evaluate_epoch(self, val_loader):
        self.model.eval()
        running_loss = 0.0
        all_val_labels, all_val_outputs = [], []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                outputs = self.model(batch).squeeze(dim=-1)
                loss = self.criterion(outputs, batch.y)
                running_loss += loss.item()

                all_val_labels.extend(batch.y.cpu().numpy())
                all_val_outputs.extend(outputs.cpu().detach().numpy())

        avg_loss = running_loss / len(val_loader)

        # Calculate metrics
        val_mae = mean_absolute_error(all_val_labels, all_val_outputs)
        val_mse = mean_squared_error(all_val_labels, all_val_outputs)
        val_rmse = math.sqrt(val_mse)

        return avg_loss, val_mae, val_mse, val_rmse

