import os
import logging
from utils import utils
import torch
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error
from config.args_resnet import parse_arguments

# Set up logging to file and console
log_path = "E:/desktop/hydro_channel/experiments/resnet/logs/train_log.txt"
os.makedirs(os.path.dirname(log_path), exist_ok=True)

# Configure the logger
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_path, mode='a'),  # Append to the log file
                        logging.StreamHandler()  # Output to console
                    ])

args = parse_arguments()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the train_model function
def train_model(train_loader, val_loader, model, criterion, optimizer, scheduler, num_epochs=50, model_save_path="best_model.pth"):
    train_losses = []
    val_losses = []

    train_mae_history = []
    train_mse_history = []
    train_rmse_history = []

    val_mae_history = []
    val_mse_history = []
    val_rmse_history = []

    # Initialize early stopping with parsed arguments
    early_stopping = utils.EarlyStopping(patience=args.early_stopping_patience, verbose=True, delta=args.early_stopping_delta)


    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_train_labels = []
        all_train_outputs = []
        total_norm = 0.0
        num_batches = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()

            # Calculate gradient norm
            grad_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            total_norm += grad_norm
            num_batches += 1

            optimizer.step()
            running_loss += loss.item()

            all_train_labels.extend(labels.cpu().numpy())
            all_train_outputs.extend(outputs.cpu().detach().numpy())

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        train_mae = mean_absolute_error(all_train_labels, all_train_outputs)
        train_mse = mean_squared_error(all_train_labels, all_train_outputs)
        train_rmse = math.sqrt(train_mse)

        train_mae_history.append(train_mae)
        train_mse_history.append(train_mse)
        train_rmse_history.append(train_rmse)

        model.eval()
        running_loss = 0.0
        all_val_labels = []
        all_val_outputs = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1))
                running_loss += loss.item()

                all_val_labels.extend(labels.cpu().numpy())
                all_val_outputs.extend(outputs.cpu().detach().numpy())

        val_loss = running_loss / len(val_loader)
        val_losses.append(val_loss)

        val_mae = mean_absolute_error(all_val_labels, all_val_outputs)
        val_mse = mean_squared_error(all_val_labels, all_val_outputs)
        val_rmse = math.sqrt(val_mse)

        val_mae_history.append(val_mae)
        val_mse_history.append(val_mse)
        val_rmse_history.append(val_rmse)

        # Scheduler step
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        # Get learning rate and gradient norm
        current_lr = optimizer.param_groups[0]['lr']
        avg_grad_norm = total_norm / num_batches

        # Log training and validation metrics along with learning rate and gradient norm
        logging.info(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, MAE: {train_mae:.4f}, MSE: {train_mse:.4f}, RMSE: {train_rmse:.4f}')
        logging.info(f'Epoch {epoch+1}, Valid Loss: {val_loss:.4f}, MAE: {val_mae:.4f}, MSE: {val_mse:.4f}, RMSE: {val_rmse:.4f}')
        logging.info(f'Epoch {epoch+1}, Learning Rate: {current_lr:.6f}, Avg Gradient Norm: {avg_grad_norm:.6f}')

        early_stopping(val_loss, model, model_save_path)

        if early_stopping.early_stop:
            logging.info("Early stopping")
            break

    utils.plot_training_metrics(
        train_mae_history, val_mae_history, train_mse_history, val_mse_history, train_rmse_history, val_rmse_history,
        save_path=args.training_validation_loss_path
    )
