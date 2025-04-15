import torch
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import math
import torch.optim.lr_scheduler as lr_scheduler
import logging

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0.001):
        """
        Args:
            patience (int): Number of epochs to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = float(delta)

    def __call__(self, val_loss, model, model_save_path):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, model_save_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, model_save_path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, model_save_path):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), model_save_path)
        self.val_loss_min = val_loss


def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



def configure_scheduler(optimizer, config):
    if config['lr_scheduler'] == 'StepLR':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['step_size'], gamma=config['gamma'])
    elif config['lr_scheduler'] == 'ExponentialLR':
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config['gamma'])
    elif config['lr_scheduler'] == 'CosineAnnealingLR':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    elif config['lr_scheduler'] == 'ReduceLROnPlateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=config['plateau_patience'], threshold=float(config['plateau_threshold']))
    else:
        raise ValueError(f"Unknown scheduler type: {config['lr_scheduler']}")

# Function to calculate mean and std
def calculate_mean_std(loader, device):
    mean = 0.
    std = 0.
    total_images_count = 0
    
    for images, _ in loader:
        images = images.to(device)
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_samples

    mean /= total_images_count
    std /= total_images_count
    return mean.cpu().numpy(), std.cpu().numpy()

# Function to compute evaluation metrics
def compute_metrics(labels, predictions):
    mae = mean_absolute_error(labels, predictions)
    mse = mean_squared_error(labels, predictions)
    rmse = math.sqrt(mse)
    return mae, mse, rmse

def calculate_mape(actuals, predictions):
    mask = actuals != 0
    if np.sum(mask) == 0:
        return np.array([])
    mape_values = np.abs((actuals[mask] - predictions[mask]) / actuals[mask]) * 100
    return mape_values

# Function to compute MAPE, returns sample-by-sample MAPE array
def calculate_mape(actuals, predictions):
    # Exclude cases where the actual value is zero and prevent division by zero
    mask = actuals != 0  
    if np.sum(mask) == 0:  
        return np.array([])  

    # Calculate sample-by-sample MAPE values
    mape_values = np.abs((actuals[mask] - predictions[mask]) / actuals[mask]) * 100
    return mape_values

def reverse_normalization(normalized_values, mean, std):
    return normalized_values * std + mean

import pandas as pd
import matplotlib.pyplot as plt

def plot_training_metrics(train_mae_history, val_mae_history,
                                   train_mse_history, val_mse_history,
                                   train_rmse_history, val_rmse_history,
                                   save_path=None, csv_path=None):
    """
    Plot MAE, MSE and RMSE curves during training and save their history to a CSV file.

    Args.
        train_mae_history (list): history of training set MAEs
        val_mae_history (list): History of MAE for validation set.
        train_mse_history (list): history of training set MSEs
        val_mse_history (list): history of the validation set MSE
        train_rmse_history (list): history of the training set RMSE
        val_rmse_history (list): history of the validation set RMSE
        save_path (str, optional): path to save the plots
        csv_path (str, optional): path to save the training metrics to a CSV file
    """

    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.labelsize'] = 36
    plt.rcParams['axes.titlesize'] = 36
    plt.rcParams['xtick.labelsize'] = 32
    plt.rcParams['ytick.labelsize'] = 32
    plt.rcParams['legend.fontsize'] = 28
    plt.rcParams['axes.linewidth'] = 2  
    plt.rcParams['xtick.major.width'] = 2  
    plt.rcParams['ytick.major.width'] = 2 

    plt.figure(figsize=(30, 20))

    plt.suptitle('The Loss Curves', fontsize=40, fontweight='bold')

    plt.subplot(3, 1, 1)
    plt.plot(train_mae_history, label='Train MAE', color='blue', linewidth=3)
    plt.plot(val_mae_history, label='Valid MAE', color='orange', linewidth=3)
    plt.title('MAE over Epochs', fontsize=36)
    plt.xlabel('Epochs', fontsize=36)
    plt.ylabel('MAE', fontsize=36)
    plt.legend(fontsize=28)

    plt.subplot(3, 1, 2)
    plt.plot(train_mse_history, label='Train MSE', color='green', linewidth=3)
    plt.plot(val_mse_history, label='Valid MSE', color='red', linewidth=3)
    plt.title('MSE over Epochs', fontsize=36)
    plt.xlabel('Epochs', fontsize=36)
    plt.ylabel('MSE', fontsize=36)
    plt.legend(fontsize=28)

    plt.subplot(3, 1, 3)
    plt.plot(train_rmse_history, label='Train RMSE', color='purple', linewidth=3)
    plt.plot(val_rmse_history, label='Valid RMSE', color='brown', linewidth=3)
    plt.title('RMSE over Epochs', fontsize=36)
    plt.xlabel('Epochs', fontsize=36)
    plt.ylabel('RMSE', fontsize=36)
    plt.legend(fontsize=28)

    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.4)

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")

    if csv_path:
        metrics_data = {
            "Epoch": list(range(1, len(train_mae_history) + 1)),
            "Train MAE": train_mae_history,
            "Valid MAE": val_mae_history,
            "Train MSE": train_mse_history,
            "Valid MSE": val_mse_history,
            "Train RMSE": train_rmse_history,
            "Valid RMSE": val_rmse_history,
        }
        df_metrics = pd.DataFrame(metrics_data)
        df_metrics.to_csv(csv_path, index=False)
        print(f"Metrics history saved to {csv_path}")


import numpy as np
import pandas as pd
import logging
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def calculate_and_print_metrics(train_actuals, train_predictions, test_actuals, test_predictions, output_csv=None):
    """
    Calculate evaluation metrics for training and test data, including R², MAE, MSE, and RMSE, and save results to logs and CSV files.

    Args.
        train_actuals (list or np.array): actuals of the training set
        train_predictions (list or np.array): predictions of the training set
        test_actuals (list or np.array): actuals of the test set
        test_predictions (list or np.array): predictions for the test set
        output_csv (str): Path to the CSV file where the metrics are stored.

    Returns.
        dict: Dictionary containing the evaluation metrics for the training and test sets.
    """

    train_r_squared = r2_score(train_actuals, train_predictions)
    train_mae = mean_absolute_error(train_actuals, train_predictions)
    train_mse = mean_squared_error(train_actuals, train_predictions)
    train_rmse = np.sqrt(train_mse)

    test_r_squared = r2_score(test_actuals, test_predictions)
    test_mae = mean_absolute_error(test_actuals, test_predictions)
    test_mse = mean_squared_error(test_actuals, test_predictions)
    test_rmse = np.sqrt(test_mse)

    logging.info(f"Train R2: {train_r_squared}")
    logging.info(f"Train MAE: {train_mae}")
    logging.info(f"Train MSE: {train_mse}")
    logging.info(f"Train RMSE: {train_rmse}")

    logging.info(f"Test R2: {test_r_squared}")
    logging.info(f"Test MAE: {test_mae}")
    logging.info(f"Test MSE: {test_mse}")
    logging.info(f"Test RMSE: {test_rmse}")

    metrics = {
        'train_r_squared': train_r_squared,
        'train_mae': train_mae,
        'train_mse': train_mse,
        'train_rmse': train_rmse,
        'test_r_squared': test_r_squared,
        'test_mae': test_mae,
        'test_mse': test_mse,
        'test_rmse': test_rmse
    }
    
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(output_csv, index=False)
    
    return metrics

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import logging

def calculate_and_plot_mape(train_actuals, train_predictions, test_actuals, test_predictions, 
                            mape_limit=3, save_path=None, csv_path=None):

    def calculate_mape(actuals, predictions):
        mask = actuals != 0  # Filter out zero values in actuals
        if np.sum(mask) == 0:  # No valid values
            return np.array([])  # Return empty array
        return np.abs((actuals[mask] - predictions[mask]) / actuals[mask]) * 100

    # Calculate sample-wise MAPE for training and testing sets
    train_mape = calculate_mape(np.array(train_actuals), np.array(train_predictions))
    test_mape = calculate_mape(np.array(test_actuals), np.array(test_predictions))

    # Handle NaN and infinite values
    train_mape = train_mape[np.isfinite(train_mape)]
    test_mape = test_mape[np.isfinite(test_mape)]

    # Ensure non-empty MAPE arrays with at least one valid data point
    if train_mape.size > 0 and test_mape.size > 0:
        # Calculate statistics
        average_train_mape = np.mean(train_mape)
        average_test_mape = np.mean(test_mape)
        median_train = np.median(train_mape)
        median_test = np.median(test_mape)
        perc_90_train = np.percentile(train_mape, 90)
        perc_90_test = np.percentile(test_mape, 90)

        # Clip MAPE to a reasonable limit
        train_mape = np.clip(train_mape, 0, mape_limit)
        test_mape = np.clip(test_mape, 0, mape_limit)

        # Sort MAPE values
        train_mape_sorted = np.sort(train_mape)
        test_mape_sorted = np.sort(test_mape)

        # Calculate cumulative distribution probabilities
        cumulative_prob_train = np.arange(1, len(train_mape_sorted) + 1) / len(train_mape_sorted)
        cumulative_prob_test = np.arange(1, len(test_mape_sorted) + 1) / len(test_mape_sorted)

        # Adjust lengths by padding the shorter array with NaNs if necessary
        max_len = max(len(train_mape_sorted), len(test_mape_sorted))
        train_mape_sorted = np.pad(train_mape_sorted, (0, max_len - len(train_mape_sorted)), constant_values=np.nan)
        test_mape_sorted = np.pad(test_mape_sorted, (0, max_len - len(test_mape_sorted)), constant_values=np.nan)
        cumulative_prob_train = np.pad(cumulative_prob_train, (0, max_len - len(cumulative_prob_train)), constant_values=np.nan)
        cumulative_prob_test = np.pad(cumulative_prob_test, (0, max_len - len(cumulative_prob_test)), constant_values=np.nan)

        # Plot cumulative distribution
        plt.figure(figsize=(8, 8))
        plt.step(train_mape_sorted, cumulative_prob_train, label='Training set', color='brown', where='post', linewidth=3)
        plt.step(test_mape_sorted, cumulative_prob_test, label='Test set', color='goldenrod', where='post', linewidth=3)

        # Add horizontal lines for median and 90-percentile
        plt.axhline(0.5, color='grey', linestyle='--', linewidth=3)
        plt.axhline(0.9, color='grey', linestyle='--', linewidth=3)

        # Text box with stats
        textstr = '\n'.join((
            f'Median @ Train = {median_train:.2f}%',
            f'Median @ Test = {median_test:.2f}%',
            f'90-percentile @ Train = {perc_90_train:.2f}%',
            f'90-percentile @ Test = {perc_90_test:.2f}%',
            f'Average @ Train = {average_train_mape:.2f}%',
            f'Average @ Test = {average_test_mape:.2f}%'
        ))
        plt.text(0.95, 0.05, textstr, transform=plt.gca().transAxes, fontsize=14, fontfamily='DejaVu Sans',
                 verticalalignment='bottom', horizontalalignment='right')
        plt.legend(fontsize=14, loc='upper left', frameon=False)
        plt.xlabel('MAPE (%)', fontsize=16, fontfamily='DejaVu Sans')
        plt.ylabel('Cumulative probability (-)', fontsize=16, fontfamily='DejaVu Sans')
        plt.xticks(fontsize=14, fontfamily='DejaVu Sans')
        plt.yticks(fontsize=14, fontfamily='DejaVu Sans')
        plt.grid(False)
        plt.gca().tick_params(axis='both', labelsize=16)
        
        # Save plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Cumulative probability plot saved to {save_path}")
        plt.close()

        # Save MAPE data to CSV
        if csv_path:
            data = {
                "Train MAPE": train_mape_sorted,
                "Cumulative Probability Train": cumulative_prob_train,
                "Test MAPE": test_mape_sorted,
                "Cumulative Probability Test": cumulative_prob_test,
            }
            df = pd.DataFrame(data)
            df.to_csv(csv_path, index=False)
            print(f"MAPE data saved to {csv_path}")

    else:
        print("Error: One or both of the MAPE arrays are empty or contain no valid data. Cannot generate plot.")

    # Calculate R², MAE, MSE, RMSE for test set
    test_r_squared = r2_score(test_actuals, test_predictions)
    test_mae = mean_absolute_error(test_actuals, test_predictions)
    test_mse = mean_squared_error(test_actuals, test_predictions)
    test_rmse = np.sqrt(test_mse)

    return test_r_squared, test_mae, test_mse, test_rmse

def plot_predictions_vs_actuals(train_actuals, train_predictions, test_actuals, test_predictions, 
                                test_r_squared, test_mae, test_mse, test_rmse, save_path=None):
    plt.figure(figsize=(8, 8)) 

    plt.rcParams["font.family"] = "DejaVu Sans"

    # Convert lists to NumPy arrays
    train_actuals = np.array(train_actuals)
    train_predictions = np.array(train_predictions)
    test_actuals = np.array(test_actuals)
    test_predictions = np.array(test_predictions)

    # Create a DataFrame for Seaborn plotting
    df_result = pd.DataFrame({
        'Actuals': np.concatenate([train_actuals, test_actuals]),
        'Predictions': np.concatenate([train_predictions, test_predictions]),
        'Dataset': ['Train'] * len(train_actuals) + ['Test'] * len(test_actuals)
    })

    # Calculate max and min values for diagonal line and axes limits
    max_val = max(df_result['Actuals'].max(), df_result['Predictions'].max())
    min_val = min(df_result['Actuals'].min(), df_result['Predictions'].min())

    # Set the same range for x and y axes
    range_val = [min_val, max_val]

    # Create a joint plot
    g = sns.jointplot(x='Actuals', y='Predictions', data=df_result, 
                      marginal_kws=dict(bins=30), kind='scatter', height=9, color='#FFC3FF')

    # Overlay scatter plot with custom markers and larger marker sizes for Train and Test sets
    sns.scatterplot(x=train_actuals, y=train_predictions, color='skyblue', marker='o', 
                    s=300, ax=g.ax_joint, label='Train')  # Train set with larger markers
    sns.scatterplot(x=test_actuals, y=test_predictions, color='#D1B8F0', marker='^', 
                    s=300, ax=g.ax_joint, label='Test')   # Test set with larger markers

    # Set the legend in the top-left corner with larger font size
    legend = g.ax_joint.legend(loc='upper left', fontsize=18, frameon=False)

    # Add the annotations for test set metrics in the bottom-right corner
    g.ax_joint.text(1.0, 0.05, 
                    f"$R^2_{{test}}$:{test_r_squared:.4f}\n"
                    f"MAE$_{{test}}$:{test_mae:.4f} A cm⁻²\n"
                    f"MSE$_{{test}}$:{test_mse:.4f} (A cm⁻²)²\n"
                    f"RMSE$_{{test}}$:{test_rmse:.4f} A cm⁻²", 
                    transform=g.ax_joint.transAxes, 
                    ha='right', va='bottom', fontsize=18, linespacing=2)

    # Set labels for axes with larger font size
    g.set_axis_labels(f'Multi-physical field calculation (A cm⁻²)', 'Model prediction (A cm⁻²)', fontsize=20)

    # Increase the font size of the tick labels
    g.ax_joint.tick_params(axis='both', which='major', labelsize=18)

    # Set the axes limits to ensure square aspect ratio
    g.ax_joint.set_xlim(range_val)
    g.ax_joint.set_ylim(range_val)
    g.ax_joint.set_aspect('equal', adjustable='box')  # Ensure equal aspect ratio

    # Plot the diagonal line (y = x) for reference
    g.ax_joint.plot(range_val, range_val, ls="--", c=".3", lw=3)  # Thicker diagonal line

    # Save the plot
    if save_path:
        plt.savefig(save_path, dpi=400, bbox_inches='tight')
        print(f"predictions_vs_actuals plot saved to {save_path}")
    else:
        plt.show()


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_residuals_violin(train_actuals, train_predictions, test_actuals, test_predictions):
    """
    Plotting the distribution of prediction errors for the training and test sets (violin plots)

    Parameters.
    train_actuals (list or np.array): actuals of the training set
    train_predictions (list or np.array): predictions of the training set
    test_actuals (list or np.array): actuals of the test set
    test_predictions (list or np.array): predictions for the test set
    """
    # Calculate residuals (actual - predicted)
    train_residuals = np.array(train_actuals) - np.array(train_predictions)
    test_residuals = np.array(test_actuals) - np.array(test_predictions)
    
    data = {
        'Residuals': np.concatenate([train_residuals, test_residuals]),
        'Dataset': ['Train'] * len(train_residuals) + ['Test'] * len(test_residuals)
    }
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(8, 6))
    sns.violinplot(x='Dataset', y='Residuals', data=df)
    plt.title('Residuals Distribution for Train and Test Sets')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.xlabel('Dataset')
    plt.axhline(0, color='red', linestyle='--') 
    plt.savefig('residuals_violin.png')
        
def visualize_contributions(resnet_contribution, vit_contribution, gating_weights, batch_index):
    """
    Visualize the expert model contribution and weight distribution of all samples in the same batch on the same plot
    :param resnet_contribution: contribution of ResNet
    :param vit_contribution: contribution of ViT
    :param gating_weights: weights of the gating network
    :param batch_index: index of the current batch, used to name the saved images
    """

    resnet_contribution = resnet_contribution.cpu().detach().numpy()
    vit_contribution = vit_contribution.cpu().detach().numpy()
    gating_weights = gating_weights.cpu().detach().numpy()

    num_samples = resnet_contribution.shape[0] 
    indices = np.arange(num_samples)

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    width = 0.35 
    ax[0].bar(indices - width/2, resnet_contribution[:, 0], width, label='ResNet Contribution', color='blue')
    ax[0].bar(indices + width/2, vit_contribution[:, 0], width, label='ViT Contribution', color='green')
    ax[0].set_title('Expert Contributions for all Samples in Batch')
    ax[0].set_xlabel('Sample Index')
    ax[0].set_ylabel('Contribution Value')
    ax[0].legend()

    ax[1].bar(indices - width/2, gating_weights[:, 0], width, label='ResNet Weight', color='blue')
    ax[1].bar(indices + width/2, gating_weights[:, 1], width, label='ViT Weight', color='green')
    ax[1].set_title('Gating Weights for all Samples in Batch')
    ax[1].set_xlabel('Sample Index')
    ax[1].set_ylabel('Weight Value')
    ax[1].legend()

    fig.suptitle(f'Contributions and Weights for Batch {batch_index}', fontsize=16)

    plt.savefig(f'contributions_and_weights_batch_{batch_index}.png')
    plt.close()

def aggregate_cross_validation_results(results):
    metrics = {}
    for fold in results:
        for key, value in results[fold].items():
            metrics.setdefault(key, []).append(value)
    avg_metrics = {key: sum(values) / len(values) for key, values in metrics.items()}
    return avg_metrics

def ensemble_predict(models, new_data_loader, device):
    ensemble_preds = []
    for model in models:
        model.eval()
        fold_preds = []
        with torch.no_grad():
            for inputs in new_data_loader:
                inputs = inputs.to(device)
                preds = model(inputs)
                fold_preds.append(preds.cpu().numpy())
        ensemble_preds.append(np.concatenate(fold_preds))
    final_predictions = np.mean(ensemble_preds, axis=0)
    return final_predictions