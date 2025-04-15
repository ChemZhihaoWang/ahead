import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

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

    margin = 0.05 * (max_val - min_val)
    x_min, x_max = min_val - margin, max_val + margin
    y_min, y_max = min_val - margin, max_val + margin

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
    g.set_axis_labels('Multi-physical field calculation (A cm⁻²)', 'Model prediction (A cm⁻²)', fontsize=20)

    # Increase the font size of the tick labels
    g.ax_joint.tick_params(axis='both', which='major', labelsize=18)

    # Set the axes limits to ensure some margin around points
    g.ax_joint.set_xlim(x_min, x_max)
    g.ax_joint.set_ylim(y_min, y_max)
    g.ax_joint.set_aspect('equal', adjustable='box')  # Ensure equal aspect ratio

    # Plot the diagonal line (y = x) for reference
    g.ax_joint.plot([x_min, x_max], [y_min, y_max], ls="--", c=".3", lw=3)  # Thicker diagonal line

    # Save the plot
    if save_path:
        plt.savefig(save_path, dpi=400, bbox_inches='tight', pad_inches=0.5)
        print(f"predictions_vs_actuals plot saved to {save_path}")
        plt.close()
    else:
        plt.show()
        plt.close()


# Main program
folder_path = r"E:\desktop\hydro_channel\explainability\single"

fold_dirs = [d for d in os.listdir(folder_path) if d.startswith('fold_') and os.path.isdir(os.path.join(folder_path, d))]

for fold_dir in fold_dirs:
    fold_path = os.path.join(folder_path, fold_dir)
    csv_files = [f for f in os.listdir(fold_path) if f.endswith('.csv')]

    for csv_file in csv_files:
        file_path = os.path.join(fold_path, csv_file)

        df = pd.read_csv(file_path)

        df = df.dropna(subset=['Test True', 'Test Prediction'])

        train_actuals = df['Train True']
        train_predictions = df['Train Prediction']
        test_actuals = df['Test True']
        test_predictions = df['Test Prediction']

        test_r_squared = r2_score(test_actuals, test_predictions)
        test_mae = mean_absolute_error(test_actuals, test_predictions)
        test_mse = mean_squared_error(test_actuals, test_predictions)
        test_rmse = np.sqrt(test_mse)

        plot_filename = os.path.splitext(csv_file)[0] + '_plot.png'
        save_path = os.path.join(fold_path, plot_filename)

        plot_predictions_vs_actuals(train_actuals, train_predictions, test_actuals, test_predictions, 
                                    test_r_squared, test_mae, test_mse, test_rmse, save_path=save_path)
