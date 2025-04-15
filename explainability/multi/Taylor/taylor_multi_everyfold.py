import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import glob
import os
from mpl_toolkits.axisartist import floating_axes
from mpl_toolkits.axisartist.grid_finder import FixedLocator, DictFormatter
from matplotlib.projections import PolarAxes
import scienceplots  

plt.style.use(['science', 'no-latex'])

plt.rcParams.update({
    'font.family': 'Arial',
    'axes.linewidth': 1.5, 
    'axes.labelweight': 'bold' 
})

def set_tayloraxes(fig, location):
    trans = PolarAxes.PolarTransform(apply_theta_transforms=False)
    r1_locs = np.hstack((np.arange(1, 10) / 10.0, [0.95, 0.99]))
    t1_locs = np.arccos(r1_locs)
    gl1 = FixedLocator(t1_locs)
    tf1 = DictFormatter(dict(zip(t1_locs, map(str, r1_locs))))

    r2_locs = np.arange(0, 2, 0.25)
    r2_labels = ['0', '0.25', '0.50', '0.75', '1.00', '1.25', '1.50', '1.75']
    gl2 = FixedLocator(r2_locs)
    tf2 = DictFormatter(dict(zip(r2_locs, r2_labels)))

    ghelper = floating_axes.GridHelperCurveLinear(
        trans, extremes=(0, np.pi/2, 0, 1.75),
        grid_locator1=gl1, tick_formatter1=tf1,
        grid_locator2=gl2, tick_formatter2=tf2
    )
    ax = floating_axes.FloatingSubplot(fig, location, grid_helper=ghelper)
    fig.add_subplot(ax)

    ax.axis["top"].set_axis_direction("bottom")
    ax.axis["top"].toggle(ticklabels=True, label=True)
    ax.axis["top"].major_ticklabels.set_axis_direction("top")
    ax.axis["top"].label.set_axis_direction("top")
    ax.axis["top"].label.set_text("Correlation")
    ax.axis["top"].label.set_fontsize(14)

    ax.axis["left"].set_axis_direction("bottom")
    ax.axis["left"].label.set_text("Standard deviation")
    ax.axis["left"].label.set_fontsize(14)

    ax.axis["right"].set_axis_direction("top")
    ax.axis["right"].toggle(ticklabels=True)
    ax.axis["right"].major_ticklabels.set_axis_direction("left")
    ax.axis["bottom"].set_visible(False)

    ax.grid(True)
    polar_ax = ax.get_aux_axes(trans)

    rs, ts = np.meshgrid(np.linspace(0, 1.75, 100), np.linspace(0, np.pi/2, 100))
    rms = np.sqrt(1 + rs**2 - 2 * rs * np.cos(ts))
    CS = polar_ax.contour(ts, rs, rms, colors='gray', linestyles='--', linewidths=2)  
    plt.clabel(CS, inline=1, fontsize=10)

    polar_ax.plot(np.linspace(0, np.pi/2, 100), np.ones(100), 'r--', linewidth=3, label="Std. Dev. = 1.0")  
    
    return polar_ax

def plot_taylor(ax, std_obs, std_pred, correlation, color, label):
    if -1 <= correlation <= 1:
        theta = np.arccos(correlation)
        radius = std_pred / std_obs if std_obs != 0 else 0
        print(f"Plotting data point: theta={theta}, radius={radius}, label={label}")
        ax.scatter(theta, radius, color=color, s=50, label=label)
    else:
        print(f"Warning: Invalid correlation value {correlation} for data point.")

model_color_map = {
    'AdaBoost Regressor': '#E6194B',       
    'Bagging Regressor': '#3CB44B',         
    'Bayesian Ridge Regression': '#0082C8', 
    'CatBoost': '#F58231',                  
    'Decision Tree': '#911EB4',             
    'ElasticNet Regression': '#46F0F0',    
    'Gradient Boosting': '#F032E6',         
    'KNN Regression': '#D2F53C',           
    'Lasso Regression': '#FABEBE',        
    'LightGBM': '#008080',              
    'Linear Regression': '#E6BEFF',     
    'Neural Network': '#AAFFC3',          
    'Random Forest': '#808000',          
    'Ridge Regression': '#FFD8B1',         
    'Support Vector Machine': '#000080'    
}

main_folder_path = r'E:\desktop\hydro_channel\explainability\multi'

fold_paths = [os.path.join(main_folder_path, f'fold_{i}') for i in range(1, 6)]

for fold_path in fold_paths:

    file_paths = glob.glob(os.path.join(fold_path, '*.csv'))
    
    observed_std_list = []
    predicted_std_list = []
    correlation_list = []
    model_names = []
    
    for file_path in file_paths:
        model_name = os.path.splitext(os.path.basename(file_path))[0]
        print(model_name)
        data = pd.read_csv(file_path)
        model_label = model_name.replace('_predictions_multi', '')
        
        if {'Test True', 'Test Prediction'}.issubset(data.columns):
            test_data = data[['Test True', 'Test Prediction']].dropna()
            if len(test_data) > 1 and test_data['Test True'].nunique() > 1 and test_data['Test Prediction'].nunique() > 1:
                test_true = test_data['Test True']
                test_pred = test_data['Test Prediction']
                
                std_observed = np.std(test_true)
                std_predicted = np.std(test_pred)
                correlation = np.corrcoef(test_true, test_pred)[0, 1]
                
                print(f"Model: {model_name}")
                print(f"Observed Std: {std_observed}, Predicted Std: {std_predicted}, Correlation: {correlation}")
                
                if not (np.isnan(std_observed) or np.isnan(std_predicted) or np.isnan(correlation)):
                    observed_std_list.append(std_observed)
                    predicted_std_list.append(std_predicted)
                    correlation_list.append(correlation)
                    model_names.append(model_label)
                else:
                    print(f"Warning: NaN values encountered for model {model_name}")
            else:
                print(f"Warning: Insufficient data for model {model_name} to calculate correlation.")
        else:
            print(f"Warning: Missing required columns in {file_path}")
    
    if len(model_names) > 0:
        metrics_df = pd.DataFrame({
            'Model': model_names,
            'Standard Deviation (Observed)': observed_std_list,
            'Standard Deviation (Pred)': predicted_std_list,
            'Correlation': correlation_list
        })
        
        data_save_path = os.path.join(fold_path, f"taylor_plot_data_{os.path.basename(fold_path)}.csv")
        metrics_df.to_csv(data_save_path, index=False)
        
        fig = plt.figure(figsize=(8, 8), dpi=1200)
        ax = set_tayloraxes(fig, 111)
        
        for i, (index, row) in enumerate(metrics_df.iterrows()):
            color = model_color_map.get(row['Model'], 'black')
            
            plot_taylor(ax, row['Standard Deviation (Observed)'], row['Standard Deviation (Pred)'], 
                        row['Correlation'], color=color, label=row['Model'])
        
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
        
        plot_save_path = os.path.join(fold_path, f"Taylor_plot_{os.path.basename(fold_path)}.png")
        plt.savefig(plot_save_path, format='png', bbox_inches='tight')
        plt.close(fig)
    else:
        print(f"No valid data points to plot in {fold_path}. Skipping...")
