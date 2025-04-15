import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
import os
import matplotlib
import warnings

matplotlib.use('Agg')  
warnings.filterwarnings("ignore", category=UserWarning, module="shap")

# 1. Loading data sets
file_path = r"E:/desktop/hydro_channel/others/cor_single.csv"
df = pd.read_csv(file_path)

# 2. Divide the dataset into features (X) and targets (y)
X = df.drop(columns=['CD'])  
y = df['CD']

# 3. Data pre-processing
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()

# 4. Use of optimal hyperparameters
best_params = {
    'n_estimators': 50,
    'bootstrap': True,
    'bootstrap_features': False,
    'max_features': 1.0,
    'max_samples': 0.6466577189132039,
    'random_state': 3407
}

# 5. Training the model with optimal hyperparameters
final_model = BaggingRegressor(**best_params)
final_model.fit(X_scaled, y_scaled)

joblib.dump(final_model, "bagging_final_model.pkl")
print("The final model has been saved as bagging_final_model.pkl")

# 6. Interpretive analysis using SHAP
print("Start SHAP analysis...")

X_sample = X_scaled[:100] 
explainer = shap.KernelExplainer(final_model.predict, X_sample)

shap_values = explainer.shap_values(X_sample)

output_dir = r"E:/desktop/hydro_channel/explainability/shap_single"
os.makedirs(output_dir, exist_ok=True)

def save_data_for_plot(filename_base, X_data, shap_vals, feature_names):
    df_to_save = pd.concat([
        pd.DataFrame(X_data, columns=feature_names),
        pd.DataFrame(shap_vals, columns=[f"SHAP_{c}" for c in feature_names])
    ], axis=1)
    df_to_save.to_csv(os.path.join(output_dir, filename_base + ".csv"), index=False)

# SHAP Summary Plot 
shap.summary_plot(shap_values, X_sample, feature_names=X.columns)
fig = plt.gcf()
fig.savefig(os.path.join(output_dir, "shap_summary_plot.png"), dpi=300)
plt.close(fig)
save_data_for_plot("shap_summary_plot", X_sample, shap_values, X.columns)

# SHAP bar plot 
shap.summary_plot(shap_values, X_sample, feature_names=X.columns, plot_type="bar")
fig = plt.gcf()
fig.savefig(os.path.join(output_dir, "shap_bar_plot.png"), dpi=300)
plt.close(fig)
save_data_for_plot("shap_bar_plot", X_sample, shap_values, X.columns)

# SHAP dependence plot 
if 'AS' in X.columns:
    shap.dependence_plot('AS', shap_values, X_sample, feature_names=X.columns)
    fig = plt.gcf()
    fig.savefig(os.path.join(output_dir, "shap_dependence_plot_AS.png"), dpi=300)
    plt.close(fig)
    as_idx = list(X.columns).index('AS')
    dep_df = pd.DataFrame({
        'AS': X_sample[:, as_idx],
        'SHAP_AS': shap_values[:, as_idx]
    })
    dep_df.to_csv(os.path.join(output_dir, "shap_dependence_plot_AS.csv"), index=False)

# SHAP Waterfall Plot for the first sample
shap_values_exp = shap.Explanation(
    values=shap_values[0],
    base_values=explainer.expected_value,
    data=X_sample[0],
    feature_names=X.columns
)
shap.waterfall_plot(shap_values_exp)
fig = plt.gcf()
fig.savefig(os.path.join(output_dir, "shap_waterfall_plot_1.png"), dpi=300)
plt.close(fig)

waterfall_df = pd.DataFrame({
    "feature": X.columns,
    "feature_value": X_sample[0],
    "SHAP_value": shap_values[0]
})
waterfall_df["base_value"] = explainer.expected_value
waterfall_df.to_csv(os.path.join(output_dir, "shap_waterfall_plot_1.csv"), index=False)

# SHAP heatmap plot 
shap_values_exp = shap.Explanation(
    values=shap_values,
    base_values=explainer.expected_value,
    data=X_sample,
    feature_names=X.columns
)
shap.plots.heatmap(shap_values_exp)
fig = plt.gcf()
fig.savefig(os.path.join(output_dir, "shap_heatmap.png"), dpi=300)
plt.close(fig)
save_data_for_plot("shap_heatmap", X_sample, shap_values, X.columns)

# SHAP decision plot
shap.decision_plot(
    base_value=explainer.expected_value,
    shap_values=shap_values,
    features=X_sample,
    feature_names=list(X.columns)
)
fig = plt.gcf()
fig.savefig(
    os.path.join(output_dir, "shap_decision_plot_all_samples.png"),
    dpi=300,
    bbox_inches='tight'
)
plt.close(fig)
save_data_for_plot("shap_decision_plot_all_samples", X_sample, shap_values, X.columns)

# SHAP force plot
force_html_path = os.path.join(output_dir, "shap_force_plot_all_samples.html")
shap.save_html(
    force_html_path,
    shap.force_plot(explainer.expected_value, shap_values, X_sample, feature_names=X.columns)
)
save_data_for_plot("shap_force_plot_all_samples", X_sample, shap_values, X.columns)

shap_values_df = pd.DataFrame(shap_values, columns=X.columns)
shap_values_df.to_csv(os.path.join(output_dir, "shap_values.csv"), index=False)
print("The SHAP value has been saved as shap_values.csv")
