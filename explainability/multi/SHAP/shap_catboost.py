import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
import joblib
import numpy as np
import os
import matplotlib
import warnings

matplotlib.use('Agg')
warnings.filterwarnings("ignore", category=UserWarning, module="shap")

# 1. Loading data sets
file_path = r"E:/desktop/hydro_channel/others/cor_multi.csv"
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
    'bagging_temperature': 0.8981468410269265,
    'colsample_bylevel': 0.5,
    'depth': 15,
    'grow_policy': 'Lossguide',
    'iterations': 2000,
    'l2_leaf_reg': 0.0007969261936887926,
    'learning_rate': 0.10242881018472783,
    'min_data_in_leaf': 14,
    'subsample': 0.8175325341308168,
    'random_state': 3407,
    'verbose': 0
}

# 5. Training the model with optimal hyperparameters
final_model = CatBoostRegressor(**best_params)
final_model.fit(X_scaled, y_scaled)

joblib.dump(final_model, "catboost_final_model.pkl")
print("The final model has been saved as catboost_final_model.pkl")

print("Start SHAP analysis...")
explainer = shap.Explainer(final_model)

shap_values = explainer(X_scaled)

output_dir = r"E:/desktop/hydro_channel/explainability/shap_multi"
os.makedirs(output_dir, exist_ok=True)

def save_data_for_plot(filename_base, X_data, shap_vals, feature_names):

    df_to_save = pd.concat([
        pd.DataFrame(X_data, columns=feature_names),
        pd.DataFrame(shap_vals, columns=[f"SHAP_{c}" for c in feature_names])
    ], axis=1)
    df_to_save.to_csv(os.path.join(output_dir, filename_base + ".csv"), index=False)

# SHAP Summary Plot
shap.summary_plot(shap_values, X_scaled, feature_names=X.columns)
fig = plt.gcf()
fig.savefig(os.path.join(output_dir, "shap_summary_plot.png"), dpi=300)
plt.close(fig)
save_data_for_plot("shap_summary_plot", X_scaled, shap_values.values, X.columns)

# SHAP bar plot
shap.summary_plot(shap_values, X_scaled, feature_names=X.columns, plot_type="bar")
fig = plt.gcf()
fig.savefig(os.path.join(output_dir, "shap_bar_plot.png"), dpi=300)
plt.close(fig)
save_data_for_plot("shap_bar_plot", X_scaled, shap_values.values, X.columns)

# SHAP dependence plot 
dependence_feature = 'AS'
shap.dependence_plot(dependence_feature, shap_values.values, X_scaled, feature_names=X.columns)
fig = plt.gcf()
fig.savefig(os.path.join(output_dir, f"shap_dependence_plot_{dependence_feature}.png"), dpi=300)
plt.close(fig)

f_idx = X.columns.get_loc(dependence_feature)
dep_df = pd.DataFrame({
    dependence_feature: X_scaled[:, f_idx],
    f"SHAP_{dependence_feature}": shap_values.values[:, f_idx]
})
dep_df.to_csv(os.path.join(output_dir, f"shap_dependence_plot_{dependence_feature}.csv"), index=False)

# SHAP Waterfall Plot for the first sample
shap_values_exp = shap.Explanation(values=shap_values[0].values,
                                   base_values=shap_values[0].base_values,
                                   data=X.iloc[0].values,
                                   feature_names=X.columns)
shap.waterfall_plot(shap_values_exp)
fig = plt.gcf()
fig.savefig(os.path.join(output_dir, "shap_waterfall_plot_1.png"), dpi=300)
plt.close(fig)
waterfall_df = pd.DataFrame({
    "feature": X.columns,
    "feature_value": X.iloc[0].values,
    "SHAP_value": shap_values[0].values
})
waterfall_df["base_value"] = shap_values[0].base_values
waterfall_df.to_csv(os.path.join(output_dir, "shap_waterfall_plot_1.csv"), index=False)

# SHAP heatmap plot
shap_values_exp = shap.Explanation(values=shap_values.values,
                                   base_values=shap_values.base_values,
                                   data=X_scaled,
                                   feature_names=X.columns)
shap.plots.heatmap(shap_values_exp)
fig = plt.gcf()
fig.savefig(os.path.join(output_dir, "shap_heatmap.png"), dpi=300)
plt.close(fig)
save_data_for_plot("shap_heatmap", X_scaled, shap_values.values, X.columns)

# SHAP decision plot
feature_names = list(X.columns)
shap.decision_plot(
    base_value=shap_values.base_values.mean(),  
    shap_values=shap_values.values,          
    features=X_scaled,                     
    feature_names=feature_names             
)
fig = plt.gcf()
fig.savefig(
    os.path.join(output_dir, "shap_decision_plot_all_samples.png"),
    dpi=300,
    bbox_inches='tight' 
)
plt.close(fig)
save_data_for_plot("shap_decision_plot_all_samples", X_scaled, shap_values.values, X.columns)

# SHAP force plot
force_html_path = os.path.join(output_dir, "shap_force_plot_all_samples.html")
shap.save_html(
    force_html_path,
    shap.force_plot(shap_values.base_values.mean(), shap_values.values, X_scaled, feature_names=feature_names)
)
save_data_for_plot("shap_force_plot_all_samples", X_scaled, shap_values.values, X.columns)

# Calculating Interaction Values with TreeExplainer
shap_interaction_values = explainer.shap_interaction_values(X_scaled)

# Summary Plot of SHAP Interaction Values
shap.summary_plot(shap_interaction_values, X_scaled, feature_names=X.columns)
fig = plt.gcf()
fig.savefig(os.path.join(output_dir, "shap_interaction_summary_plot.png"), dpi=300)
plt.close(fig)

n_samples, n_feats, _ = shap_interaction_values.shape
interaction_flat = shap_interaction_values.reshape(n_samples, -1)

interaction_cols = []
for i in range(n_feats):
    for j in range(n_feats):
        interaction_cols.append(f"SHAP_INTERACTION_{X.columns[i]}_{X.columns[j]}")

interaction_df = pd.concat([
    pd.DataFrame(X_scaled, columns=X.columns),
    pd.DataFrame(interaction_flat, columns=interaction_cols)
], axis=1)
interaction_df.to_csv(os.path.join(output_dir, "shap_interaction_summary_plot.csv"), index=False)

shap_values_df = pd.DataFrame(shap_values.values, columns=X.columns)
shap_values_df.to_csv(os.path.join(output_dir, "shap_values.csv"), index=False)
print("The SHAP value has been saved as shap_values.csv")
