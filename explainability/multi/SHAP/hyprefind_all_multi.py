import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from catboost import CatBoostRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import joblib
import numpy as np

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

# 4. Define hyperparameter space
catboost_param_space = {
    'iterations': Integer(50, 2000),
    'learning_rate': Real(0.001, 1.0, prior='log-uniform'),
    'depth': Integer(1, 16),
    'l2_leaf_reg': Real(1e-6, 1e2, prior='log-uniform'),
    'bagging_temperature': Real(0, 1),
    'colsample_bylevel': Real(0.5, 1.0),
    'subsample': Real(0.5, 1.0),
    'grow_policy': Categorical(['SymmetricTree', 'Depthwise', 'Lossguide']),
    'min_data_in_leaf': Integer(1, 50),
}

# 5. Optimizing hyperparameters with BayesSearchCV
bayes_search = BayesSearchCV(
    estimator=CatBoostRegressor(verbose=0, random_state=3407),
    search_spaces=catboost_param_space,
    scoring='neg_mean_squared_error',
    cv=5,  
    n_iter=50,  
    n_jobs=-1,
    random_state=3047
)

print("Start CatBoost hyperparameter optimization...")
bayes_search.fit(X_scaled, y_scaled)

best_params = bayes_search.best_params_
print("Optimal hyperparameters:", best_params)

with open("best_catboost_params.txt", "w") as f:
    f.write(str(best_params))

# 6. Training the final model with optimal hyperparameters
final_model = CatBoostRegressor(**best_params, random_state=3407, verbose=0)
final_model.fit(X_scaled, y_scaled)

# Save the trained model
joblib.dump(final_model, "catboost_final_model.pkl")
print("The final model has been saved as catboost_final_model.pkl")
