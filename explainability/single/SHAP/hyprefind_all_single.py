import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingRegressor
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import joblib
import numpy as np

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

# 4. Define hyperparameter space
bagging_param_space = {
    'n_estimators': Integer(50, 2000),          
    'max_samples': Real(0.5, 1.0),            
    'max_features': Real(0.5, 1.0),            
    'bootstrap': Categorical([True, False]),   
    'bootstrap_features': Categorical([True, False]),  
}

# 5. Optimizing hyperparameters with BayesSearchCV
bagging_regressor = BaggingRegressor(random_state=3407)

bayes_search = BayesSearchCV(
    estimator=bagging_regressor,
    search_spaces=bagging_param_space,
    scoring='neg_mean_squared_error',
    cv=5,  
    n_iter=50,  
    n_jobs=-1,
    random_state=3047
)

print("Start Bagging Regressor hyperparameter optimization...")
bayes_search.fit(X_scaled, y_scaled)

best_params = bayes_search.best_params_
print("Optimal hyperparameters:", best_params)

with open("best_bagging_params.txt", "w") as f:
    f.write(str(best_params))

# 6. Training the final model with optimal hyperparameters
final_model = BaggingRegressor(
    **best_params,
    random_state=3407
)
final_model.fit(X_scaled, y_scaled)

joblib.dump(final_model, "bagging_final_model.pkl")
print("The final model has been saved as bagging_final_model.pkl")
