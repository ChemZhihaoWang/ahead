import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor
from skopt import BayesSearchCV  
import numpy as np
from skopt.space import Real, Integer, Categorical
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler 
import os
import joblib

class MLPRegressorWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, hidden_layer_sizes=100, activation='relu', solver='adam',
                 alpha=0.0001, batch_size='auto', learning_rate_init=0.001, max_iter=200, random_state=3407):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.random_state = random_state
    
    def fit(self, X, y):
        hidden_layer_sizes = self.hidden_layer_sizes
        if isinstance(hidden_layer_sizes, int):
            hidden_layer_sizes = (hidden_layer_sizes,)
        self.model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                                  activation=self.activation,
                                  solver=self.solver,
                                  alpha=self.alpha,
                                  batch_size=self.batch_size,
                                  learning_rate_init=self.learning_rate_init,
                                  max_iter=self.max_iter,
                                  random_state=self.random_state)
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)

# 1. Loading data sets
file_path = r"E:/desktop/hydro_channel/others/cor_single.csv"
df = pd.read_csv(file_path)

df.dropna(inplace=True)

# 2. Divide the dataset into features (X) and targets (y)
X = df.drop(columns=['CD'])  
y = df['CD']

models = {
    "Linear Regression": LinearRegression(),
    "Lasso Regression": Lasso(),
    "Ridge Regression": Ridge(),
    "ElasticNet Regression": ElasticNet(),
    "Bayesian Ridge Regression": BayesianRidge(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(random_state=3407),
    "Bagging Regressor": BaggingRegressor(random_state=3407),
    "Gradient Boosting": GradientBoostingRegressor(random_state=3407),
    "AdaBoost Regressor": AdaBoostRegressor(random_state=3407),
    "XGBoost": xgb.XGBRegressor(random_state=3407),
    "LightGBM": lgb.LGBMRegressor(),
    "CatBoost": CatBoostRegressor(verbose=0, random_state=3407),
    "Support Vector Machine": SVR(),
    "KNN Regression": KNeighborsRegressor(),
    "Neural Network": MLPRegressorWrapper(random_state=3407)
}

param_space = {
    # Regression class models
    "Lasso Regression": {'alpha': Real(1e-6, 1e2, prior='log-uniform')},
    "Ridge Regression": {'alpha': Real(1e-6, 1e2, prior='log-uniform')},
    "ElasticNet Regression": {
        'alpha': Real(1e-6, 1e2, prior='log-uniform'),
        'l1_ratio': Real(0.0, 1.0)
    },
    # Decision tree type models
    "Decision Tree": {
        'max_depth': Integer(1, 30),
        'min_samples_split': Integer(2, 50),
        'min_samples_leaf': Integer(1, 50),
        'max_features': Categorical(['auto', 'sqrt', 'log2', None])
    },
    "Random Forest": {
        'n_estimators': Integer(50, 2000),
        'max_depth': Integer(1, 30),
        'min_samples_split': Integer(2, 50),
        'min_samples_leaf': Integer(1, 50),
        'max_features': Categorical(['auto', 'sqrt', 'log2', None]),
        'bootstrap': Categorical([True, False])
    },
    "Bagging Regressor": {
        'n_estimators': Integer(50, 2000),
        'max_samples': Real(0.5, 1.0),
        'max_features': Real(0.5, 1.0),
        'bootstrap': Categorical([True, False]),
        'bootstrap_features': Categorical([True, False])
    },
    # Boost class models
    "Gradient Boosting": {
        'n_estimators': Integer(50, 2000),
        'learning_rate': Real(0.001, 1.0, prior='log-uniform'),
        'max_depth': Integer(1, 30),
        'subsample': Real(0.5, 1.0),
        'min_samples_split': Integer(2, 50),
        'min_samples_leaf': Integer(1, 50),
        'max_features': Categorical(['auto', 'sqrt', 'log2', None])
    },
    "AdaBoost Regressor": {
        'n_estimators': Integer(50, 2000),
        'learning_rate': Real(0.001, 1.0, prior='log-uniform'),
        'loss': Categorical(['linear', 'square', 'exponential'])
    },
    "XGBoost": {
        'n_estimators': Integer(50, 2000),
        'max_depth': Integer(1, 30),
        'learning_rate': Real(0.001, 1.0, prior='log-uniform'),
        'subsample': Real(0.5, 1.0),
        'colsample_bytree': Real(0.5, 1.0),
        'gamma': Real(0, 5),
        'reg_alpha': Real(0, 1),
        'reg_lambda': Real(1, 10)
    },
    "LightGBM": {
        'n_estimators': Integer(50, 2000),
        'learning_rate': Real(0.001, 1.0, prior='log-uniform'),
        'max_depth': Integer(-1, 30),
        'num_leaves': Integer(20, 300),
        'min_child_samples': Integer(5, 100),
        'subsample': Real(0.5, 1.0),
        'colsample_bytree': Real(0.5, 1.0),
        'reg_alpha': Real(0, 1),
        'reg_lambda': Real(0, 1)
    },
    "CatBoost": {
        'iterations': Integer(50, 2000),
        'learning_rate': Real(0.001, 1.0, prior='log-uniform'),
        'depth': Integer(1, 16),
        'l2_leaf_reg': Real(1e-6, 1e2, prior='log-uniform'),
        'bagging_temperature': Real(0, 1),
        'colsample_bylevel': Real(0.5, 1.0),
        'subsample': Real(0.5, 1.0),
        'grow_policy': Categorical(['SymmetricTree', 'Depthwise', 'Lossguide']),
        'min_data_in_leaf': Integer(1, 50),
    },
    # Other models
    "Support Vector Machine": {
        'C': Real(1e-3, 1e3, prior='log-uniform'),
        'epsilon': Real(1e-4, 1e1, prior='log-uniform'),
        'kernel': Categorical(['linear', 'rbf', 'poly', 'sigmoid']),
        'degree': Integer(2, 5),  
        'gamma': Real(1e-4, 1e1, prior='log-uniform'), 
    },
    "KNN Regression": {
        'n_neighbors': Integer(1, 50),
        'weights': Categorical(['uniform', 'distance']),
        'p': Integer(1, 2)  
    },
    # Neural network models
    "Neural Network": {
        'hidden_layer_sizes': Integer(50, 2000),
        'alpha': Real(1e-5, 1e-2, prior='log-uniform'),
        'learning_rate_init': Real(1e-4, 1e-2, prior='log-uniform'),
        'activation': Categorical(['identity', 'logistic', 'tanh', 'relu']),
        'solver': Categorical(['lbfgs', 'adam']),
        'batch_size': Integer(16, 256),
        'max_iter': Integer(200, 3000)
    },
}

all_metrics = []

kf = KFold(n_splits=5, shuffle=True, random_state=3407)

base_output_dir = r"E:/desktop/hydro_channel/explainability/single"

for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
    print(f"Starting fold {fold} of 5-folds...")

    fold_dir = os.path.join(base_output_dir, f"fold_{fold}")
    os.makedirs(fold_dir, exist_ok=True)

    best_hyperparameters_file = os.path.join(fold_dir, "best_hyperparameters_single.txt")
    with open(best_hyperparameters_file, "w") as f:
        f.write(f"Best Hyperparameters for each model in fold {fold}:\n")
        f.write("-----------------------------------\n")
    
    # 3. Division of the data set into training and test sets
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # 4. Data pre-processing
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)
    
    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

    if np.isnan(X_train).any() or np.isnan(y_train).any():
        print(f"NaN values found in training data on fold {fold}, removing NaNs.")
        X_train = X_train[~np.isnan(X_train).any(axis=1)]
        y_train = y_train[~np.isnan(y_train)]

    fold_metrics = []
    
    for name, model in models.items():
        print(f"Training and tuning {name} with Bayesian Optimization on fold {fold}...")
        
        # Bayesian optimization with BayesSearchCV
        if name in param_space:
            bayes_search = BayesSearchCV(
                estimator=model,
                search_spaces=param_space[name],
                scoring='neg_mean_squared_error',
                cv=5,
                n_iter=30,
                n_jobs=-1,
                verbose=0,
                random_state=3407
            )

            try:
                bayes_search.fit(X_train, y_train)
                best_model = bayes_search.best_estimator_
                best_params = bayes_search.best_params_
                print(f"Best parameters for {name} on fold {fold}: {best_params}")
            except Exception as e:
                print(f"Error during hyperparameter tuning of {name} on fold {fold}: {e}")

                best_model = model.fit(X_train, y_train)
                try:
                    best_params = best_model.get_params()
                except:
                    best_params = {}
        else: 
            best_model = model.fit(X_train, y_train)
            try:
                best_params = best_model.get_params()
            except:
                best_params = {}
        
        with open(best_hyperparameters_file, "a") as f:
            f.write(f"{name}:\n")
            f.write(f"{best_params}\n")
            f.write("\n")

        model_filename = os.path.join(fold_dir, f"{name}_model_checkpoint.pkl")
        joblib.dump(best_model, model_filename)
        print(f"Model checkpoint for {name} on fold {fold} saved in: {model_filename}")

        y_train_pred_scaled = best_model.predict(X_train)
        y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).ravel()

        y_test_pred_scaled = best_model.predict(X_test)
        y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).ravel()

        y_train_true = scaler_y.inverse_transform(y_train.reshape(-1, 1)).ravel()
        y_test_true = y_test.values  

        train_len = len(y_train_true)
        test_len = len(y_test_true)
    
        if train_len > test_len:
            predictions_df = pd.DataFrame({
                'Train True': y_train_true,
                'Train Prediction': y_train_pred,
                'Test True': list(y_test_true) + [None] * (train_len - test_len),
                'Test Prediction': list(y_test_pred) + [None] * (train_len - test_len)
            })
        else:
            predictions_df = pd.DataFrame({
                'Train True': list(y_train_true) + [None] * (test_len - train_len),
                'Train Prediction': list(y_train_pred) + [None] * (test_len - train_len),
                'Test True': y_test_true,
                'Test Prediction': y_test_pred
            })
    
        predictions_filename = os.path.join(fold_dir, f"{name}_predictions_single.csv")
        predictions_df.to_csv(predictions_filename, index=False)
        print(f"The results of {name} on fold {fold} have been saved in: {predictions_filename}")

        r2 = r2_score(y_test_true, y_test_pred)
        mae = mean_absolute_error(y_test_true, y_test_pred)
        mse = mean_squared_error(y_test_true, y_test_pred)
        rmse = np.sqrt(mse)

        fold_metrics.append({
            "Model": name,
            "Fold": fold,
            "R2": r2,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse
        })

        print(f"{name} R2 for test on fold {fold}: {r2}")
        print(f"{name} MAE for test on fold {fold}: {mae}")
        print(f"{name} MSE for test on fold {fold}: {mse}")
        print(f"{name} RMSE for test on fold {fold}: {rmse}")
        print("-" * 40)

    fold_metrics_df = pd.DataFrame(fold_metrics)
    metrics_filename = os.path.join(fold_dir, "model_performance_metrics_bayes_opt_single.csv")
    fold_metrics_df.to_csv(metrics_filename, index=False)
    print(f"Model performance metrics for fold {fold} saved in: {metrics_filename}")
    
    all_metrics.extend(fold_metrics)

metrics_df = pd.DataFrame(all_metrics)

summary_metrics = []
for name in models.keys():
    model_metrics = metrics_df[metrics_df['Model'] == name]
    mean_metrics = model_metrics.mean(numeric_only=True)
    std_metrics = model_metrics.std(numeric_only=True)
    
    summary_metrics.append({
        'Model': name,
        'Fold': 'Mean',
        'R2': mean_metrics['R2'],
        'MAE': mean_metrics['MAE'],
        'MSE': mean_metrics['MSE'],
        'RMSE': mean_metrics['RMSE']
    })

    summary_metrics.append({
        'Model': name,
        'Fold': 'Std',
        'R2': std_metrics['R2'],
        'MAE': std_metrics['MAE'],
        'MSE': std_metrics['MSE'],
        'RMSE': std_metrics['RMSE']
    })

summary_df = pd.DataFrame(summary_metrics)
final_metrics_df = pd.concat([metrics_df, summary_df], ignore_index=True)

print(final_metrics_df)

final_metrics_df.to_csv(os.path.join(base_output_dir, "model_performance_metrics_bayes_opt_single.csv"), index=False)
