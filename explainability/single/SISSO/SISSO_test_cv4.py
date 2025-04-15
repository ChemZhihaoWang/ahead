import pandas as pd
import numpy as np

csv_path = r"D:\Desktop\hydro_channel\val_fold4.csv"

df = pd.read_csv(csv_path)

df.drop(columns=df.columns[0], inplace=True)  

y_true = df["CD"].values

features = df[["AS","NC","DD"]].copy()

def compute_rmse_mae(y_true, y_pred):
    residuals = y_true - y_pred
    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))
    return rmse, mae

def scd(x):
    return 1 / (np.pi * (1 + x**2))

def formula_hp11(row):
    c0 = 1.788412445
    c1 = 0.239126459
    return c0 + c1 * np.log(row["NC"])

def formula_hp12(row):
    c0 = 1.721388953
    c1 = 0.2369889166
    return c0 + c1 * np.log(row["NC"] + row["DD"])

def formula_hp13(row):
    c0 = 1.721388953
    c1 = 0.2369889166
    return c0 + c1 * np.log(row["NC"] + row["DD"])

def formula_hp21(row):
    c0 = 1.848020924
    c1 = 0.193695528
    c2 = 0.0001939808774
    d001 = np.log(row["NC"])
    d002 = row["DD"]**3
    return c0 + c1 * d001 + c2 * d002

def formula_hp22(row):
    c0 = 2.332823343
    c1 = -3.190314902
    c2 = 0.0007747493653
    d001 = scd(row["NC"]) * row["NC"]
    d002 = np.exp(np.cbrt(row["AS"]))
    return c0 + c1 * d001 + c2 * d002

def formula_hp23(row):
    c0 = 2.232522241
    c1 = 8.567197542e-6
    c2 = -0.1391239232
    d001 = (row["AS"] - row["DD"])**2
    d002 = np.exp(-row["NC"]) / scd(row["NC"])
    return c0 + c1 * d001 + c2 * d002

models = [
    {"hyperparam": 11, "predict_func": formula_hp11},
    {"hyperparam": 12, "predict_func": formula_hp12},
    {"hyperparam": 13, "predict_func": formula_hp13},
    {"hyperparam": 21, "predict_func": formula_hp21},
    {"hyperparam": 22, "predict_func": formula_hp22},
    {"hyperparam": 23, "predict_func": formula_hp23},
]

results = []  

for model in models:
    hp = model["hyperparam"]
    predict_func = model["predict_func"]
    
    y_pred = features.apply(predict_func, axis=1)
    
    rmse, mae = compute_rmse_mae(y_true, y_pred)
    
    results.append({"hyperparam": hp, "RMSE": rmse, "MAE": mae})

results_df = pd.DataFrame(results)
results_df.to_csv("cv4_evaluation_results.csv", index=False)

print("The prediction is complete and the results have been saved to cv4_evaluation_results.csv:")
print(results_df)
