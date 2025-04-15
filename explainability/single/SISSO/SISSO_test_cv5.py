import pandas as pd
import numpy as np

csv_path = r"D:\Desktop\hydro_channel\val_fold5.csv"

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
    c0 = 1.775226569
    c1 = 0.2445809037
    return c0 + c1 * np.log(row["NC"])

def formula_hp12(row):
    c0 = 1.713150117
    c1 = 0.2395368246
    return c0 + c1 * np.log(row["NC"] + row["DD"])

def formula_hp13(row):
    c0 = 1.713150117
    c1 = 0.2395368246
    return c0 + c1 * np.log(row["NC"] + row["DD"])

def formula_hp21(row):
    c0 = 2.292491261
    c1 = 7.458756891e-6
    c2 = -0.7201609232
    d001 = row["AS"]**2
    d002 = 1.0 / row["NC"]
    return c0 + c1 * d001 + c2 * d002

def formula_hp22(row):
    c0 = 2.349953178
    c1 = -3.285534246
    c2 = 5.839101141e-6
    d001 = scd(row["NC"]) * row["NC"]
    d002 = (row["AS"] + row["DD"])**2
    return c0 + c1 * d001 + c2 * d002

def formula_hp23(row):
    c0 = 2.226997039
    c1 = 7.968310807e-6
    c2 = -0.1361604876
    d001 = (row["AS"]**2) + (row["NC"] * row["DD"])
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
results_df.to_csv("cv5_evaluation_results.csv", index=False)

print("The prediction is complete and the results have been saved to cv5_evaluation_results.csv:")
print(results_df)
