import pandas as pd
import numpy as np

csv_path = r"D:\Desktop\hydrogen_channel_0325\dataset\SISSO\single\val_fold2.csv"

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
    c0 = 1.793006682
    c1 = 0.2362592449
    return c0 + c1 * np.log(row["NC"])

def formula_hp12(row):
    c0 = 1.730834524
    c1 = 0.2332407993
    return c0 + c1 * np.log(row["NC"] + row["DD"])

def formula_hp13(row):
    c0 = 1.909000091
    c1 = 0.03946668673
    return c0 + c1 * (np.log(row["AS"]) * np.log(row["NC"]))

def formula_hp21(row):
    c0 = 1.851685822
    c1 = 0.1928743363
    c2 = 0.0001917094502
    d001 = np.log(row["NC"])
    d002 = row["DD"]**3
    return c0 + c1 * d001 + c2 * d002

def formula_hp22(row):
    c0 = 2.326421417
    c1 = -3.081132164
    c2 = 0.00078282481
    d001 = scd(row["NC"]) * row["NC"]
    d002 = np.exp(np.cbrt(row["AS"]))
    return c0 + c1 * d001 + c2 * d002

def formula_hp23(row):
    c0 = 2.229782512
    c1 = 8.209388713e-6
    c2 = -0.1334207762
    d001 = (row["AS"]**3) / (row["AS"] + row["DD"])
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
results_df.to_csv("cv2_evaluation_results.csv", index=False)

print("The prediction is complete and the results have been saved to cv2_evaluation_results.csv:")
print(results_df)
