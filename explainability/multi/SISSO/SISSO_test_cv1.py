import pandas as pd
import numpy as np

csv_path = r"D:\Desktop\hydrogen_channel_0325\dataset\SISSO\multi\val_fold1.csv"

df = pd.read_csv(csv_path)

df.drop(columns=df.columns[0], inplace=True) 
y_true = df["CD"].values
features = df[["AS","AL","NS","NB","DD"]].copy()

def compute_rmse_mae(y_true, y_pred):
    """返回 (RMSE, MAE)"""
    residuals = y_true - y_pred
    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))
    return rmse, mae


def formula_hp11(row):

    c0 = 1.982411855
    c1 = 0.002794548926
    return c0 + c1 * row["AS"]

def formula_hp12(row):

    c0 = 2.055104727
    c1 = 0.0004498097134
    return c0 + c1 * (np.log(row["AS"]) * row["AS"])

def formula_hp13(row):

    c0 = 2.055104727
    c1 = 0.0004498097134
    return c0 + c1 * (np.log(row["AS"]) * row["AS"])

def formula_hp21(row):

    c0 = 2.012565169
    c1 = 0.001463723811
    c2 = 1.564782172
    d001 = row["AS"] + row["AL"]
    d002 = row["NB"] / row["AS"]
    return c0 + c1 * d001 + c2 * d002

def formula_hp22(row):

    c0 = 1.997097283
    c1 = 0.0009369205633
    c2 = -0.8536990989
    d001 = (row["AS"] + row["AL"]) + row["AS"]  
    d002 = row["NB"] / (row["AL"] - row["AS"])
    return c0 + c1 * d001 + c2 * d002

def formula_hp23(row):

    c0 = 2.363244531
    c1 = 3.197026989e-5
    c2 = 12.0001015
    d001 = (row["AS"]**2) / np.log(row["AS"])
    d002 = (row["NB"] - row["NS"]) / (row["AL"] * row["NS"])
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
results_df.to_csv("cv1_evaluation_results.csv", index=False)

print("The prediction is complete and the results have been saved to cv1_evaluation_results.csv:")
print(results_df)
