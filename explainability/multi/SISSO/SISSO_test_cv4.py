import pandas as pd
import numpy as np

csv_path = r"D:\Desktop\hydro_channel\val_fold4.csv"

df = pd.read_csv(csv_path)
df.drop(columns=df.columns[0], inplace=True)

y_true = df["CD"].values

features = df[["AS","AL","NS","NB","DD"]].copy()

def compute_rmse_mae(y_true, y_pred):

    residuals = y_true - y_pred
    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))
    return rmse, mae

def formula_hp11(row):

    c0 = 1.981735206
    c1 = 0.002802455003
    return c0 + c1 * row["AS"]

def formula_hp12(row):

    c0 = 2.053308208
    c1 = 0.0004524510001
    return c0 + c1 * (np.log(row["AS"]) * row["AS"])

def formula_hp13(row):

    c0 = 2.053308208
    c1 = 0.0004524510001
    return c0 + c1 * (np.log(row["AS"]) * row["AS"])

def formula_hp21(row):

    c0 = 1.975123978
    c1 = 0.001483001798
    c2 = 0.07483607124
    
    d001 = row["AS"] + row["AL"]
    d002 = np.cbrt(row["NB"])  
    
    return c0 + c1 * d001 + c2 * d002

def formula_hp22(row):

    c0 = 1.988349221
    c1 = 0.0009564462453
    c2 = -0.8145154460
    
    d001 = (row["AS"] + row["AL"]) + row["AS"] 
    d002 = row["NB"] / (row["AL"] - row["AS"])
    
    return c0 + c1 * d001 + c2 * d002

def formula_hp23(row):

    c0 = 2.442923556
    c1 = 0.0005373745780
    c2 = -15.60074326
    
    d001 = np.exp(np.cbrt(row["AS"]))
    d002 = (row["AL"]**(-1)) / np.cbrt(row["NS"]) 
    
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
