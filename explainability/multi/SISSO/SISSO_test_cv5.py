import pandas as pd
import numpy as np

csv_path = r"D:\Desktop\hydro_channel\val_fold5.csv"

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

    c0 = 1.976007041
    c1 = 0.002813183725
    return c0 + c1 * row["AS"]

def formula_hp12(row):

    c0 = 2.085666198
    c1 = 0.0003709808892
    return c0 + c1 * (np.cbrt(row["AS"]) * row["AS"])

def formula_hp13(row):

    c0 = 2.085666198
    c1 = 0.0003709808892
    return c0 + c1 * (np.cbrt(row["AS"]) * row["AS"])

def formula_hp21(row):

    c0 = 2.001031695
    c1 = 0.00148052523
    c2 = 1.581600952
    
    d001 = row["AS"] + row["AL"]
    d002 = row["NB"] / row["AS"]
    return c0 + c1 * d001 + c2 * d002

def formula_hp22(row):

    c0 = 2.001031695
    c1 = 0.00148052523
    c2 = 1.581600952
    
    d001 = row["AS"] + row["AL"]
    d002 = row["NB"] / row["AS"]
    return c0 + c1 * d001 + c2 * d002

def formula_hp23(row):

    c0 = 2.388711139
    c1 = 0.0005752131423
    c2 = -0.4098056885

    d001 = np.exp(np.cbrt(row["AS"]))
    d002 = row["AS"] / ( row["AL"] * (row["NS"] + row["DD"]) )

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

print("Prediction complete, results saved to cv5_evaluation_results.csv:")
print(results_df)
