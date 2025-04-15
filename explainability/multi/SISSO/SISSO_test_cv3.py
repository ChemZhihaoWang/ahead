import pandas as pd
import numpy as np

csv_path = r"D:\Desktop\hydro_channel\val_fold3.csv"

df = pd.read_csv(csv_path)
df.drop(columns=df.columns[0], inplace=True)

y_true = df["CD"].values

features = df[["AS","AL","NS","NB","DD"]].copy()

def compute_rmse_mae(y_true, y_pred):
    """Return (RMSE, MAE)"""
    residuals = y_true - y_pred
    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))
    return rmse, mae

def formula_hp11(row):

    c0 = 1.973458101
    c1 = 0.002814465802
    return c0 + c1 * row["AS"]

def formula_hp12(row):

    c0 = 2.085020543
    c1 = 0.0003695485526
    return c0 + c1 * (np.cbrt(row["AS"]) * row["AS"])

def formula_hp13(row):

    c0 = 2.085020543
    c1 = 0.0003695485526
    return c0 + c1 * (np.cbrt(row["AS"]) * row["AS"])

def formula_hp21(row):

    c0 = 2.002382683
    c1 = 0.001471125444
    c2 = 1.624288246
    d001 = row["AS"] + row["AL"]
    d002 = row["NB"] / row["AS"]
    return c0 + c1 * d001 + c2 * d002

def formula_hp22(row):

    c0 = 3.320523992
    c1 = 0.04863304157
    c2 = -1.694713779
    d001 = np.sqrt(row["AS"] + row["AL"])
    d002 = np.exp(-(row["NB"] / row["AS"]))
    return c0 + c1 * d001 + c2 * d002

def formula_hp23(row):

    c0 = 1.871526274
    c1 = 0.0009131349999
    c2 = 3.520941722
    d001 = (row["AS"] + row["AL"]) / np.log(row["AL"])
    d002 = (row["AL"] * row["NB"]) / (row["AS"]**2)
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
results_df.to_csv("cv3_evaluation_results.csv", index=False)

print("The prediction is complete and the results have been saved to cv3_evaluation_results.csv:")
print(results_df)
