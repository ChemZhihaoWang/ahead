import pandas as pd
import numpy as np

csv_path = r"D:\Desktop\hydrogen_channel_0325\dataset\SISSO\multi\val_fold2.csv"

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

    c0 = 1.988780285
    c1 = 0.00275388679
    return c0 + c1 * row["AS"]

def formula_hp12(row):

    c0 = 2.05852087
    c1 = 0.0004452388922
    return c0 + c1 * (np.log(row["AS"]) * row["AS"])

def formula_hp13(row):

    c0 = 2.05852087
    c1 = 0.0004452388922
    return c0 + c1 * (np.log(row["AS"]) * row["AS"])

def formula_hp21(row):

    c0 = 1.984122579
    c1 = 0.001463743153
    c2 = 0.07313167412

    d001 = row["AS"] + row["AL"]
    d002 = np.cbrt(row["NB"]) 

    return c0 + c1 * d001 + c2 * d002

def formula_hp22(row):

    c0 = 2.027828462
    c1 = 0.0003133702285
    c2 = 0.09981653262

    d001 = np.cbrt(row["AL"]) * row["AS"]
    d002 = np.cbrt(abs(row["NS"] - row["DD"]))

    return c0 + c1 * d001 + c2 * d002

def formula_hp23(row):

    c0 = 2.444080734
    c1 = 0.0005302547235
    c2 = -15.46285616

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
results_df.to_csv("cv2_evaluation_results.csv", index=False)

print("The prediction is complete and the results have been saved to cv2_evaluation_results.csv:")
print(results_df)
