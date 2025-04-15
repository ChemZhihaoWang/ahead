import pandas as pd
import numpy as np

data_path = r"D:\Desktop\hydro_channel\test_data.csv"
df = pd.read_csv(data_path)

df.drop(columns=df.columns[0], inplace=True)  
y_true = df[df.columns[0]].values            

features = df[["AS","NC","DD"]].copy()

def compute_metrics(y_true, y_pred):
    residuals = y_true - y_pred
    sse = np.sum(residuals**2)                       
    sst = np.sum((y_true - np.mean(y_true))**2)       

    r2 = 1 - sse / sst if sst != 0 else 0.0
    mse = np.mean(residuals**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(residuals))

    return r2, mse, rmse, mae

def scd(x):
    return 1 / (np.pi * (1 + x**2))

c0 = 2.328268082
c1 = -3.140679507
c2 = 0.0007868155505

def predict(row):
    d001 = scd(row["NC"]) * row["NC"]
    d002 = np.exp(np.cbrt(row["AS"]))
    return c0 + c1 * d001 + c2 * d002

y_pred = features.apply(predict, axis=1)

r2, mse, rmse, mae = compute_metrics(y_true, y_pred)

results = {
    "R2": [r2],
    "MSE": [mse],
    "RMSE": [rmse],
    "MAE": [mae]
}
results_df = pd.DataFrame(results)
results_df.to_csv("test_evaluation.csv", index=False)

print("The prediction is complete and the results have been saved to test_evaluation.csv:")
print(results_df)
