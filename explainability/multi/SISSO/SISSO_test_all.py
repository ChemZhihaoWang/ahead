import pandas as pd
import numpy as np

data_path = r"D:\Desktop\hydrogen_channel_0325\dataset\SISSO\multi\test_data.csv"
df = pd.read_csv(data_path)

df.drop(columns=df.columns[0], inplace=True)  
y_true = df[df.columns[0]].values           

features = df[df.columns[1:]].copy()
features.columns = ["AS","AL","NS","NB","DD"]  

def compute_metrics(y_true, y_pred):
 
    residuals = y_true - y_pred
    sse = np.sum(residuals**2)
    sst = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - sse / sst if sst != 0 else 0.0

    mse = np.mean(residuals**2)
    rmse = np.sqrt(mse)

    mae = np.mean(np.abs(residuals))
    
    return r2, mse, rmse, mae

c0 = 1.990900543
c1 = 0.0009474928791
c2 = -0.8340538087

def predict(row):
    d001 = 2.0*row["AS"] + row["AL"]
    d002 = row["NB"] / (row["AL"] - row["AS"])
    return c0 + c1*d001 + c2*d002

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
