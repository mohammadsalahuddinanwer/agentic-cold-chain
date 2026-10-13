# Save as check_units.py in cold_chain_agent/
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

df = pd.read_csv("data/raw/euro_crop_agricultural_logistics.csv")
temp = pd.to_numeric(df["Storage_Temperature"], errors='coerce')
temp = temp.replace([np.inf, -np.inf], np.nan)
temp = temp.fillna(temp.median())
q01 = np.percentile(temp, 1)
q99 = np.percentile(temp, 99)
temp = temp.clip(q01, q99)

scaler = RobustScaler()
scaler.fit(temp.values.reshape(-1, 1))

median = scaler.center_[0]
iqr    = scaler.scale_[0]

print(f"Temperature median: {median:.4f} °C")
print(f"Temperature IQR:    {iqr:.4f} °C")
print(f"")
print(f"Conversion:")
print(f"  RMSE in °C: {17.57 * iqr:.2f} °C")
print(f"  MAE  in °C: {6.75  * iqr:.2f} °C")
print(f"  MSE  in °C²:{308.67 * (iqr**2):.2f} °C²")
print(f"")
print(f"What scaled threshold 4.059 means in °C:")
print(f"  4.059 × {iqr:.4f} + {median:.4f} = {4.059 * iqr + median:.2f} °C")