# Save as check_synthetic_data.py in cold_chain_agent/
import numpy as np
import pandas as pd

X   = np.load("data/synthetic/cold_chain_X.npy")
y_r = np.load("data/synthetic/cold_chain_y_reg.npy")
y_c = np.load("data/synthetic/cold_chain_y_clf.npy")
meta = pd.read_csv("data/synthetic/cold_chain_metadata.csv")

print(f"=== SYNTHETIC DATA SANITY CHECK ===")
print(f"\nShape:")
print(f"  X (windows):      {X.shape}   — (windows, timesteps, features)")
print(f"  y_reg (targets):  {y_r.shape}")
print(f"  y_clf (labels):   {y_c.shape}")

print(f"\nTemperature column (X[:,:,0]) — should be 2-8°C range:")
temp = X[:, :, 0]
print(f"  Min:  {temp.min():.2f}°C")
print(f"  Max:  {temp.max():.2f}°C")
print(f"  Mean: {temp.mean():.2f}°C")
print(f"  Std:  {temp.std():.2f}°C")

print(f"\nRegression targets (y_reg) — temp at +30 min:")
print(f"  Min:  {y_r.min():.2f}°C")
print(f"  Max:  {y_r.max():.2f}°C")
print(f"  Mean: {y_r.mean():.2f}°C")

print(f"\nAnomaly labels (y_clf):")
print(f"  Total windows:   {len(y_c):,}")
print(f"  Anomaly windows: {y_c.sum():,} ({y_c.mean()*100:.1f}%)")
print(f"  Normal windows:  {(~y_c.astype(bool)).sum():,} ({(1-y_c.mean())*100:.1f}%)")

print(f"\nAnomaly type distribution:")
print(meta['anomaly_type'].value_counts())

print(f"\n=== VERDICT ===")
if 1.0 <= temp.mean() <= 15.0:
    print(" Temperature range looks realistic for cold chain")
else:
    print("  Temperature range may be outside expected cold chain values")

if 5 <= y_c.mean()*100 <= 30:
    print(" Anomaly rate is reasonable (5-30%)")
else:
    print(f"  Anomaly rate {y_c.mean()*100:.1f}% — check generator parameters")