"""
fit_monitor_scaler.py
=====================
Saves the scaling used when the Monitor was trained, so that inference applies
exactly the same transformation. 

Writes outputs/checkpoints/monitor_scaler.joblib containing
  scaler  : RobustScaler fitted on the TRAINING portion only
  y_mean  : mean of the regression target on the training portion
  y_std   : standard deviation of the same

The subsample and split are reproduced exactly as in train_on_synthetic.py
(np.random.seed(42); np.random.choice(len(X), 70000); first 80% for training),
which is the same reconstruction that val_selected_threshold_eval.py verified
against the training run.
"""
import numpy as np
import joblib
from sklearn.preprocessing import RobustScaler

from config import Config


def main():
    d = Config.SYNTHETIC_DATA_DIR
    X = np.load(d / "cold_chain_X.npy")
    y_r = np.load(d / "cold_chain_y_reg.npy")
    print(f"Loaded X={X.shape}")

    np.random.seed(42)
    idx = np.random.choice(len(X), min(70000, len(X)), replace=False)
    X, y_r = X[idx], y_r[idx]
    train_end = int(len(X) * 0.80)

    scaler = RobustScaler().fit(X.reshape(-1, X.shape[-1])[:train_end * X.shape[1]])
    y_mean = float(y_r[:train_end].mean())
    y_std = float(y_r[:train_end].std() + 1e-8)

    Config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = Config.CHECKPOINT_DIR / "monitor_scaler.joblib"
    joblib.dump({"scaler": scaler, "y_mean": y_mean, "y_std": y_std,
                 "n_features": int(X.shape[-1]), "window": int(X.shape[1])}, path)

    print(f"Fitted on {train_end:,} training windows")
    print(f"  y_mean = {y_mean:.4f}, y_std = {y_std:.4f}")
    print(f"  feature centres: {np.round(scaler.center_, 3)}")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
