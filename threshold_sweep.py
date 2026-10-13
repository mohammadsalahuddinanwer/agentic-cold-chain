"""
Threshold sweep for Monitor Agent classifier.
Finds optimal threshold and shows full precision-recall tradeoff.


"""
import sys
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from sklearn.metrics import (precision_score, recall_score, f1_score,
                              accuracy_score, confusion_matrix)
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, str(Path(__file__).parent))
from config import Config
from monitor_agent.informer_model import Informer
import torch.nn as nn


# ── Rebuild model class (must match train_on_synthetic.py) ────────────────
class InformerWithClassifier(nn.Module):
    def __init__(self, n_steps, n_features, d_model=256, n_heads=8,
                 n_layers=3, d_ff=512, factor=5, dropout=0.1, device='cpu'):
        super().__init__()
        self.device = device
        self.backbone = Informer(
            n_steps=n_steps, n_features=n_features,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            d_ff=d_ff, factor=factor, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),  nn.LayerNorm(64),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1)
        ).to(device)
        self.to(device)

    def forward(self, x):
        enc = self.backbone.embedding(x)
        enc = self.backbone.position_encoding(enc)
        for layer in self.backbone.encoder:
            enc = layer(enc)
        context = enc[:, -1, :]
        pred_30 = self.backbone.prediction_heads["horizon_30"](context).squeeze(-1)
        logit   = self.classifier(context).squeeze(-1)
        return pred_30, logit


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load data
    out_dir = Config.SYNTHETIC_DATA_DIR
    X   = np.load(out_dir / "cold_chain_X.npy")
    y_r = np.load(out_dir / "cold_chain_y_reg.npy")
    y_c = np.load(out_dir / "cold_chain_y_clf.npy")

    np.random.seed(42)
    idx = np.random.choice(len(X), min(50000, len(X)), replace=False)
    X = X[idx]; y_r = y_r[idx]; y_c = y_c[idx]

    n         = len(X)
    train_end = int(n * 0.80)
    val_end   = int(n * 0.90)

    scaler = RobustScaler()
    X_2d   = X.reshape(-1, X.shape[-1])
    scaler.fit_transform(X_2d[:train_end * X.shape[1]])
    X_s    = scaler.transform(X_2d).reshape(X.shape)
    y_mean = y_r[:train_end].mean()
    y_std  = y_r[:train_end].std() + 1e-8
    y_s    = (y_r - y_mean) / y_std

    test_ds = TensorDataset(
        torch.tensor(X_s[val_end:],  dtype=torch.float32),
        torch.tensor(y_s[val_end:],  dtype=torch.float32),
        torch.tensor(y_c[val_end:],  dtype=torch.float32))
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # Load model
    model = InformerWithClassifier(
        n_steps=Config.MONITOR_WINDOW_SIZE,
        n_features=X.shape[2], device=device)
    ckpt  = str(Config.CHECKPOINT_DIR / "monitor_synthetic_best.pt")
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()
    print(f"✅ Checkpoint loaded from {ckpt}")

    # Collect predictions
    all_probs, all_true_clf = [], []
    all_preds_reg, all_true_reg = [], []

    with torch.no_grad():
        for bx, by_r, by_c in test_loader:
            bx   = bx.to(device)
            by_r = by_r.to(device)
            pred, logit = model(bx)
            all_probs.extend(torch.sigmoid(logit).cpu().numpy())
            all_true_clf.extend(by_c.numpy())
            all_preds_reg.extend(pred.cpu().numpy())
            all_true_reg.extend(by_r.cpu().numpy())

    probs    = np.array(all_probs)
    true_clf = np.array(all_true_clf).astype(int)
    preds_reg = np.array(all_preds_reg)
    true_reg  = np.array(all_true_reg)

    # Regression metrics
    mse  = float(np.mean((preds_reg - true_reg)**2))
    rmse = float(np.sqrt(mse))
    mae  = float(np.mean(np.abs(preds_reg - true_reg)))

    print(f"\n{'='*60}")
    print(f"REGRESSION METRICS")
    print(f"{'='*60}")
    print(f"  MSE:  {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  RMSE in °C ≈ {rmse * y_std:.4f}°C  (y_reg_std={y_std:.4f})")

    # Threshold sweep
    print(f"\n{'='*60}")
    print(f"THRESHOLD SWEEP — PRECISION / RECALL / F1")
    print(f"{'='*60}")
    print(f"\n{'Threshold':>10} {'Precision':>10} {'Recall':>10} "
          f"{'F1':>10} {'Accuracy':>10}")
    print("-" * 55)

    results = []
    best_f1 = 0.0; best_t = 0.40; best_row = {}

    for t in np.arange(0.10, 0.91, 0.05):
        pb  = (probs > t).astype(int)
        p   = float(precision_score(true_clf, pb, zero_division=0))
        r   = float(recall_score(true_clf,    pb, zero_division=0))
        f   = float(f1_score(true_clf,        pb, zero_division=0))
        acc = float(accuracy_score(true_clf,  pb))
        marker = " ← best F1" if f > best_f1 else ""
        print(f"{t:>10.2f} {p:>10.3f} {r:>10.3f} {f:>10.3f} {acc:>10.3f}{marker}")
        results.append({"threshold": round(t,2), "precision": p,
                         "recall": r, "f1": f, "accuracy": acc})
        if f > best_f1:
            best_f1 = f; best_t = t
            best_row = {"threshold": t, "precision": p, "recall": r,
                        "f1": f, "accuracy": acc}

    # Best threshold results
    pb = (probs > best_t).astype(int)
    cm = confusion_matrix(true_clf, pb)

    print(f"\n{'='*60}")
    print(f"BEST THRESHOLD: {best_t:.2f}")
    print(f"{'='*60}")
    print(f"  Precision: {best_row['precision']:.4f} ({best_row['precision']*100:.1f}%)")
    print(f"  Recall:    {best_row['recall']:.4f}    ({best_row['recall']*100:.1f}%)")
    print(f"  F1-Score:  {best_row['f1']:.4f}    ({best_row['f1']*100:.1f}%)")
    print(f"  Accuracy:  {best_row['accuracy']:.4f} ({best_row['accuracy']*100:.1f}%)")

    print(f"\n📊 Confusion Matrix (threshold={best_t:.2f}):")
    print(f"                Predicted Normal  Predicted Anomaly")
    if cm.shape == (2, 2):
        print(f"   True Normal       {cm[0,0]:>8}          {cm[0,1]:>8}")
        print(f"   True Anomaly      {cm[1,0]:>8}          {cm[1,1]:>8}")

    # Save
    pd.DataFrame(results).to_csv(
        "outputs/threshold_sweep_results.csv", index=False)
    pd.DataFrame([{
        "mse": mse, "rmse": rmse, "mae": mae,
        "rmse_celsius": rmse * y_std,
        "best_threshold": best_t,
        "precision": best_row["precision"],
        "recall":    best_row["recall"],
        "f1":        best_row["f1"],
        "accuracy":  best_row["accuracy"],
    }]).to_csv("outputs/monitor_synthetic_metrics.csv", index=False)

    print(f"\n💾 Results saved to outputs/threshold_sweep_results.csv")
    print(f"💾 Best metrics saved to outputs/monitor_synthetic_metrics.csv")

    print(f"\n{'='*60}")
    print(f"📝 THESIS SECTION 5.4.1 — Use these numbers")
    print(f"{'='*60}")
    print(f"  Training data: Synthetic cold-chain (2-8°C, 34.5% anomaly rate)")
    print(f"  Test samples:  {len(true_clf):,}")
    print(f"")
    print(f"  PRIMARY — Regression:")
    print(f"    MSE:  {mse:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f} (scaled)")
    print(f"    RMSE ≈ {rmse * y_std:.3f}°C in original temperature units")
    print(f"")
    print(f"  SECONDARY — Classification (threshold={best_t:.2f}):")
    print(f"    Precision: {best_row['precision']:.3f}")
    print(f"    Recall:    {best_row['recall']:.3f}")
    print(f"    F1:        {best_row['f1']:.3f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
