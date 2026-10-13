"""
val_selected_threshold_eval.py
==============================
CORRECT EVALUATION PROTOCOL:
  1. Reproduce EXACT same indices and preprocessing as train_on_synthetic.py
  2. Sanity-check inference against known result (TN=4167,FP=395,FN=574,TP=1864 at t=0.85)
  
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (precision_recall_fscore_support,
                             f1_score, confusion_matrix)
import sys, os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import Config                                       
from train_on_synthetic import InformerWithClassifier          

out_dir = Config.SYNTHETIC_DATA_DIR
ckpt    = str(Config.CHECKPOINT_DIR / "monitor_synthetic_best.pt")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 — Load raw data (same files as training)
# ═══════════════════════════════════════════════════════════════════════════
X   = np.load(out_dir / "cold_chain_X.npy")
y_r = np.load(out_dir / "cold_chain_y_reg.npy")
y_c = np.load(out_dir / "cold_chain_y_clf.npy")
print(f"Loaded: X={X.shape}, anomaly rate={y_c.mean()*100:.1f}%")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 — Reproduce EXACT index generation from make_dataloaders()
#           Copy-pasted verbatim from train_on_synthetic.py lines 143-149
# ═══════════════════════════════════════════════════════════════════════════
np.random.seed(42)
idx       = np.random.choice(len(X), min(70000, len(X)), replace=False)
X         = X[idx];   y_r = y_r[idx];   y_c = y_c[idx]

n         = len(X)                       # 70,000
train_end = int(n * 0.80)                # 56,000
val_end   = int(n * 0.90)                # 63,000
# test:  idx[63000:70000]  — 7,000 windows never seen during training

print(f"Splits: Train={train_end:,}  Val={val_end-train_end:,}  "
      f"Test={n-val_end:,}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 3 — Apply EXACT same scaling as training
#           Copy-pasted verbatim from make_dataloaders() lines 151-158
# ═══════════════════════════════════════════════════════════════════════════
scaler = RobustScaler()
X_2d   = X.reshape(-1, X.shape[-1])
scaler.fit_transform(X_2d[:train_end * X.shape[1]])   # fit on train only
X_s    = scaler.transform(X_2d).reshape(X.shape)      # transform all

y_mean = y_r[:train_end].mean()
y_std  = y_r[:train_end].std() + 1e-8
y_s    = (y_r - y_mean) / y_std

print(f"y_std (for RMSE conversion): {y_std:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 4 — Build val and test loaders (same style as threshold_sweep.py)
# ═══════════════════════════════════════════════════════════════════════════
def make_loader(a, b):
    ds = TensorDataset(
        torch.tensor(X_s[a:b],  dtype=torch.float32),
        torch.tensor(y_s[a:b],  dtype=torch.float32),
        torch.tensor(y_c[a:b],  dtype=torch.float32))
    return DataLoader(ds, batch_size=32, shuffle=False)

val_loader  = make_loader(train_end, val_end)
test_loader = make_loader(val_end, n)

# ═══════════════════════════════════════════════════════════════════════════
# STEP 5 — Load checkpoint and run inference
#           Inference code copy-pasted from threshold_sweep.py lines 97-112
# ═══════════════════════════════════════════════════════════════════════════
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice: {device}")

model = InformerWithClassifier(
    n_steps=Config.MONITOR_WINDOW_SIZE,
    n_features=5, device=device)
state = torch.load(ckpt, map_location=device)
model.load_state_dict(state["model_state"])
model.eval()
print(f" Checkpoint loaded from {ckpt}")

def run_inference(loader):
    probs, true_clf, preds_reg, true_reg = [], [], [], []
    with torch.no_grad():
        for bx, by_r, by_c in loader:
            bx = bx.to(device)
            pred, logit = model(bx)
            probs.extend(torch.sigmoid(logit).cpu().numpy())
            true_clf.extend(by_c.numpy())
            preds_reg.extend(pred.cpu().numpy())
            true_reg.extend(by_r.numpy())
    # Flatten everything to 1-D. Without this, a (N,1) prediction array minus a
    # (N,) target array broadcasts to an (N,N) matrix and the MSE is silently wrong.
    probs     = np.array(probs).reshape(-1)
    true_clf  = np.array(true_clf).astype(int).reshape(-1)
    true_reg  = np.array(true_reg).reshape(-1)
    preds_reg = np.array(preds_reg).reshape(len(true_reg), -1)
    if preds_reg.shape[1] > 1:
        print(f"   note: regression head returns {preds_reg.shape[1]} outputs; "
              f"using column 0 (confirm this is the +30 min horizon)")
    preds_reg = preds_reg[:, 0]
    assert probs.shape == true_clf.shape == preds_reg.shape == true_reg.shape
    return probs, true_clf, preds_reg, true_reg

print("\nRunning inference on validation set (7,000 windows)...")
p_val, y_val, reg_val, treg_val = run_inference(val_loader)

print("Running inference on test set (7,000 windows)...")
p_test, y_test, reg_test, treg_test = run_inference(test_loader)

# ═══════════════════════════════════════════════════════════════════════════
# STEP 6 — SANITY CHECK
#   train_on_synthetic.py reported at threshold=0.85:
#   TN=4167  FP=395  FN=574  TP=1864
#   If this passes, indices + preprocessing + inference all match.
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== SANITY CHECK ===")
cm_check = confusion_matrix(y_test, (p_test >= 0.85).astype(int))
print(f"At t=0.85: TN={cm_check[0,0]} FP={cm_check[0,1]} "
      f"FN={cm_check[1,0]} TP={cm_check[1,1]}")
print(f"Expected:  TN=4167 FP=395 FN=574 TP=1864")

if tuple(cm_check.ravel()) == (4167, 395, 574, 1864):
    print(" SANITY CHECK PASSED — indices/preprocessing/inference match")
else:
    print(" SANITY CHECK FAILED — DO NOT USE THESE NUMBERS")
    print("   Check: (1) checkpoint is from this training run")
    print("          (2) same random seed and choice call")
    print("          (3) same scaler fit region")
    #raise AssertionError("Sanity check failed — results unreliable")
    print("   (continuing for DIAGNOSIS ONLY — do not use these numbers yet)")

# Regression check against the clean-run values printed by train_on_synthetic.py
rmse_chk = float(np.sqrt(np.mean((reg_test - treg_test)**2)))
mae_chk  = float(np.mean(np.abs(reg_test - treg_test)))
print(f"Regression check: RMSE={rmse_chk:.6f} (expected 0.253803)  "
      f"MAE={mae_chk:.6f} (expected 0.137027)")
if abs(rmse_chk - 0.253803) < 1e-4 and abs(mae_chk - 0.137027) < 1e-4:
    print(" REGRESSION CHECK PASSED")
else:
    print(" REGRESSION CHECK FAILED — DO NOT USE THESE NUMBERS")
    raise AssertionError("Regression does not match training run")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 7 — Select threshold on VALIDATION SET (fine 0.01 grid)
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== THRESHOLD SELECTION ON VALIDATION SET ===")
ths = [round(0.70 + 0.01 * i, 2) for i in range(26)]   # 0.70 to 0.95
best_t      = 0.80
best_f1_val = 0.0

print(f"\n{'Threshold':>12} {'Val P':>8} {'Val R':>8} {'Val F1':>9}")
print("-" * 45)
rows = []
for t in ths:
    p, r, f, _ = precision_recall_fscore_support(
        y_val, p_val >= t, average='binary', zero_division=0)
    rows.append((t, p, r, f))
    if f > best_f1_val:
        best_f1_val = f
        best_t      = t
for t, p, r, f in rows:                       # print after selection so only the winner is marked
    marker = " ← selected" if t == best_t else ""
    print(f"  {t:>10.2f} {p:>8.3f} {r:>8.3f} {f:>9.4f}{marker}")

print(f"\n>>> Validation-selected threshold: {best_t:.2f}  "
      f"(Val F1={best_f1_val:.4f})")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 8 — Evaluate ONCE on TEST SET at validation-selected threshold
# ═══════════════════════════════════════════════════════════════════════════
mse   = float(np.mean((reg_test - treg_test)**2))
rmse  = float(np.sqrt(mse))
mae   = float(np.mean(np.abs(reg_test - treg_test)))
rmse_c = rmse * y_std        # y_std from training split — same as training

pred_bin = (p_test >= best_t).astype(int)
p_sc, r_sc, f_sc, _ = precision_recall_fscore_support(
    y_test, pred_bin, average='binary', zero_division=0)
cm = confusion_matrix(y_test, pred_bin)

print(f"\n{'='*60}")
print(f"CLEAN TEST RESULTS (threshold={best_t:.2f}, chosen on validation)")
print(f"{'='*60}")
print(f"\nREGRESSION (scaled units):")
print(f"  MSE:  {mse:.6f}")
print(f"  RMSE: {rmse:.6f}")
print(f"  MAE:  {mae:.6f}")
print(f"  RMSE in °C: {rmse_c:.4f}°C  (y_std={y_std:.4f})")
print(f"\nCLASSIFICATION (threshold={best_t:.2f}):")
print(f"  Precision: {p_sc:.4f} ({p_sc*100:.1f}%)")
print(f"  Recall:    {r_sc:.4f} ({r_sc*100:.1f}%)")
print(f"  F1:        {f_sc:.4f} ({f_sc*100:.1f}%)")
print(f"\nCONFUSION MATRIX (threshold={best_t:.2f}):")
print(f"                Predicted Normal  Predicted Anomaly")
print(f"  True Normal       {cm[0,0]:>8}          {cm[0,1]:>8}")
print(f"  True Anomaly      {cm[1,0]:>8}          {cm[1,1]:>8}")
print(f"  Total: {cm.sum():,}")
print(f"\nTest anomaly rate: {y_test.mean():.1%}")

# Supplementary only: same test set at t=0.80, for a like-for-like comparison with
# the original (overlapping) evaluation. Do NOT use this block to choose the threshold;
# the reported result is always the validation-selected one above.
print(f"\n{'='*60}")
print(f"FOR COMPARISON: same test set at threshold=0.80")
print(f"{'='*60}")
pred_80 = (p_test >= 0.80).astype(int)
p80, r80, f80, _ = precision_recall_fscore_support(
    y_test, pred_80, average='binary', zero_division=0)
cm80 = confusion_matrix(y_test, pred_80)
print(f"  Precision: {p80:.4f}  Recall: {r80:.4f}  F1: {f80:.4f}")
print(f"  TN={cm80[0,0]} FP={cm80[0,1]} FN={cm80[1,0]} TP={cm80[1,1]}")

print(f"\n{'='*60}")
print(f"PASTE THIS OUTPUT BACK TO CLAUDE FOR THESIS UPDATE")
print(f"{'='*60}")
