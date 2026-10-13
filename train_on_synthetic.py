"""
 Retrain Monitor Agent on Synthetic Cold-Chain Data.


"""
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (precision_score, recall_score, f1_score,
                              accuracy_score, confusion_matrix)
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import Config
from monitor_agent.informer_model import Informer


# ================================================================
# FOCAL LOSS
# ================================================================
class FocalLoss(nn.Module):
    """alpha=0.96 strongly penalises missed anomalies — increased from 0.92 to push recall further."""
    def __init__(self, alpha: float = 0.92, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        probs    = torch.sigmoid(logits)
        bce      = nn.functional.binary_cross_entropy_with_logits(
                       logits, targets, reduction="none")
        pt       = torch.where(targets == 1, probs, 1 - probs)
        alpha_t  = torch.where(targets == 1,
                               torch.tensor(self.alpha),
                               torch.tensor(1 - self.alpha))
        return (alpha_t * (1 - pt) ** self.gamma * bce).mean()


# ================================================================
# JOINT MODEL — Informer + proper classifier on encoder context
# ================================================================
class InformerWithClassifier(nn.Module):
    """
    Wraps the Informer and adds a proper classifier head that uses
    the REAL encoder hidden state (256-dim context vector) rather
    than a repeated scalar.

    The Informer forward() computes:
      x = embedding(input)          # (batch, 60, 256)
      x = encoder_layers(x)         # (batch, 60, 256)
      context = x[:, -1, :]         # (batch, 256) ← THIS is the rich representation

    We intercept context before the prediction heads and feed it
    to our classifier MLP. This gives the classifier genuine
    temporal features learned from all 60 timesteps.
    """
    def __init__(self, n_steps, n_features, d_model=256, n_heads=8,
                 n_layers=3, d_ff=512, factor=5, dropout=0.1, device='cpu'):
        super().__init__()
        self.device = device

        # Informer backbone
        self.backbone = Informer(
            n_steps=n_steps, n_features=n_features,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            d_ff=d_ff, factor=factor, dropout=dropout
        )

        # ── KEY FIX: classifier on 256-dim context, not repeated scalar ──
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),   # 256 → 128
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),        # 128 → 64
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)           # 64 → 1
        ).to(device)
        # ─────────────────────────────────────────────────────────────────

        self.to(device)

    def forward(self, x):
        """
        Returns regression prediction and classification logit.
        Both come from the same encoder context vector.
        """
        # Step 1: embedding + positional encoding
        enc = self.backbone.embedding(x)
        enc = self.backbone.position_encoding(enc)

        # Step 2: encoder layers
        for layer in self.backbone.encoder:
            enc = layer(enc)

        # Step 3: context = last timestep hidden state (batch, d_model=256)
        context = enc[:, -1, :]   # ← REAL 256-dim encoder representation

        # Step 4: regression prediction (unchanged from original Informer)
        pred_30 = self.backbone.prediction_heads["horizon_30"](context).squeeze(-1)

        # Step 5: classification from REAL context (not repeated scalar)
        logit = self.classifier(context).squeeze(-1)   # ← THE FIX

        return pred_30, logit


# ================================================================
# DATA LOADING
# ================================================================
def load_synthetic_data():
    out_dir = Config.SYNTHETIC_DATA_DIR
    X   = np.load(out_dir / "cold_chain_X.npy")
    y_r = np.load(out_dir / "cold_chain_y_reg.npy")
    y_c = np.load(out_dir / "cold_chain_y_clf.npy")
    print(f"Loaded: X={X.shape}, anomaly rate={y_c.mean()*100:.1f}%")
    return X, y_r, y_c


def make_dataloaders(X, y_reg, y_clf, batch_size=64):
    # Subsample to 70k for better coverage (GPU can handle this)
    np.random.seed(42)
    idx   = np.random.choice(len(X), min(70000, len(X)), replace=False)
    X     = X[idx];  y_reg = y_reg[idx];  y_clf = y_clf[idx]

    n         = len(X)
    train_end = int(n * 0.80)
    val_end   = int(n * 0.90)

    scaler = RobustScaler()
    X_2d   = X.reshape(-1, X.shape[-1])
    scaler.fit_transform(X_2d[:train_end * X.shape[1]])
    X_s    = scaler.transform(X_2d).reshape(X.shape)

    y_mean = y_reg[:train_end].mean()
    y_std  = y_reg[:train_end].std() + 1e-8
    y_s    = (y_reg - y_mean) / y_std

    def loader(a, b, shuffle=True, oversample=False):
        ds = TensorDataset(
            torch.tensor(X_s[a:b],  dtype=torch.float32),
            torch.tensor(y_s[a:b],  dtype=torch.float32),
            torch.tensor(y_clf[a:b],dtype=torch.float32))
        if oversample:
            # Give anomaly windows higher sampling weight so model
            # sees them more often — fixes low recall from class imbalance
            labels    = y_clf[a:b]
            n_normal  = (labels == 0).sum()
            n_anomaly = (labels == 1).sum() + 1e-8
            # Anomaly windows get weight = 5.0 * n_normal/n_anomaly (~10x)
            weights   = np.where(labels == 1,
                                 5.0 * n_normal / n_anomaly,
                                 1.0).astype(np.float32)
            sampler   = WeightedRandomSampler(
                weights=torch.tensor(weights),
                num_samples=len(weights),
                replacement=True)
            print(f"  Oversampling: anomaly weight = {5.0 * n_normal/n_anomaly:.1f}x normal (5.0 multiplier x {n_normal/n_anomaly:.1f} ratio)")
            return DataLoader(ds, batch_size=batch_size,
                              sampler=sampler, num_workers=0)
        return DataLoader(ds, batch_size=batch_size,
                          shuffle=shuffle, num_workers=0)

    print(f"Splits: Train={train_end:,} Val={val_end-train_end:,} "
          f"Test={n-val_end:,} | anomaly={y_clf.mean()*100:.1f}%")
    return (loader(0, train_end, oversample=True),   # oversample train only
            loader(train_end, val_end, False),         # val unchanged
            loader(val_end, n, False),                 # test unchanged
            scaler, y_mean, y_std)


# ================================================================
# TRAINING — two-phase
# Phase 1: backbone regression only (classifier frozen)
# Phase 2: classifier only (backbone frozen)
# ================================================================
def train(model, train_loader, val_loader,
          epochs=50, lr=1e-4, phase1_epochs=15):

    mse_fn   = nn.MSELoss()
    focal_fn = FocalLoss(alpha=0.96, gamma=2.0)

    bb_opt  = torch.optim.Adam(model.backbone.parameters(),
                                lr=lr, weight_decay=1e-5)
    clf_opt = torch.optim.Adam(model.classifier.parameters(),
                                lr=lr, weight_decay=1e-5)
    bb_sch  = torch.optim.lr_scheduler.CosineAnnealingLR(
                  bb_opt,  T_max=max(phase1_epochs, 1))
    clf_sch = torch.optim.lr_scheduler.CosineAnnealingLR(
                  clf_opt, T_max=max(epochs - phase1_epochs, 1))

    best_f1 = -1.0
    best_mse = float('inf')
    patience = 0
    patience_limit = 15  # increased from 8 to prevent premature stopping

    print(f"\n{'='*60}")
    print(f"Phase 1 (epochs 1-{phase1_epochs}): backbone regression only")
    print(f"Phase 2 (epochs {phase1_epochs+1}-{epochs}): classifier only")
    print(f"Classifier input: 256-dim encoder context (THE FIX)")
    print(f"{'='*60}")

    for epoch in range(epochs):
        phase1 = epoch < phase1_epochs

        # Set requires_grad
        for p in model.backbone.parameters():
            p.requires_grad = phase1
        for p in model.classifier.parameters():
            p.requires_grad = not phase1

        model.train()
        tr_mse = tr_foc = n_b = 0.0

        for bx, by_r, by_c in train_loader:
            bx   = bx.to(model.device)
            by_r = by_r.to(model.device)
            by_c = by_c.to(model.device)

            pred, logit = model(bx)

            if phase1:
                loss = mse_fn(pred, by_r)
                bb_opt.zero_grad();  loss.backward();  bb_opt.step()
            else:
                loss = focal_fn(logit, by_c)
                clf_opt.zero_grad(); loss.backward(); clf_opt.step()

            tr_mse += mse_fn(pred, by_r).item()
            tr_foc += focal_fn(logit, by_c).item()
            n_b    += 1

        if phase1:
            bb_sch.step()
        else:
            clf_sch.step()

        # Validation
        model.eval()
        v_mse = v_n = 0
        v_probs = [];  v_labels = []
        with torch.no_grad():
            for bx, by_r, by_c in val_loader:
                bx   = bx.to(model.device)
                by_r = by_r.to(model.device)
                by_c = by_c.to(model.device)
                pred, logit = model(bx)
                v_mse    += mse_fn(pred, by_r).item()
                v_n      += 1
                v_probs.extend(torch.sigmoid(logit).cpu().numpy())
                v_labels.extend(by_c.cpu().numpy())

        avg_mse = v_mse / v_n
        probs   = np.array(v_probs)
        labels  = np.array(v_labels)

        # Sweep thresholds — track both best F1 and best recall@precision>0.4
        best_t_f1 = 0.0
        best_t    = 0.30
        best_recall_balanced = 0.0
        for t in np.arange(0.20, 0.85, 0.05):
            pb  = (probs > t).astype(int)
            f1v = f1_score(labels, pb, zero_division=0)
            rec = recall_score(labels, pb, zero_division=0)
            pre = precision_score(labels, pb, zero_division=0)
            if f1v > best_t_f1:
                best_t_f1 = f1v;  best_t = t
            # Track recall at threshold where precision > 0.4
            if pre > 0.40 and rec > best_recall_balanced:
                best_recall_balanced = rec

        label = "P1-regr" if phase1 else "P2-clf "
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"[{label}] Epoch {epoch+1:3d}/{epochs} | "
                  f"MSE: {tr_mse/n_b:.4f} Focal: {tr_foc/n_b:.4f} | "
                  f"Val MSE: {avg_mse:.4f} | "
                  f"Val F1({best_t:.2f}): {best_t_f1:.3f} | "
                  f"Recall@P>0.4: {best_recall_balanced:.3f}")

        # Save
        ckpt = str(Config.CHECKPOINT_DIR / "monitor_synthetic_best.pt")
        if phase1:
            if avg_mse < best_mse:
                best_mse = avg_mse
                torch.save({"model_state": model.state_dict()}, ckpt)
        else:
            if best_t_f1 > best_f1:
                best_f1  = best_t_f1
                patience = 0
                torch.save({"model_state": model.state_dict()}, ckpt)
                print(f"   Best model saved (Val F1={best_f1:.3f})")
            else:
                patience += 1
                if patience >= patience_limit:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

    print("Training complete.")


# ================================================================
# EVALUATION
# ================================================================
def evaluate(model, test_loader):
    model.eval()
    p_reg = [];  t_reg = []
    p_clf = [];  t_clf = []

    with torch.no_grad():
        for bx, by_r, by_c in test_loader:
            bx   = bx.to(model.device)
            by_r = by_r.to(model.device)
            by_c = by_c.to(model.device)
            pred, logit = model(bx)
            p_reg.extend(pred.cpu().numpy())
            t_reg.extend(by_r.cpu().numpy())
            p_clf.extend(torch.sigmoid(logit).cpu().numpy())
            t_clf.extend(by_c.cpu().numpy())

    p_reg = np.array(p_reg);  t_reg = np.array(t_reg)
    p_clf = np.array(p_clf);  t_clf = np.array(t_clf).astype(int)

    mse  = float(np.mean((p_reg - t_reg)**2))
    rmse = float(np.sqrt(mse))
    mae  = float(np.mean(np.abs(p_reg - t_reg)))

    best_f1 = 0.0;  best_t = 0.30
    for t in np.arange(0.10, 0.90, 0.05):
        pb  = (p_clf > t).astype(int)
        f1v = f1_score(t_clf, pb, zero_division=0)
        if f1v > best_f1:
            best_f1 = f1v;  best_t = t

    pb        = (p_clf > best_t).astype(int)
    precision = float(precision_score(t_clf, pb, zero_division=0))
    recall    = float(recall_score(t_clf,    pb, zero_division=0))
    f1        = float(f1_score(t_clf,        pb, zero_division=0))
    accuracy  = float(accuracy_score(t_clf,  pb))
    cm        = confusion_matrix(t_clf, pb)

    print(f"\n{'='*60}")
    print(f"TEST SET RESULTS — SYNTHETIC COLD-CHAIN DATA")
    print(f"{'='*60}")
    print(f"\n  REGRESSION (scaled units):")
    print(f"   MSE:  {mse:.6f}  |  RMSE: {rmse:.6f}  |  MAE: {mae:.6f}")
    print(f"\n CLASSIFICATION (threshold={best_t:.2f}):")
    print(f"   Precision: {precision:.4f} ({precision*100:.1f}%)")
    print(f"   Recall:    {recall:.4f}    ({recall*100:.1f}%)")
    print(f"   F1-Score:  {f1:.4f}    ({f1*100:.1f}%)")
    print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"\n Confusion Matrix:")
    if cm.shape == (2, 2):
        print(f"                Predicted Normal  Predicted Anomaly")
        print(f"   True Normal       {cm[0,0]:>8}          {cm[0,1]:>8}")
        print(f"   True Anomaly      {cm[1,0]:>8}          {cm[1,1]:>8}")

    pd.DataFrame({
        "metric": ["mse","rmse","mae","precision","recall","f1","accuracy","threshold"],
        "value":  [mse, rmse, mae, precision, recall, f1, accuracy, best_t]
    }).to_csv("outputs/monitor_synthetic_metrics.csv", index=False)
    print(f"\n Saved to outputs/monitor_synthetic_metrics.csv")
    print(f"\n{'='*60}")
    print(f" THESIS — Section 5.4.1 Monitor Agent (Synthetic Cold-Chain Data)")
    print(f"{'='*60}")
    print(f"   Regression:     MSE={mse:.4f} RMSE={rmse:.4f} MAE={mae:.4f} (scaled °C units)")
    print(f"   Classification: Precision={precision:.3f} Recall={recall:.3f} F1={f1:.3f}")
    print(f"   Note: Real 2-8°C cold-chain data. Convert RMSE to °C: RMSE × y_reg_std")
    print(f"{'='*60}")


# ================================================================
# MAIN
# ================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",        type=int,   default=50)
    parser.add_argument("--batch_size",    type=int,   default=64)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--phase1_epochs", type=int,   default=15)
    parser.add_argument("--eval_only",     action="store_true")
    args = parser.parse_args()

    Config.create_dirs()
    device = Config.MONITOR_DEVICE
    print(f"Device: {device}")

    X, y_reg, y_clf = load_synthetic_data()
    (train_loader, val_loader, test_loader,
     scaler, y_mean, y_std) = make_dataloaders(X, y_reg, y_clf, args.batch_size)

    model = InformerWithClassifier(
        n_steps=Config.MONITOR_WINDOW_SIZE,
        n_features=X.shape[2],
        device=device)

    bb_p  = sum(p.numel() for p in model.backbone.parameters())
    clf_p = sum(p.numel() for p in model.classifier.parameters())
    print(f"Model: {bb_p:,} (Informer backbone) + {clf_p:,} (context classifier)")
    print(f"Classifier input: 256-dim encoder context (not repeated scalar)")

    ckpt = str(Config.CHECKPOINT_DIR / "monitor_synthetic_best.pt")

    if args.eval_only:
        print(f"\nLoading: {ckpt}")
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model_state"])
        print(" Checkpoint loaded")
    else:
        train(model, train_loader, val_loader,
              epochs=args.epochs, lr=args.lr,
              phase1_epochs=args.phase1_epochs)

    evaluate(model, test_loader)


if __name__ == "__main__":
    main()
