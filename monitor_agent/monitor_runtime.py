"""
monitor_agent/monitor_runtime.py
================================
Runtime wrapper that makes the deployed Monitor behave exactly like the Monitor
that was evaluated in the thesis.

"""
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn

from config import Config
from .informer_model import Informer

# Feature order used to build the training windows
# (generate_cold_chain_training_data.py: temperature, humidity, vibration, door, CO2)
FEATURE_KEYS = [
    ("temperature_c", "temperature", "Storage_Temperature"),
    ("humidity_pct", "humidity", "Storage_Humidity"),
    ("vibration_mg", "vibration", "Vibration_Level"),
    ("door_open", "door", "door_status"),
    ("co2_ppm", "co2", "CO2"),
]
FEATURE_DEFAULTS = [5.0, 70.0, 8.0, 0.0, 410.0]


class InformerWithClassifier(nn.Module):
    """Identical architecture to train_on_synthetic.py (256-128-64-1 head)."""

    def __init__(self, n_steps, n_features, d_model=256, n_heads=8, n_layers=3,
                 d_ff=512, factor=5, dropout=0.1, device="cpu"):
        super().__init__()
        self.backbone = Informer(n_steps=n_steps, n_features=n_features, d_model=d_model,
                                 n_heads=n_heads, n_layers=n_layers, d_ff=d_ff,
                                 factor=factor, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1))
        self.to(device)

    def encode(self, x):
        enc = self.backbone.position_encoding(self.backbone.embedding(x))
        for layer in self.backbone.encoder:
            enc = layer(enc)
        return enc[:, -1, :]                      # context vector

    def forward(self, x):
        context = self.encode(x)
        temps = {h: self.backbone.prediction_heads[f"horizon_{h}"](context).squeeze(-1)
                 for h in (30, 60, 90)
                 if f"horizon_{h}" in self.backbone.prediction_heads}
        return temps, self.classifier(context).squeeze(-1)


class MonitorRuntime:
    """Drop-in replacement for AnomalyPredictor in the orchestration path."""

    def __init__(self, checkpoint: str = None, scaler_path: str = None,
                 device: str = None, n_features: int = 5):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = Path(checkpoint or (Config.CHECKPOINT_DIR / "monitor_synthetic_best.pt"))
        scl = Path(scaler_path or (Config.CHECKPOINT_DIR / "monitor_scaler.joblib"))
        if not scl.exists():
            raise FileNotFoundError(
                f"{scl} not found — run `python fit_monitor_scaler.py` first so that "
                "inference uses the same scaling as training.")
        bundle = joblib.load(scl)
        self.scaler, self.y_mean, self.y_std = bundle["scaler"], bundle["y_mean"], bundle["y_std"]

        self.model = InformerWithClassifier(Config.MONITOR_WINDOW_SIZE, n_features,
                                            device=self.device)
        state = torch.load(str(ckpt), map_location=self.device)
        self.model.load_state_dict(state.get("model_state", state))
        self.model.eval()

    # ── input handling ──────────────────────────────────────────────────
    @staticmethod
    def _row(reading: dict):
        out = []
        for keys, default in zip(FEATURE_KEYS, FEATURE_DEFAULTS):
            value = next((reading[k] for k in keys if k in reading), default)
            out.append(float(value))
        return out

    def _to_array(self, window):
        if isinstance(window, np.ndarray):
            arr = window.astype(np.float32)
        elif isinstance(window, list) and window and isinstance(window[0], dict):
            arr = np.array([self._row(r) for r in window], dtype=np.float32)
        else:
            raise TypeError(f"Unsupported window type: {type(window)}")
        if arr.ndim != 2 or arr.shape[1] != len(FEATURE_KEYS):
            raise ValueError(f"Expected (n_steps, {len(FEATURE_KEYS)}) window, got {arr.shape}")
        ws = Config.MONITOR_WINDOW_SIZE
        if len(arr) < ws:                                  # pad at the front if short
            arr = np.vstack([np.repeat(arr[:1], ws - len(arr), axis=0), arr])
        return arr[-ws:]                                   # most recent ws readings

    # ── inference ───────────────────────────────────────────────────────
    def predict_anomaly(self, window, threshold: float = None) -> dict:
        """Anomaly probability from the trained classifier, plus temperature
        forecasts. `threshold` is a PROBABILITY (default Config.MONITOR_PROB_THRESHOLD)."""
        threshold = threshold if threshold is not None else getattr(
            Config, "MONITOR_PROB_THRESHOLD", 0.80)
        arr = self._to_array(window)
        scaled = self.scaler.transform(arr).astype(np.float32)
        with torch.no_grad():
            x = torch.tensor(scaled, device=self.device).unsqueeze(0)
            temps, logit = self.model(x)
            prob = float(torch.sigmoid(logit).item())
        is_anomaly = prob >= threshold
        out = {}
        for h in (30, 60, 90):
            t = temps.get(h)
            out[h] = {
                "temperature": (float(t.item()) * self.y_std + self.y_mean
                                if t is not None else None),
                "anomaly_probability": prob,
                "is_anomaly": is_anomaly,
                "confidence": prob,          # the classifier probability, not a rescaled gap
            }
        return out
