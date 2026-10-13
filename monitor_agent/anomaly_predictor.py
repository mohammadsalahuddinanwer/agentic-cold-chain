"""Wrapper for training and inference with Informer model."""
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, List
import warnings

from .informer_model import Informer

class AnomalyPredictor:
    """
    Handles training, prediction, and evaluation of the Informer monitor.
    """
    def __init__(self, n_steps: int, n_features: int, device: str = "cuda", **kwargs):
        self.n_steps = n_steps
        self.n_features = n_features
        self.device = device if torch.cuda.is_available() else "cpu"

        self.model = Informer(
            n_steps=n_steps,
            n_features=n_features,
            d_model=128,      # half the size
            n_heads=4,        # half the heads
            n_layers=2,       # one fewer layer
            d_ff=256,         # half the feedforward
            factor=5,
            dropout=0.1
        ).to(self.device)

        self.scaler = StandardScaler()

    def prepare_data(self, df, target_col="temperature_c"):
        """Prepare time‑series windows for training."""
        feature_cols = [col for col in df.columns if col != "timestamp" and not df[col].dtype == object]
        X_scaled = self.scaler.fit_transform(df[feature_cols].values)

        X, y = [], []
        for i in range(len(X_scaled) - self.n_steps - 90):
            X.append(X_scaled[i:i + self.n_steps])
            y.append(X_scaled[i + self.n_steps + 30, feature_cols.index(target_col)])

        return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32)

    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 100, lr: float = 1e-5):
        """Train the model."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0

            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_x)
                pred = outputs["predictions"][30]
                loss = criterion(pred, batch_y)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_x)
                    pred = outputs["predictions"][30]
                    val_loss += criterion(pred, batch_y).item()

            avg_val_loss = val_loss / len(val_loader)
            
            # Print every epoch (since you set (epoch+1) % 1 == 0)
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.6f}, Val Loss: {avg_val_loss:.6f}")
            
            # Also print every 10 epochs with 4 decimal places
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {avg_val_loss:.4f}")

            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), "outputs/checkpoints/monitor_best.pt")
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    print("Early stopping triggered")
                    break

    def predict_anomaly(self, window, threshold: float = 1.0) -> dict:
        """
        Predict temperature at horizons and flag anomalies.
        
        Args:
            window: numpy array (n_steps, n_features) OR list of dicts
            threshold: temperature threshold for anomaly detection
        """
        import numpy as np
        
        # --- Handle list of dicts (from orchestrator) ---
        if isinstance(window, list) and len(window) > 0 and isinstance(window[0], dict):
            # Convert list of dicts to numpy array
            # Use the same feature order as training (from config)
            from config import Config
            feature_cols = Config.EURO_CROP_FEATURES
            # Extract values in correct order
            window = np.array([[d.get(col, 0.0) for col in feature_cols] for d in window])
        
        # If window is already a numpy array, just ensure it's float32
        elif isinstance(window, np.ndarray):
            window = window.astype(np.float32)
        else:
            raise TypeError(f"Unsupported window type: {type(window)}")
        
        # --- DEBUG: Print input stats ---
        #print(f"DEBUG: window shape: {window.shape}")
        #print(f"DEBUG: window min: {window.min():.6f}, max: {window.max():.6f}")
        #print(f"DEBUG: window contains NaN? {np.isnan(window).any()}")
        #print(f"DEBUG: window contains Inf? {np.isinf(window).any()}")
        # --- DEBUG: Print the first feature (temperature) values ---
        #print("DEBUG (predict_anomaly): Input window shape:", window.shape)
        #print("DEBUG (predict_anomaly): First 10 values of column 0:", window[:10, 0])
        #print("DEBUG (predict_anomaly): Max of column 0:", window[:, 0].max())
        
        self.model.eval()
        with torch.no_grad():
            window_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(self.device)
            outputs = self.model(window_tensor)

            predictions = {}
            for h in [30, 60, 90]:
                pred_temp = outputs["predictions"][h].item()
                predictions[h] = {
                    "temperature": pred_temp,
                    "is_anomaly": pred_temp > threshold,
                    "confidence": min(1.0, max(0.0, (pred_temp - threshold + 1) / (threshold + 1)))
                }
                print(f"DEBUG (predict_anomaly): Prediction for {h} min: {pred_temp:.4f}")

        return predictions

    def load_checkpoint(self, path: str):
        """Load a saved model checkpoint."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save(self.model.state_dict(), path)