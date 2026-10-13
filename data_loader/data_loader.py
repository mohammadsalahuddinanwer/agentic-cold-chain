"""Data loader for real cold‑chain datasets."""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path
import torch
from torch.utils.data import TensorDataset, DataLoader


class ColdChainDataLoader:
    """
    Loads and preprocesses the three real datasets:
    - Euro Crop Agricultural Logistics Dataset
    - Smart Logistics Supply Chain Dataset
    - Mendeley Cold Storage Dataset
    """

    def __init__(self, config):
        self.config = config

    def load_euro_crop(self, window_size=60, horizon=30, test_size=0.2, val_size=0.1):
        """
        Load Euro Crop dataset for time‑series forecasting.
        """
        import numpy as np
        df = pd.read_csv(self.config.EURO_CROP_PATH)
        feature_cols = self.config.EURO_CROP_FEATURES

        # Select features
        data_df = df[feature_cols].copy()

        # --- Step 1: Force all columns to numeric, coerce errors to NaN ---
        for col in data_df.columns:
            data_df[col] = pd.to_numeric(data_df[col], errors='coerce')

        # --- Step 2: Replace infinite values with NaN ---
        data_df = data_df.replace([np.inf, -np.inf], np.nan)

        # --- Step 3: Drop columns that are entirely NaN ---
        data_df = data_df.dropna(axis=1, how='all')

        # --- Step 4: Fill remaining NaN with column median ---
        for col in data_df.columns:
            median_val = data_df[col].median()
            if np.isnan(median_val):
                data_df = data_df.drop(columns=[col])
            else:
                data_df[col] = data_df[col].fillna(median_val)

        # --- Step 5: Drop any rows that still have NaN ---
        data_df = data_df.dropna()

        # --- Step 6: Clip extreme outliers ---
        for col in data_df.columns:
            q01 = np.nanpercentile(data_df[col].values, 1)
            q99 = np.nanpercentile(data_df[col].values, 99)
            if q01 != q99:
                data_df[col] = data_df[col].clip(lower=q01, upper=q99)

        # --- Step 7: Final check ---
        data = data_df.values.astype(np.float64)
        if not np.all(np.isfinite(data)):
            finite_mask = np.all(np.isfinite(data), axis=1)
            data = data[finite_mask]
            data_df = data_df[finite_mask]

        # --- Step 8: Scale ---
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        data_scaled = scaler.fit_transform(data)

        # --- DEBUG: Check for inf after scaling ---
        print(f"data_scaled shape: {data_scaled.shape}")
        for col_idx in range(data_scaled.shape[1]):
            col_data = data_scaled[:, col_idx]
            has_nan = np.isnan(col_data).any()
            has_inf = np.isinf(col_data).any()
            if has_nan or has_inf:
                print(f"❌ Column {col_idx}: NaN={has_nan}, Inf={has_inf}, min={col_data.min():.6f}, max={col_data.max():.6f}")
            else:
                print(f"✅ Column {col_idx}: min={col_data.min():.6f}, max={col_data.max():.6f}")

        # --- FIX: Replace inf with finite values ---
        if np.isinf(data_scaled).any():
            print("⚠️ Replacing inf with 0.0")
            data_scaled = np.nan_to_num(data_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        # --- Step 9: Create sliding windows ---
        X, y = [], []
        for i in range(len(data_scaled) - window_size - horizon):
            X.append(data_scaled[i:i + window_size])
            y.append(data_scaled[i + window_size + horizon, 0])

        X = np.array(X)
        y = np.array(y)

        # --- DEBUG: Check targets ---
        print(f"Target values - min: {y.min():.6f}, max: {y.max():.6f}, mean: {y.mean():.6f}")
        print(f"Any NaN in y? {np.isnan(y).any()}")
        print(f"Any inf in y? {np.isinf(y).any()}")

        # --- Step 10: Split ---
        from sklearn.model_selection import train_test_split
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.config.SEED
        )
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=self.config.SEED
        )

        return (
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32),
        )

    def load_mendeley(self, test_size=0.2, val_size=0.1):
        """
        Load Mendeley dataset for classification (Good/Bad).

        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test as torch tensors.
        """
        df = pd.read_csv(self.config.MENDELEY_PATH)

        # Features
        feature_cols = self.config.MENDELEY_FEATURES
        X = df[feature_cols].values

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Target: convert Good/Bad to binary (Bad=1)
        le = LabelEncoder()
        y = le.fit_transform(df[self.config.MENDELEY_TARGET])

        # Split into train/val/test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=self.config.SEED, stratify=y
        )

        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=self.config.SEED, stratify=y_temp
        )

        return (
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
            torch.tensor(y_val, dtype=torch.long),
            torch.tensor(y_test, dtype=torch.long),
        )

    def load_smart_logistics(self, test_size=0.2, val_size=0.1):
        """
        Load Smart Logistics dataset for delay prediction (binary).

        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test as torch tensors.
        """
        df = pd.read_csv(self.config.SMART_LOGISTICS_PATH)

        # Encode Traffic_Status
        traffic_map = {'Low': 0, 'Medium': 1, 'High': 2}
        df['Traffic_Status'] = df['Traffic_Status'].map(traffic_map)

        # Features
        feature_cols = self.config.SMART_LOGISTICS_FEATURES
        X = df[feature_cols].values

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Target: Logistics_Delay (binary 0/1)
        y = df[self.config.SMART_LOGISTICS_TARGET].values

        # Split into train/val/test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=self.config.SEED, stratify=y
        )

        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=self.config.SEED, stratify=y_temp
        )

        return (
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
            torch.tensor(y_val, dtype=torch.long),
            torch.tensor(y_test, dtype=torch.long),
        )

    def load_euro_crop_dataloaders(self, batch_size=64, **kwargs):
        """Load Euro Crop data as PyTorch DataLoaders."""
        X_train, X_val, X_test, y_train, y_val, y_test = self.load_euro_crop(**kwargs)

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)

        return train_loader, val_loader, test_loader

    def load_mendeley_dataloaders(self, batch_size=32, **kwargs):
        """Load Mendeley data as PyTorch DataLoaders."""
        X_train, X_val, X_test, y_train, y_val, y_test = self.load_mendeley(**kwargs)

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)

        return train_loader, val_loader, test_loader