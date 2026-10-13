"""Load and preprocess public datasets for cold-chain logistics."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path
from typing import Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class PublicDatasetLoader:
    """
    Loader for public cold-chain datasets.
    Supports: Euro Crop, Smart Logistics, Mendeley Cold Storage.
    """

    def __init__(self, config):
        self.config = config

    def load_euro_crop(self) -> pd.DataFrame:
        """
        Load Euro Crop Agricultural Logistics Dataset.
        Returns: DataFrame with all columns.
        """
        try:
            df = pd.read_csv(self.config.EURO_CROP_PATH)
            print(f"Loaded Euro Crop dataset: {len(df)} rows, {len(df.columns)} columns")
            return df
        except FileNotFoundError:
            print(f"File not found: {self.config.EURO_CROP_PATH}")
            print("Please download the dataset from Kaggle and place it in the raw data folder.")
            return pd.DataFrame()

    def load_smart_logistics(self) -> pd.DataFrame:
        """
        Load Smart Logistics Supply Chain Dataset.
        Returns: DataFrame with all columns.
        """
        try:
            df = pd.read_csv(self.config.SMART_LOGISTICS_PATH)
            print(f"Loaded Smart Logistics dataset: {len(df)} rows, {len(df.columns)} columns")
            return df
        except FileNotFoundError:
            print(f"File not found: {self.config.SMART_LOGISTICS_PATH}")
            print("Please download the dataset from Kaggle and place it in the raw data folder.")
            return pd.DataFrame()

    def load_mendeley(self) -> pd.DataFrame:
        """
        Load Mendeley Cold Storage Dataset.
        Returns: DataFrame with all columns.
        """
        try:
            df = pd.read_csv(self.config.MENDELEY_PATH)
            print(f"Loaded Mendeley dataset: {len(df)} rows, {len(df.columns)} columns")
            return df
        except FileNotFoundError:
            print(f"File not found: {self.config.MENDELEY_PATH}")
            print("Please download the dataset from Mendeley Data and place it in the raw data folder.")
            return pd.DataFrame()

    def preprocess_euro_crop_for_monitor(self, df: pd.DataFrame, window_size: int = 60,
                                         horizon: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess Euro Crop data for time-series monitoring.

        Args:
            df: DataFrame from load_euro_crop()
            window_size: Number of time steps per window.
            horizon: Steps ahead to predict.

        Returns:
            X: Input windows (n_samples, window_size, n_features)
            y: Target values (n_samples,)
        """
        feature_cols = self.config.EURO_CROP_FEATURES
        target_col = self.config.EURO_CROP_TARGET

        # Select features
        data = df[feature_cols].values

        # Scale
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)

        # Create windows
        X, y = [], []
        for i in range(len(data_scaled) - window_size - horizon):
            X.append(data_scaled[i:i + window_size])
            # Predict the first feature (temperature)
            y.append(data_scaled[i + window_size + horizon, 0])

        return np.array(X), np.array(y)

    def preprocess_mendeley_for_classifier(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess Mendeley data for classification.

        Args:
            df: DataFrame from load_mendeley()

        Returns:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,)
        """
        feature_cols = self.config.MENDELEY_FEATURES
        X = df[feature_cols].values

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Encode target: Good=0, Bad=1
        le = LabelEncoder()
        y = le.fit_transform(df[self.config.MENDELEY_TARGET])

        return X_scaled, y

    def preprocess_smart_for_classifier(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess Smart Logistics data for classification.

        Args:
            df: DataFrame from load_smart_logistics()

        Returns:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,)
        """
        # Encode Traffic_Status
        traffic_map = {'Low': 0, 'Medium': 1, 'High': 2}
        df['Traffic_Status'] = df['Traffic_Status'].map(traffic_map)

        feature_cols = self.config.SMART_LOGISTICS_FEATURES
        X = df[feature_cols].values

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Target
        y = df[self.config.SMART_LOGISTICS_TARGET].values

        return X_scaled, y

    def get_dataset_summary(self, dataset_name: str) -> dict:
        """Get summary statistics for a dataset."""
        loaders = {
            'euro_crop': self.load_euro_crop,
            'smart': self.load_smart_logistics,
            'mendeley': self.load_mendeley
        }

        df = loaders[dataset_name]()
        if df.empty:
            return {"error": "Dataset not found"}

        return {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "null_count": df.isnull().sum().sum(),
            "data_types": df.dtypes.to_dict()
        }