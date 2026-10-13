"""Systematic evaluation with real Euro Crop data and controlled anomalies."""
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler


# --- Mock LLM for fast evaluation (no real model inference) ---
class MockLLM:
    def generate(self, prompt, **kwargs):
        return '{"diagnosis": {"anomaly_type": "compressor_failure", "severity": 0.8, "root_cause": "Simulated anomaly", "recommended_action": "dispatch_backup"}, "confidence": 0.9}'


class ExperimentRunner:
    """
    Runs systematic experiments following van Lon (2017) methodology.
    Uses real Euro Crop data (scaled) and injects controlled anomalies.
    """

    def __init__(self, orchestrator, generator):
        self.orchestrator = orchestrator
        self.generator = generator
        self.results = []

    def _load_and_scale_euro_crop(self):
        """Load Euro Crop data and apply the exact same scaling as training."""
        from config import Config
        import pandas as pd
        import numpy as np

        feature_cols = Config.EURO_CROP_FEATURES
        df = pd.read_csv(Config.EURO_CROP_PATH)

        # 1. Clean (same as data_loader)
        data_df = df[feature_cols].copy()
        for col in data_df.columns:
            data_df[col] = pd.to_numeric(data_df[col], errors='coerce')
        data_df = data_df.replace([np.inf, -np.inf], np.nan)
        for col in data_df.columns:
            median_val = data_df[col].median()
            if np.isnan(median_val):
                data_df = data_df.drop(columns=[col])
            else:
                data_df[col] = data_df[col].fillna(median_val)
        data_df = data_df.dropna()
        for col in data_df.columns:
            q01 = np.nanpercentile(data_df[col].values, 1)
            q99 = np.nanpercentile(data_df[col].values, 99)
            if q01 != q99:
                data_df[col] = data_df[col].clip(lower=q01, upper=q99)

        # 2. Scale
        scaler = RobustScaler()
        data_scaled = scaler.fit_transform(data_df.values)
        return data_scaled, feature_cols

    def run_experiment(self, dynamism: str, urgency: str, scale: str, n_runs: int = None) -> List[Dict]:
        if n_runs is None:
            from config import Config
            n_runs = Config.EXPERIMENT_RUNS

        # Load and scale Euro Crop data once
        data_scaled, feature_cols = self._load_and_scale_euro_crop()

        window_size = 60
        total_windows = len(data_scaled) - window_size
        if total_windows < 10:
            raise ValueError("Not enough data for sliding windows")

        # Number of windows per run based on scale
        scale_map = {"small": 10, "medium": 30, "large": 50}
        n_windows_to_use = min(scale_map[scale], total_windows)

        metrics_list = []

        for run in tqdm(range(n_runs), desc=f"{dynamism}/{urgency}/{scale}"):
            run_metrics = {
                "detection_lead_time": [],
                "resolution_latency": [],
                "anomalies_detected": 0
            }

            # Randomly select windows for this run
            indices = np.random.choice(total_windows, n_windows_to_use, replace=False)

            for idx in indices:
                # Get the scaled window as a numpy array
                window_np = data_scaled[idx:idx + window_size].copy()

                # Inject anomaly based on dynamism
                prob_anomaly = {
                    "low": 0.5,
                    "medium": 0.7,
                    "high": 0.9
                }.get(dynamism, 0.5)

                anomaly_injected = False
                if np.random.random() < prob_anomaly:
                    anomaly_injected = True
                    spike_start = np.random.randint(10, 40)
                    spike_len = np.random.randint(5, 15)
                    spike_val = np.random.uniform(5.0, 15.0)
                    for j in range(spike_start, min(spike_start + spike_len, window_size)):
                        window_np[j, 0] = spike_val

                # Convert to list of dicts with proper feature names
                sensor_dicts = [
                    {col: float(window_np[i][j]) for j, col in enumerate(feature_cols)}
                    for i in range(window_size)
                ]

                # Run orchestrator
                result = self.orchestrator.run(
                    sensor_window=sensor_dicts,
                    logs=["Driver note: en route"],
                    context={"product_type": "vaccines", "severity": 0.5}
                )

                # --- IMPORTANT: Use integer keys (30, 60, 90) ---
                pred = result.get('monitor_prediction', {}).get(30, {})
                if pred.get('is_anomaly', False):
                    run_metrics["detection_lead_time"].append(pred.get('confidence', 30))
                    run_metrics["anomalies_detected"] += 1

                if result.get('resolution_plan'):
                    run_metrics["resolution_latency"].append(
                        result['resolution_plan'].get('estimated_time_seconds', 60)
                    )

            # Avoid empty lists
            if not run_metrics["detection_lead_time"]:
                run_metrics["detection_lead_time"] = [0]
            if not run_metrics["resolution_latency"]:
                run_metrics["resolution_latency"] = [0]

            metrics_list.append({
                "run": run,
                "dynamism": dynamism,
                "urgency": urgency,
                "scale": scale,
                "mean_detection_lead_time": np.mean(run_metrics["detection_lead_time"]),
                "mean_resolution_latency": np.mean(run_metrics["resolution_latency"]),
                "anomalies_detected": run_metrics["anomalies_detected"]
            })

        return metrics_list

    def run_full_grid(self) -> pd.DataFrame:
        from config import Config

        dynamism_levels = ["low", "medium", "high"]
        urgency_levels = ["low"]        # You can expand to ["low", "medium", "high"] if needed
        scale_levels = ["small"]        # You can expand to ["small", "medium", "large"] if needed

        all_results = []

        for d in dynamism_levels:
            for u in urgency_levels:
                for s in scale_levels:
                    results = self.run_experiment(d, u, s, n_runs=Config.EXPERIMENT_RUNS)
                    all_results.extend(results)

        return pd.DataFrame(all_results)

    def save_results(self, df: pd.DataFrame, path: str):
        df.to_csv(path, index=False)