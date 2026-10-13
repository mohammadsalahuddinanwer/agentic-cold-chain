"""
Step 1 of 2-Month Plan: Generate Labelled Synthetic Cold-Chain Training Data.

This script generates a large dataset of cold-chain shipment windows with:
  - Exact anomaly timestamps (ground truth labels)
  - Multiple anomaly types matching your RAG knowledge base
  - Calibrated to WHO pharmaceutical storage guidelines (2-8°C for vaccines)
  - Parameters from Loisel et al. (2021) — already cited in your thesis


"""
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import random

sys.path.insert(0, str(Path(__file__).parent))
from config import Config


# ================================================================
# ANOMALY PARAMETERS — from Loisel et al. (2021), WHO guidelines
# These are already cited in your thesis Appendix B
# ================================================================
ANOMALY_PARAMS = {
    "compressor_failure": {
        "probability_per_hour": 0.025,   # increased 5x for training coverage
        "temp_rise_rate": 0.5,           # °C per minute — rapid rise
        "duration_minutes": (20, 90),
        "severity_range": (0.6, 1.0),
    },
    "door_left_open": {
        "probability_per_hour": 0.070,   # increased for training coverage
        "temp_rise_rate": 0.15,          # °C per minute — gradual rise
        "duration_minutes": (10, 40),
        "severity_range": (0.3, 0.7),
    },
    "power_outage": {
        "probability_per_hour": 0.030,   # increased for training coverage
        "temp_rise_rate": 0.8,           # rapid — no cooling at all
        "duration_minutes": (15, 60),
        "severity_range": (0.7, 1.0),
    },
    "extreme_weather": {
        "probability_per_hour": 0.045,   # increased for training coverage
        "temp_rise_rate": 0.1,           # slow external heating
        "duration_minutes": (30, 120),
        "severity_range": (0.3, 0.6),
    },
    "sensor_malfunction": {
        "probability_per_hour": 0.025,   # increased for training coverage
        "temp_rise_rate": 0.0,           # not a real temp rise
        "duration_minutes": (5, 30),
        "severity_range": (0.2, 0.5),
        "spike_noise": 3.0,              # erratic readings
    },
    "driver_error": {
        "probability_per_hour": 0.030,   # human error — common in practice
        "temp_rise_rate": 0.3,           # moderate rise from wrong settings
        "duration_minutes": (15, 60),
        "severity_range": (0.4, 0.8),
    },
    "gps_blackout": {
        "probability_per_hour": 0.035,   # signal loss — no temperature effect
        "temp_rise_rate": 0.0,           # GPS blackout — no thermal anomaly
        "duration_minutes": (10, 45),
        "severity_range": (0.2, 0.5),
        "spike_noise": 0.0,
    },
}

# Product temperature specs — WHO guidelines
PRODUCT_SPECS = {
    "vaccines":        {"t_min": 2.0,   "t_max": 8.0,   "critical_breach": 8.0},
    "pharmaceuticals": {"t_min": 2.0,   "t_max": 8.0,   "critical_breach": 8.0},
    "fresh_produce":   {"t_min": 0.0,   "t_max": 4.0,   "critical_breach": 6.0},
    "frozen":          {"t_min": -18.0, "t_max": -15.0, "critical_breach": -12.0},
    "dairy":           {"t_min": 0.0,   "t_max": 4.0,   "critical_breach": 6.0},
}

PRODUCT_WEIGHTS = [0.35, 0.25, 0.20, 0.10, 0.10]  # vaccine-heavy


# ================================================================
# SHIPMENT GENERATOR
# ================================================================
class ColdChainDataGenerator:
    """
    Generates labelled cold-chain shipment time series.
    Each shipment produces many 60-step windows with exact anomaly labels.
    """

    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        self.window_size = Config.MONITOR_WINDOW_SIZE  # 60
        self.horizon = 30  # predict 30 minutes ahead

    def generate_shipment_series(self, shipment_id: str,
                                  product_type: str = "vaccines") -> dict:
        """
        Generate a single shipment as a continuous time series.
        Duration: 2-48 hours (Loisel et al., 2021)
        Sampling: 1 reading per minute (industrial standard)
        """
        spec = PRODUCT_SPECS[product_type]
        duration_hours = random.uniform(2, 48)
        n_points = int(duration_hours * 60)  # 1 per minute

        # ── Base temperature — sinusoidal + small noise (realistic cold chain)
        t = np.linspace(0, duration_hours, n_points)
        base_temp = (spec["t_min"] + spec["t_max"]) / 2
        amplitude = (spec["t_max"] - spec["t_min"]) / 4
        temperature = (base_temp
                       + amplitude * np.sin(2 * np.pi * t / 24)
                       + np.random.normal(0, 0.05, n_points))

        # ── Humidity: correlated with temperature
        humidity = 70 + 5 * np.sin(2 * np.pi * t / 24) + np.random.normal(0, 2, n_points)
        humidity = np.clip(humidity, 40, 95)

        # ── Vibration: log-normal, spikes during road events
        vibration = np.random.exponential(8, n_points)
        # Add road bump events
        for _ in range(int(duration_hours * 2)):
            bump_pos = random.randint(0, n_points - 1)
            vibration[bump_pos] = random.uniform(20, 80)

        # ── Door status: binary, occasional opens
        door_open = np.zeros(n_points)
        for _ in range(int(duration_hours * 0.5)):
            open_pos = random.randint(0, n_points - 10)
            duration_open = random.randint(1, 8)
            door_open[open_pos:open_pos + duration_open] = 1.0

        # ── CO2 proxy (correlated with door opens and occupancy)
        co2 = 400 + 50 * door_open + np.random.normal(0, 10, n_points)

        # ── Anomaly injection — exact labels
        anomaly_labels = np.zeros(n_points, dtype=int)
        anomaly_types = ["none"] * n_points
        anomaly_severity = np.zeros(n_points)

        current_minute = 0
        while current_minute < n_points:
            # Check each anomaly type
            for anomaly_type, params in ANOMALY_PARAMS.items():
                prob_per_minute = params["probability_per_hour"] / 60.0
                if random.random() < prob_per_minute:
                    # Anomaly occurs at this minute
                    duration = random.randint(*params["duration_minutes"])
                    severity = random.uniform(*params["severity_range"])
                    end_minute = min(current_minute + duration, n_points)

                    for j in range(current_minute, end_minute):
                        # Apply temperature rise
                        if anomaly_type == "sensor_malfunction":
                            # Erratic readings
                            temperature[j] += np.random.normal(
                                0, params.get("spike_noise", 2.0))
                        else:
                            elapsed = j - current_minute
                            temperature[j] += (params["temp_rise_rate"]
                                               * elapsed * severity)

                        anomaly_labels[j] = 1
                        anomaly_types[j] = anomaly_type
                        anomaly_severity[j] = severity

            current_minute += 1

        # Clip temperature to realistic bounds
        max_temp = spec["critical_breach"] + 5.0
        temperature = np.clip(temperature, spec["t_min"] - 2, max_temp)

        return {
            "shipment_id": shipment_id,
            "product_type": product_type,
            "n_points": n_points,
            "temperature": temperature,
            "humidity": humidity,
            "vibration": vibration,
            "door_open": door_open,
            "co2": co2,
            "anomaly_labels": anomaly_labels,
            "anomaly_types": anomaly_types,
            "anomaly_severity": anomaly_severity,
        }

    def extract_windows(self, shipment: dict) -> tuple:
        """
        Extract 60-step windows with labels from a shipment.

        For each window of 60 minutes:
          X: (60, 5) array of [temperature, humidity, vibration, door_open, co2]
          y_reg: temperature value at +30 minutes (regression target)
          y_clf: 1 if any anomaly occurs in the CURRENT 60-minute window (classification target)
          metadata: anomaly type and severity for the window
        """
        n = shipment["n_points"]
        ws = self.window_size  # 60
        hz = self.horizon      # 30

        if n < ws + hz:
            return [], [], [], []

        # Stack features: (n_points, 5)
        features = np.stack([
            shipment["temperature"],
            shipment["humidity"],
            shipment["vibration"],
            shipment["door_open"],
            shipment["co2"],
        ], axis=1).astype(np.float32)

        X_windows, y_reg, y_clf, metadata = [], [], [], []

        for i in range(0, n - ws - hz, 5):  # stride=5 for efficiency
            window = features[i:i + ws]       # (60, 5)
            future_temp = features[i + ws + hz - 1, 0]   # temp at +30 min
            # Label: any anomaly in the next 30 minutes after window end?
            future_anomaly = int(
                shipment["anomaly_labels"][i:i + ws].any())
            # Anomaly type for metadata
            future_types = [
                t for t in shipment["anomaly_types"][i:i + ws]
                if t != "none"
            ]
            anomaly_type_label = future_types[0] if future_types else "none"
            severity = float(
                shipment["anomaly_severity"][i:i + ws].max())

            X_windows.append(window)
            y_reg.append(future_temp)
            y_clf.append(future_anomaly)
            metadata.append({
                "shipment_id": shipment["shipment_id"],
                "product_type": shipment["product_type"],
                "window_start": i,
                "anomaly_type": anomaly_type_label,
                "severity": severity,
                "is_anomaly": future_anomaly,
            })

        return X_windows, y_reg, y_clf, metadata


# ================================================================
# MAIN GENERATION PIPELINE
# ================================================================
def generate_dataset(n_shipments: int = 1000, seed: int = 42):
    """
    Generate the full cold-chain training dataset.

    Args:
        n_shipments: Number of shipments to generate
                     1000 → ~60,000-80,000 windows (recommended)
                     500  → ~30,000-40,000 windows (quick test)
    """
    print(f"\n{'='*60}")
    print(f"Cold-Chain Synthetic Data Generator")
    print(f"Based on Loisel et al. (2021) & WHO guidelines")
    print(f"{'='*60}")
    print(f"Generating {n_shipments} shipments...")
    print(f"Window size: 60 min, Horizon: 30 min, Stride: 5 min")

    Config.create_dirs()
    generator = ColdChainDataGenerator(seed=seed)

    all_X, all_y_reg, all_y_clf, all_meta = [], [], [], []

    product_types = list(PRODUCT_SPECS.keys())

    for i in range(n_shipments):
        product_type = random.choices(product_types, weights=PRODUCT_WEIGHTS)[0]
        shipment = generator.generate_shipment_series(
            f"SHIP_{i:05d}", product_type)
        X, y_r, y_c, meta = generator.extract_windows(shipment)

        all_X.extend(X)
        all_y_reg.extend(y_r)
        all_y_clf.extend(y_c)
        all_meta.extend(meta)

        if (i + 1) % 100 == 0:
            print(f"  Generated {i+1}/{n_shipments} shipments "
                  f"({len(all_X):,} windows so far)...")

    # Convert to arrays
    X_arr    = np.array(all_X,    dtype=np.float32)   # (N, 60, 5)
    y_reg    = np.array(all_y_reg, dtype=np.float32)  # (N,)
    y_clf    = np.array(all_y_clf, dtype=np.int32)    # (N,) binary
    meta_df  = pd.DataFrame(all_meta)

    print(f"\n{'='*60}")
    print(f"Dataset Summary")
    print(f"{'='*60}")
    print(f"Total windows:    {len(X_arr):,}")
    print(f"Window shape:     {X_arr.shape}")
    print(f"Anomaly rate:     {y_clf.mean()*100:.1f}%  "
          f"({y_clf.sum():,} anomaly / {(~y_clf.astype(bool)).sum():,} normal)")
    print(f"\nAnomaly breakdown:")
    for atype in ANOMALY_PARAMS.keys():
        count = (meta_df["anomaly_type"] == atype).sum()
        pct   = count / len(meta_df) * 100
        print(f"  {atype:<25}: {count:>6,} windows ({pct:.1f}%)")
    print(f"  {'none (normal)':<25}: "
          f"{(meta_df['anomaly_type']=='none').sum():>6,} windows")

    print(f"\nTemperature stats:")
    print(f"  Min: {y_reg.min():.2f}°C")
    print(f"  Max: {y_reg.max():.2f}°C")
    print(f"  Mean: {y_reg.mean():.2f}°C")

    # Save
    out_dir = Config.SYNTHETIC_DATA_DIR
    np.save(out_dir / "cold_chain_X.npy",     X_arr)
    np.save(out_dir / "cold_chain_y_reg.npy", y_reg)
    np.save(out_dir / "cold_chain_y_clf.npy", y_clf)
    meta_df.to_csv(out_dir / "cold_chain_metadata.csv", index=False)

    print(f"\n{'='*60}")
    print(f"Saved to {out_dir}/")
    print(f"  cold_chain_X.npy        — {X_arr.nbytes/1e6:.1f} MB")
    print(f"  cold_chain_y_reg.npy    — regression targets")
    print(f"  cold_chain_y_clf.npy    — binary anomaly labels")
    print(f"  cold_chain_metadata.csv — anomaly types per window")
    print(f"{'='*60}")
    print(f"\n✅ Next step: run train_on_synthetic.py to retrain the Monitor Agent")

    return X_arr, y_reg, y_clf, meta_df


def main():
    parser = argparse.ArgumentParser(
        description="Generate labelled synthetic cold-chain training data")
    parser.add_argument("--n_shipments", type=int, default=1000,
                        help="Number of shipments (default: 1000 → ~70k windows)")
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    generate_dataset(n_shipments=args.n_shipments, seed=args.seed)


if __name__ == "__main__":
    main()
