"""Synthetic data generator for cold‑chain shipments (LTNgen‑style)."""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

@dataclass
class Shipment:
    """Represents a single cold‑chain shipment."""
    shipment_id: str
    product_type: str  # "vaccines", "pharmaceuticals", "fresh_produce", "frozen"
    temperature_min: float  # °C
    temperature_max: float  # °C
    start_time: datetime
    end_time: datetime
    origin: Tuple[float, float]
    destination: Tuple[float, float]
    events: List[Dict] = field(default_factory=list)
    sensor_readings: List[Dict] = field(default_factory=list)

class ColdChainGenerator:
    """
    Generates synthetic cold‑chain shipments with realistic anomalies.
    Based on LTNgen methodology from LangTest.
    """

    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)

        # Product specifications: (t_min, t_max, prob_door, prob_comp, prob_gps)
        self.product_specs = {
            "vaccines": (2.0, 8.0, 0.015, 0.003, 0.008),
            "pharmaceuticals": (2.0, 8.0, 0.012, 0.002, 0.006),
            "fresh_produce": (0.0, 4.0, 0.020, 0.005, 0.010),
            "frozen": (-18.0, -15.0, 0.010, 0.002, 0.005),
            "dairy": (0.0, 4.0, 0.018, 0.004, 0.009),
        }

        # Anomaly types and probabilities (per hour)
        self.anomaly_types = {
            "door_left_open": 0.015,
            "compressor_failure": 0.003,
            "gps_blackout": 0.008,
            "extreme_weather": 0.020,
            "driver_error": 0.010,
            "power_outage": 0.005
        }

        # Location ranges (simplified: Europe, US)
        self.locations = [
            (52.5200, 13.4050),  # Berlin
            (40.7128, -74.0060), # New York
            (51.5074, -0.1278),  # London
            (48.8566, 2.3522),   # Paris
            (35.6895, 139.6917), # Tokyo
            (55.7558, 37.6173),  # Moscow
            (34.0522, -118.2437),# Los Angeles
            (37.7749, -122.4194),# San Francisco
            (41.9028, 12.4964),  # Rome
            (28.6139, 77.2090)   # New Delhi
        ]

    def generate_shipment(self, shipment_id: str, product_type: str = None) -> Shipment:
        """Generate a single shipment with complete timeline and anomalies."""
        if product_type is None:
            product_type = random.choice(list(self.product_specs.keys()))

        t_min, t_max, prob_door, prob_comp, prob_gps = self.product_specs[product_type]

        # Random duration between 2 and 48 hours
        duration_hours = random.uniform(2, 48)
        start_time = datetime.now() - timedelta(hours=random.uniform(0, 72))
        end_time = start_time + timedelta(hours=duration_hours)

        # Random origin and destination
        origin = random.choice(self.locations)
        destination = random.choice(self.locations)
        while destination == origin:
            destination = random.choice(self.locations)

        shipment = Shipment(
            shipment_id=shipment_id,
            product_type=product_type,
            temperature_min=t_min,
            temperature_max=t_max,
            start_time=start_time,
            end_time=end_time,
            origin=origin,
            destination=destination
        )

        # Generate sensor readings
        self._generate_sensor_readings(shipment)

        # Generate anomalies
        self._generate_anomalies(shipment)

        return shipment

    def _generate_sensor_readings(self, shipment: Shipment):
        """Generate temperature, humidity, vibration, door status at 1‑minute intervals."""
        duration_hours = (shipment.end_time - shipment.start_time).total_seconds() / 3600
        n_points = int(duration_hours * 60)  # 1 sample per minute

        timestamps = [
            shipment.start_time + timedelta(minutes=i)
            for i in range(n_points)
        ]

        # Base temperature (sinusoidal drift + noise)
        t = np.linspace(0, duration_hours, n_points)
        base_temp = (shipment.temperature_min + shipment.temperature_max) / 2
        amplitude = (shipment.temperature_max - shipment.temperature_min) / 2
        temperature = base_temp + amplitude * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 0.1, n_points)

        # Humidity (correlated with temperature)
        humidity = 70 + 10 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 5, n_points)
        humidity = np.clip(humidity, 20, 100)

        # Vibration (mostly low, spikes at events)
        vibration = np.random.exponential(10, n_points)

        # Door status (normally closed, occasional checks)
        door_open = np.zeros(n_points, dtype=int)
        door_change_times = np.random.choice(n_points, size=int(n_points * 0.02), replace=False)
        for change in door_change_times:
            door_open[change:change+5] = 1  # door open for 5 minutes

        for i, ts in enumerate(timestamps):
            shipment.sensor_readings.append({
                "timestamp": ts,
                "temperature_c": temperature[i],
                "humidity_pct": humidity[i],
                "vibration_mg": vibration[i],
                "door_open": bool(door_open[i])
            })

    def _generate_anomalies(self, shipment: Shipment):
        """Inject anomalies based on time‑varying probabilities."""
        n_readings = len(shipment.sensor_readings)
        duration_hours = (shipment.end_time - shipment.start_time).total_seconds() / 3600

        # Determine anomalies by type
        for anomaly_type, prob_per_hour in self.anomaly_types.items():
            expected_count = prob_per_hour * duration_hours
            if expected_count > np.random.exponential(2):
                # Occurrence time within shipment
                start_idx = random.randint(0, n_readings - 60)
                anomaly_duration = random.randint(10, 120)  # minutes
                end_idx = min(start_idx + anomaly_duration, n_readings)

                for idx in range(start_idx, end_idx):
                    reading = shipment.sensor_readings[idx]
                    anomaly_event = {
                        "type": anomaly_type,
                        "timestamp": reading["timestamp"],
                        "severity": random.uniform(0.3, 1.0),
                        "is_anomaly": True
                    }

                    # Modify sensor readings based on anomaly type
                    if anomaly_type == "door_left_open":
                        reading["door_open"] = True
                        reading["temperature_c"] += random.uniform(0.5, 2.0)
                    elif anomaly_type == "compressor_failure":
                        reading["temperature_c"] += random.uniform(2.0, 5.0)
                    elif anomaly_type == "extreme_weather":
                        reading["temperature_c"] += random.uniform(1.0, 3.0)
                        reading["humidity_pct"] += random.uniform(10, 30)
                    elif anomaly_type == "gps_blackout":
                        # Simulate GPS loss (handled separately)
                        pass
                    elif anomaly_type == "power_outage":
                        reading["temperature_c"] += random.uniform(1.0, 4.0)
                        reading["vibration_mg"] = 0

                    shipment.events.append(anomaly_event)

    def generate_dataset(self, n_shipments: int = 1000) -> List[Shipment]:
        """Generate a full dataset."""
        shipments = []
        for i in range(n_shipments):
            shipment = self.generate_shipment(f"SHIP_{i:04d}")
            shipments.append(shipment)
        return shipments

    def to_dataframe(self, shipments: List[Shipment]) -> pd.DataFrame:
        """Convert shipments to DataFrame for training."""
        records = []
        for shipment in shipments:
            for reading in shipment.sensor_readings:
                record = {
                    "shipment_id": shipment.shipment_id,
                    "product_type": shipment.product_type,
                    "temperature_c": reading["temperature_c"],
                    "humidity_pct": reading["humidity_pct"],
                    "vibration_mg": reading["vibration_mg"],
                    "door_open": int(reading["door_open"]),
                    "timestamp": reading["timestamp"],
                    "is_anomaly": any(e["timestamp"] == reading["timestamp"] for e in shipment.events)
                }
                records.append(record)
        return pd.DataFrame(records)