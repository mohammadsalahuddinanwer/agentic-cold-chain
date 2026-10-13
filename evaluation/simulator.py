"""Simulator for evaluating agent systems in cold-chain logistics."""
import random
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple


class ColdChainSimulator:
    """
    Simulates cold-chain logistics environment with configurable parameters.
    Follows van Lon (2017) methodology for dynamism, urgency, scale.
    """

    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)

    def generate_shipment(self, shipment_id: str, product_type: str = "vaccines") -> Dict:
        """Generate a single shipment with sensor trace and anomalies."""
        # Product specs
        specs = {
            "vaccines": {"t_min": 2.0, "t_max": 8.0, "duration": (2, 48)},
            "pharmaceuticals": {"t_min": 2.0, "t_max": 8.0, "duration": (2, 48)},
            "fresh_produce": {"t_min": 0.0, "t_max": 4.0, "duration": (4, 72)},
            "frozen": {"t_min": -18.0, "t_max": -15.0, "duration": (6, 96)},
            "dairy": {"t_min": 0.0, "t_max": 4.0, "duration": (3, 24)},
        }

        spec = specs.get(product_type, specs["vaccines"])
        duration_hours = random.uniform(*spec["duration"])
        start_time = datetime.now() - timedelta(hours=random.uniform(0, 72))
        end_time = start_time + timedelta(hours=duration_hours)

        # Generate sensor readings
        n_points = int(duration_hours * 60)  # 1 per minute
        timestamps = [start_time + timedelta(minutes=i) for i in range(n_points)]

        # Base temperature with sinusoidal variation
        t = np.linspace(0, duration_hours, n_points)
        base_temp = (spec["t_min"] + spec["t_max"]) / 2
        amplitude = (spec["t_max"] - spec["t_min"]) / 2
        temperature = base_temp + amplitude * np.sin(2 * np.pi * t / 24) + np.random.normal(0, 0.1, n_points)

        # Add some anomalies
        anomalies = []
        anomaly_prob = 0.05  # 5% chance of anomaly
        for i in range(n_points):
            if random.random() < anomaly_prob:
                duration = random.randint(5, 30)
                for j in range(i, min(i + duration, n_points)):
                    temperature[j] += random.uniform(1.0, 4.0)
                    anomalies.append({
                        "timestamp": timestamps[j],
                        "severity": random.uniform(0.3, 1.0),
                        "type": random.choice(["door_left_open", "compressor_failure", "extreme_weather"])
                    })

        readings = [
            {
                "timestamp": ts,
                "temperature_c": float(temp),
                #"humidity_pct": float(70 + 10 * np.sin(2 * np.pi * i / 1440) + random.normal(0, 5)),
                "humidity_pct": float(70 + 10 * np.sin(2 * np.pi * i / 1440) + np.random.normal(0, 5)),
                "vibration_mg": float(np.random.exponential(10)),
                "door_open": bool(random.random() < 0.02)
            }
            for i, (ts, temp) in enumerate(zip(timestamps, temperature))
        ]

        return {
            "shipment_id": shipment_id,
            "product_type": product_type,
            "start_time": start_time,
            "end_time": end_time,
            "sensor_readings": readings,
            "anomalies": anomalies
        }

    def generate_scenario(self, n_shipments: int = 10, dynamism: str = "medium",
                          urgency: str = "medium") -> List[Dict]:
        """
        Generate a test scenario with specified dynamism and urgency.

        Args:
            n_shipments: Number of shipments
            dynamism: 'low', 'medium', 'high'
            urgency: 'low', 'medium', 'high'
        """
        # Map parameters
        dynamism_map = {"low": 0.05, "medium": 0.15, "high": 0.30}
        urgency_map = {"low": 0.2, "medium": 0.5, "high": 0.8}

        dynamism_rate = dynamism_map.get(dynamism, 0.15)
        urgency_rate = urgency_map.get(urgency, 0.5)

        shipments = []
        for i in range(n_shipments):
            # Random product type with some weighting
            product_types = ["vaccines", "pharmaceuticals", "fresh_produce", "frozen", "dairy"]
            weights = [0.3, 0.2, 0.2, 0.15, 0.15]
            product_type = random.choices(product_types, weights=weights)[0]

            shipment = self.generate_shipment(f"SHIP_{i:04d}", product_type)

            # Modify based on dynamism
            if dynamism_rate > 0.2:
                # More anomalies for high dynamism
                shipment["sensor_readings"] = self._add_noise(shipment["sensor_readings"], dynamism_rate)

            # Modify based on urgency
            if urgency_rate > 0.6:
                # Shorten windows for urgent shipments
                shipment["end_time"] = shipment["start_time"] + timedelta(
                    hours=random.uniform(1, 12)
                )

            shipments.append(shipment)

        return shipments

    def _add_noise(self, readings: List[Dict], noise_level: float) -> List[Dict]:
        """Add noise to readings based on dynamism level."""
        noisy = []
        for r in readings:
            new_r = r.copy()
            new_r["temperature_c"] += random.uniform(-noise_level * 2, noise_level * 2)
            noisy.append(new_r)
        return noisy