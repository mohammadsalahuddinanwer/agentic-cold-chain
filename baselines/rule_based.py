"""Baseline B1: Reactive rule‑based anomaly detection."""
from typing import List, Dict, Any

class RuleBasedBaseline:
    """
    Simple threshold‑based detection. No prediction, no automated resolution.
    Triggers an email alert when temperature exceeds threshold.
    """

    def __init__(self, threshold: float = 8.0):
        self.threshold = threshold

    def process(self, sensor_data: List[Dict]) -> Dict[str, Any]:
        anomalies = []
        for reading in sensor_data:
            if reading.get('temperature_c', 0) > self.threshold:
                anomalies.append({
                    "timestamp": reading.get('timestamp'),
                    "temperature": reading['temperature_c'],
                    "message": f"ALERT: Temperature {reading['temperature_c']}°C exceeds threshold {self.threshold}°C"
                })

        resolution_time = len(anomalies) * 300 if anomalies else 0

        return {
            "anomalies_detected": len(anomalies),
            "detection_latency": 0,
            "resolution_time_seconds": resolution_time,
            "alerts": anomalies,
            "cost_multiplier": 1.0
        }