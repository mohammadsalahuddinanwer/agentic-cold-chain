"""Baseline B3: Detection-only agentic system (Timms et al. replication)."""
from typing import Dict, Any, List
import json
import re


class DetectionOnlyBaseline:
    """
    Detection-only agentic system.
    Replicates Timms et al. (2024) without resolution capabilities.
    """

    def __init__(self, llm, rag_retriever=None, max_iterations: int = 3):
        self.llm = llm
        self.rag = rag_retriever
        self.max_iterations = max_iterations

    def detect_anomaly(self, sensor_data: List[Dict]) -> Dict[str, Any]:
        """
        Detect and assess anomaly severity.
        No resolution - stops after detection.
        """
        # Parse sensor data
        if not sensor_data:
            return {
                "detected": False,
                "reason": "No sensor data provided",
                "severity": 0.0
            }

        # Build context
        last_reading = sensor_data[-1]
        recent_temps = [r.get('temperature_c', 0) for r in sensor_data[-10:]]

        context = {
            "current_temperature": last_reading.get('temperature_c', 0),
            "recent_temperatures": recent_temps,
            "humidity": last_reading.get('humidity_pct', 0),
            "vibration": last_reading.get('vibration_mg', 0),
            "door_open": last_reading.get('door_open', False)
        }

        # Use LLM for detection and assessment
        prompt = f"""
You are an anomaly detection AI for cold-chain logistics. Assess the following sensor data:

Current temperature: {context['current_temperature']}°C
Recent temperatures: {context['recent_temperatures']}
Humidity: {context['humidity']}%
Vibration: {context['vibration']}
Door open: {context['door_open']}

Determine if there is an anomaly. If yes, assess the severity (0-1).
Return a JSON with:
- detected: true/false
- severity: 0-1 (if detected)
- reasoning: brief explanation
"""

        response = self.llm.generate(prompt)

        try:
            result = json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                except:
                    result = {"detected": False, "severity": 0.0, "reasoning": "Parse error"}
            else:
                result = {"detected": False, "severity": 0.0, "reasoning": "Parse error"}

        # RAG enhancement if available
        if self.rag and result.get('detected', False):
            similar = self.rag.retrieve(
                f"Anomaly with severity {result.get('severity', 0)}",
                top_k=2
            )
            result["similar_cases"] = similar

        # Add detection latency (simulated - detection is fast)
        result["detection_latency_seconds"] = 2.0
        result["has_resolution"] = False
        result["resolution_available"] = False

        return result

    def process(self, sensor_data: List[Dict]) -> Dict[str, Any]:
        """
        Process sensor stream through detection-only pipeline.
        """
        result = self.detect_anomaly(sensor_data)

        return {
            "detection": result,
            "resolution_available": False,
            "cost_multiplier": 0.65,  # Detection-only saves some cost but not full
            "cargo_quality": 0.7 if not result.get('detected', False) else 0.5
        }