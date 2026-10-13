"""Baseline B4: Multi‑agent system without LLMs (van Lon‑style)."""
import random
from typing import Dict, List, Any

class MASNoLLMBaseline:
    """
    Decentralised contract‑net multi‑agent system for logistics.
    Based on van Lon's DynCNET implementation.
    """

    def __init__(self, n_agents: int = 10, threshold: float = 1.0):
        self.n_agents = n_agents
        self.threshold = threshold  # <-- ADD THIS LINE
        self.agents = [{"id": i, "available": True} for i in range(n_agents)]

    def assign_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        bids = []
        for agent in self.agents:
            if agent["available"]:
                distance = random.uniform(0, 100)
                bid = distance * 0.5
                bids.append({"agent": agent, "bid": bid})

        if not bids:
            return {"assigned": False, "reason": "No available agents"}

        best = min(bids, key=lambda x: x["bid"])
        best["agent"]["available"] = False

        return {
            "assigned": True,
            "agent_id": best["agent"]["id"],
            "bid": best["bid"],
            "estimated_cost": best["bid"] * 10
        }

    def complete_task(self, agent_id: int):
        for agent in self.agents:
            if agent["id"] == agent_id:
                agent["available"] = True
                break

    def process_anomaly(self, anomaly: Dict[str, Any]) -> Dict[str, Any]:
        task = {
            "type": "reroute",
            "origin": anomaly.get("location", "unknown"),
            "severity": anomaly.get("severity", 0.5)
        }

        result = self.assign_task(task)

        return {
            "resolution": result,
            "resolution_latency": result.get("bid", 50) * 2,
            "cost": result.get("estimated_cost", 100),
            "cargo_safety": 0.7 if result["assigned"] else 0.3
        }
    
    def process(self, sensor_data: List[Dict]) -> Dict[str, Any]:
        """
        Process sensor data: detect anomaly and run MAS only if anomaly detected.
        """
        # Extract temperature readings
        temps = []
        for record in sensor_data:
            for key, value in record.items():
                if 'temperature' in key.lower():
                    temps.append(value)
                    break
            else:
                temps.append(list(record.values())[0])

        is_anomaly = any(t > self.threshold for t in temps)

        if not is_anomaly:
            return {
                "cost_multiplier": 0.0,
                "resolution_time_seconds": 0.0,
                "anomalies_detected": 0
            }

        severity = max(temps) / 10.0
        severity = min(1.0, severity)
        anomaly = {"location": "unknown", "severity": severity}
        result = self.process_anomaly(anomaly)

        return {
            "cost_multiplier": result.get("cost", 100) / 100,
            "resolution_time_seconds": result.get("resolution_latency", 0),
            "anomalies_detected": 1 if result.get("assigned", False) else 0
        }