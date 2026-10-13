"""Generate RAG knowledge base for the Diagnosis Agent."""
import json
import random
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd


class KnowledgeBaseGenerator:
    """
    Generates synthetic knowledge base entries for RAG.
    Creates anomaly cases, resolution strategies, and operational logs.
    """

    def __init__(self, config):
        self.config = config
        self.knowledge_dir = config.KNOWLEDGE_BASE_DIR

        # Templates for generating synthetic data
        self.anomaly_types = [
            "door_left_open",
            "compressor_failure",
            "gps_blackout",
            "extreme_weather",
            "driver_error",
            "power_outage"
        ]

        self.anomaly_descriptions = {
            "door_left_open": [
                "Driver left the rear door open during a routine inspection.",
                "Door was not properly sealed after loading.",
                "Temperature rose due to door being left ajar.",
                "Door seal malfunction allowed warm air entry."
            ],
            "compressor_failure": [
                "Compressor malfunction detected; temperature rising rapidly.",
                "Refrigeration unit failed during transit.",
                "Compressor overheating; cooling capacity reduced.",
                "Mechanical failure in the cooling system."
            ],
            "gps_blackout": [
                "GPS signal lost in remote area.",
                "Satellite connectivity interrupted.",
                "GPS antenna malfunction.",
                "Signal blockage due to terrain."
            ],
            "extreme_weather": [
                "Extreme heat wave caused internal temperature rise.",
                "Severe storm affected cooling efficiency.",
                "Unusually high ambient temperature.",
                "Weather conditions outside normal operating range."
            ],
            "driver_error": [
                "Driver set incorrect temperature.",
                "Driver failed to close door properly.",
                "Operator error during loading.",
                "Miscommunication about temperature settings."
            ],
            "power_outage": [
                "Power supply interrupted.",
                "Battery failure in refrigeration unit.",
                "Electrical fault caused system shutdown.",
                "Generator failed during transit."
            ]
        }

        self.resolutions = {
            "door_left_open": [
                "Notify driver immediately. Verify door closed after 5 minutes.",
                "Dispatch maintenance to inspect door seal.",
                "Monitor temperature for 10 minutes; if no recovery, escalate.",
                "Send alert to warehouse for arrival inspection."
            ],
            "compressor_failure": [
                "Immediately dispatch backup reefer.",
                "Reroute to nearest maintenance facility.",
                "Alert receiver of potential delay.",
                "Activate emergency cooling protocol."
            ],
            "gps_blackout": [
                "Switch to manual tracking.",
                "Notify dispatcher with last known location.",
                "Continue on planned route until signal returns.",
                "Use alternative navigation methods."
            ],
            "extreme_weather": [
                "Adjust cooling settings to compensate.",
                "Monitor temperature every 15 minutes.",
                "Reroute to avoid severe weather.",
                "Notify receiver of potential delay."
            ],
            "driver_error": [
                "Contact driver immediately to correct setting.",
                "Verify door is properly closed.",
                "Monitor temperature recovery.",
                "If recovery fails, escalate to backup dispatch."
            ],
            "power_outage": [
                "Dispatch backup generator immediately.",
                "Send backup reefer to transfer cargo.",
                "Alert maintenance team.",
                "Notify receiver of critical delay."
            ]
        }

        self.locations = [
            "Berlin", "Frankfurt", "Munich", "Paris", "London",
            "Brussels", "Amsterdam", "Madrid", "Rome", "Vienna",
            "Zurich", "Prague", "Warsaw", "Copenhagen", "Stockholm",
            "Hamburg", "Cologne", "Dusseldorf", "Stuttgart", "Nuremberg"
        ]

    def generate_anomaly_cases(self, n_cases: int = 50) -> List[Dict]:
        """Generate synthetic anomaly case entries."""
        cases = []

        for i in range(n_cases):
            anomaly_type = random.choice(self.anomaly_types)
            severity = random.choice(["low", "medium", "high"])
            severity_score = {"low": 0.3, "medium": 0.6, "high": 0.9}[severity]

            description = random.choice(self.anomaly_descriptions[anomaly_type])
            resolution = random.choice(self.resolutions[anomaly_type])
            location = random.choice(self.locations)

            case = {
                "text": f"Anomaly case #{i+1}: {description} Temperature deviation: {severity} severity. "
                        f"Location: {location}. Resolution: {resolution}",
                "metadata": {
                    "type": anomaly_type,
                    "severity": severity,
                    "severity_score": severity_score,
                    "location": location,
                    "resolution_strategy": resolution,
                    "case_id": f"CASE_{i+1:04d}"
                }
            }
            cases.append(case)

        return cases

    def generate_resolution_strategies(self) -> List[Dict]:
        """Generate resolution strategy entries."""
        strategies = []

        for anomaly_type in self.anomaly_types:
            for severity in ["low", "medium", "high"]:
                resolution = random.choice(self.resolutions[anomaly_type])

                strategy = {
                    "text": f"If anomaly type is '{anomaly_type}' and severity is {severity}, "
                            f"strategy: {resolution}",
                    "metadata": {
                        "anomaly_type": anomaly_type,
                        "severity": severity,
                        "strategy": resolution,
                        "priority": {"low": 3, "medium": 2, "high": 1}[severity]
                    }
                }
                strategies.append(strategy)

        return strategies

    def generate_operational_logs(self, n_logs: int = 30) -> List[Dict]:
        """Generate synthetic operational log entries."""
        logs = []

        for i in range(n_logs):
            anomaly_type = random.choice(self.anomaly_types)
            location = random.choice(self.locations)
            driver_id = f"D{random.randint(100, 999)}"

            description = random.choice(self.anomaly_descriptions[anomaly_type])
            resolution = random.choice(self.resolutions[anomaly_type])

            log = {
                "text": f"Driver {driver_id} reported: {description}. "
                        f"Location: {location}. Action taken: {resolution}",
                "metadata": {
                    "type": anomaly_type,
                    "driver_id": driver_id,
                    "location": location,
                    "timestamp": f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}T{random.randint(8,18):02d}:{random.randint(0,59):02d}:00",
                    "resolution": resolution
                }
            }
            logs.append(log)

        return logs

    def generate_and_save(self):
        """Generate all knowledge base entries and save to JSON files."""
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)

        # Generate anomaly cases
        print("Generating anomaly cases...")
        cases = self.generate_anomaly_cases(50)
        with open(self.knowledge_dir / "anomaly_cases.json", 'w') as f:
            json.dump(cases, f, indent=2)

        # Generate resolution strategies
        print("Generating resolution strategies...")
        strategies = self.generate_resolution_strategies()
        with open(self.knowledge_dir / "resolution_strategies.json", 'w') as f:
            json.dump(strategies, f, indent=2)

        # Generate operational logs
        print("Generating operational logs...")
        logs = self.generate_operational_logs(30)
        with open(self.knowledge_dir / "operational_logs.json", 'w') as f:
            json.dump(logs, f, indent=2)

        print(f"Knowledge base saved to {self.knowledge_dir}")
        print(f"Generated: {len(cases)} cases, {len(strategies)} strategies, {len(logs)} logs")