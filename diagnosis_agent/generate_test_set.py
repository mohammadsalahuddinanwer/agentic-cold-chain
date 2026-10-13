"""Generate a labelled test set for Diagnosis Agent evaluation."""
import json
import random
from pathlib import Path
from typing import List, Dict

def generate_test_set(n_cases: int = 100, output_path: str = "data/knowledge_base/diagnosis_test_set.json"):
    """
    Generate a labelled test set of anomaly descriptions with ground truth.
    
    Args:
        n_cases: Number of test cases to generate (default: 100)
        output_path: Where to save the JSON file
    """
    
    # Define anomaly types and their descriptions
    anomaly_templates = {
        "door_left_open": [
            "Temperature rose gradually by 1.5°C over 15 minutes while the driver was loading cargo.",
            "The rear door was left open during a routine inspection, causing a slow temperature rise.",
            "A door seal failure caused warm air to enter, raising temperature by 2°C over 20 minutes.",
            "Driver reported door ajar; temperature increased from 4°C to 6°C in 10 minutes.",
            "Loading dock delay: door remained open for 25 minutes, temperature rose steadily.",
            "Door left open during unloading, temperature rose from 4°C to 5.5°C in 12 minutes.",
            "Inspection team left door open; temperature rose by 1.8°C over 18 minutes.",
            "Door seal degraded, causing gradual temperature rise of 0.5°C per minute.",
            "Driver forgot to close door after checking load; temperature rose to 6.2°C.",
            "Warehouse staff left door open during inventory; temperature increased by 2.3°C.",
            "Door left ajar during temperature check; temperature rose from 4°C to 5.8°C.",
            "Door seal leaking; temperature rose from 4°C to 6.3°C over 22 minutes.",
            "Driver left door open for 8 minutes; temperature rose from 4°C to 5.2°C.",
            "Door not properly sealed after loading; temperature rose from 4°C to 6.0°C."
        ],
        "compressor_failure": [
            "Rapid temperature rise from 4°C to 8°C in under 8 minutes; refrigeration unit stopped.",
            "Compressor failed; temperature spiked by 3.5°C in 6 minutes.",
            "Cooling unit malfunction: temperature rose from 4°C to 9°C in 5 minutes.",
            "Refrigerant leak detected; temperature climbed from 4°C to 7.5°C in 10 minutes.",
            "Compressor overheating; temperature increased from 4°C to 8.2°C in 7 minutes.",
            "Refrigeration unit shut down; temperature rose from 4°C to 10°C in 12 minutes.",
            "Compressor fan failure; temperature rose from 4°C to 7°C in 9 minutes.",
            "Gas leak in cooling system; temperature rose from 4°C to 8.5°C in 11 minutes.",
            "Compressor belt snapped; temperature rose from 4°C to 9.5°C in 8 minutes.",
            "Cooling system failure; temperature rose from 4°C to 11°C in 15 minutes.",
            "Compressor seized; temperature rose from 4°C to 8.7°C in 9 minutes.",
            "Refrigerant pressure drop; temperature rose from 4°C to 7.8°C in 13 minutes.",
            "Compressor motor burned out; temperature rose from 4°C to 9.2°C in 10 minutes."
        ],
        "power_outage": [
            "Power failure caused refrigeration to stop; temperature rose from 4°C to 9°C in 15 minutes.",
            "Electrical outage: temperature rose from 4°C to 8.5°C in 12 minutes.",
            "Generator failed; temperature rose from 4°C to 10°C in 18 minutes.",
            "Power surge caused refrigeration unit to restart; temperature rose from 4°C to 7°C.",
            "Battery failure; temperature rose from 4°C to 8°C in 14 minutes.",
            "Power cut for 20 minutes; temperature rose from 4°C to 9.5°C.",
            "Backup generator failed to start; temperature rose from 4°C to 11°C.",
            "Electrical fault caused cooling system shutdown; temperature rose to 8.5°C.",
            "Power fluctuations caused intermittent cooling; temperature rose to 7.5°C.",
            "Main power outage; temperature rose from 4°C to 10.5°C in 22 minutes.",
            "Battery depleted; temperature rose from 4°C to 8.2°C in 16 minutes.",
            "Fuse blown; temperature rose from 4°C to 7.5°C in 10 minutes."
        ],
        "driver_error": [
            "Driver set temperature incorrectly to 0°C instead of 4°C; corrected after 30 minutes.",
            "Operator error: temperature was set to -5°C instead of 4°C.",
            "Driver failed to close door properly; temperature rose from 4°C to 6.5°C.",
            "Driver left door ajar; temperature rose from 4°C to 7°C in 15 minutes.",
            "Driver ignored temperature alarm; temperature rose from 4°C to 8°C.",
            "Driver misread thermostat and adjusted incorrectly; temperature rose to 5.5°C.",
            "Driver failed to report temperature issue; temperature rose to 6.8°C.",
            "Driver turned off cooling unit by mistake; temperature rose from 4°C to 9°C.",
            "Driver's error caused temperature fluctuation between 2°C and 6°C.",
            "Driver set defrost cycle incorrectly; temperature rose to 5.5°C.",
            "Driver did not monitor temperature; temperature rose from 4°C to 6.2°C.",
            "Driver ignored warning lights; temperature rose from 4°C to 7.0°C."
        ],
        "extreme_weather": [
            "External temperature reached 38°C; internal temperature rose from 4°C to 6.5°C.",
            "Heatwave caused internal temperature to rise from 4°C to 7°C in 30 minutes.",
            "Extreme cold: external temperature dropped to -15°C; internal temperature fluctuated.",
            "Heavy rainfall caused humidity to spike; temperature rose from 4°C to 5.5°C.",
            "Storm caused power fluctuations; temperature rose from 4°C to 6°C.",
            "High ambient temperature (40°C) caused cooling system to struggle; temperature rose to 7°C.",
            "Snowstorm delayed delivery; temperature rose from 4°C to 6.5°C.",
            "Extreme weather caused road closure; temperature rose from 4°C to 5.8°C.",
            "Hurricane warning: temperature rose from 4°C to 6.2°C.",
            "Heatwave: cooling system efficiency reduced; temperature rose from 4°C to 7.5°C.",
            "High humidity caused condensation; temperature rose from 4°C to 5.5°C."
        ],
        "gps_blackout": [
            "GPS signal lost in rural area; temperature remained stable at 4°C.",
            "GPS blackout due to tunnel; temperature remained stable.",
            "Satellite signal lost; temperature remained stable.",
            "GPS antenna malfunction; temperature stable.",
            "GPS signal lost for 45 minutes; temperature stable at 4°C.",
            "GPS blackout in remote area; temperature stable.",
            "GPS signal interference; temperature stable.",
            "GPS lost due to solar flare; temperature stable.",
            "GPS blackout in mountain region; temperature stable.",
            "GPS signal lost in urban canyon; temperature stable.",
            "GPS signal lost in underground parking; temperature stable."
        ]
    }
    
    test_cases = []
    
    # Calculate cases per type (ensure balanced distribution)
    anomaly_types = list(anomaly_templates.keys())
    cases_per_type = n_cases // len(anomaly_types)
    remainder = n_cases % len(anomaly_types)
    
    for i, anomaly_type in enumerate(anomaly_types):
        templates = anomaly_templates[anomaly_type]
        # Add extra cases to first few types if remainder > 0
        n_for_type = cases_per_type + (1 if i < remainder else 0)
        
        # If we have more templates than needed, sample; otherwise repeat
        if len(templates) >= n_for_type:
            selected = random.sample(templates, n_for_type)
        else:
            selected = random.choices(templates, k=n_for_type)
        
        for description in selected:
            test_cases.append({
                "text": description,
                "ground_truth": anomaly_type,
                "metadata": {
                    "type": anomaly_type,
                    "source": "synthetic"
                }
            })
    
    # Shuffle to avoid order bias
    random.shuffle(test_cases)
    
    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(test_cases, f, indent=2)
    
    print(f" Generated {len(test_cases)} test cases")
    print(f" Saved to {output_path}")
    
    # Print distribution
    type_counts = {}
    for case in test_cases:
        gt = case["ground_truth"]
        type_counts[gt] = type_counts.get(gt, 0) + 1
    print("\n Distribution:")
    for t, count in type_counts.items():
        print(f"  {t}: {count}")
    
    return test_cases

if __name__ == "__main__":
    # Generate 100 test cases
    generate_test_set(n_cases=100, output_path="data/knowledge_base/diagnosis_test_set.json")