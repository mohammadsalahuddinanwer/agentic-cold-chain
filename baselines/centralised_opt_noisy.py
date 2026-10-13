"""
Centralised optimisation with imperfect (noisy) diagnosis.
This mimics real-world scenarios where anomaly classification is ~66% accurate.
"""
import random
import numpy as np
from typing import Dict, Any, List  # <-- FIXED: Added missing imports
from .centralised_opt import CentralisedOptimisationBaseline

class CentralisedOptimisationNoisyBaseline(CentralisedOptimisationBaseline):
    """
    Injects classification noise based on a confusion matrix.
    The central planner receives the WRONG anomaly type X% of the time.
    """
    def __init__(self, n_vehicles=10, tabu_size=10, max_iterations=100, 
                 noise_rate=0.34, confusion_matrix=None):
        super().__init__(n_vehicles, tabu_size, max_iterations)
        self.noise_rate = noise_rate  # 0.34 corresponds to 66% accuracy
        self.confusion_matrix = confusion_matrix  # 6x6 matrix from Appendix D
        
        if self.confusion_matrix is None:
            # Fallback: uniform confusion across classes
            self.confusion_matrix = np.ones((6, 6)) * 0.1
            np.fill_diagonal(self.confusion_matrix, 0.66)  # 66% diagonal accuracy
    
    def corrupt_label(self, true_label: int) -> int:
        """Given true label, sample a noisy label based on confusion matrix."""
        if random.random() > self.noise_rate:
            return true_label  # Correct
        # Sample from confusion matrix row (probabilities of misclassification)
        probs = self.confusion_matrix[true_label] / np.sum(self.confusion_matrix[true_label])
        return np.random.choice(range(6), p=probs)
    
    def process_anomaly(self, anomaly: Dict[str, Any]) -> Dict[str, Any]:
        """
        Override: before optimisation, corrupt the anomaly type based on diagnosis noise.
        """
        # Simulate diagnosis noise
        true_type = anomaly.get('type', 0)  # 0-5
        noisy_type = self.corrupt_label(true_type)
        
        # Modify the task urgency based on the noisy diagnosis
        tasks = anomaly.get('tasks', [{'urgency': 0.5}])
        # If misclassified, urgency might be set incorrectly
        if noisy_type != true_type:
            # Increase randomness in urgency to simulate poor diagnosis
            for task in tasks:
                task['urgency'] = random.uniform(0.1, 0.9)  # Random urgency
        
        # Run the parent optimiser with the noisy data
        return super().process_anomaly({
            "tasks": tasks,
            "severity": anomaly.get('severity', 0.5)
        })