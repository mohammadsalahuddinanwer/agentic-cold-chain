"""Metrics for evaluating the cold-chain anomaly system."""
import numpy as np
from typing import List, Dict, Any
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


class MetricsCalculator:
    """
    Calculates evaluation metrics for the agent system.
    """

    @staticmethod
    def detection_metrics(predictions: List[bool], ground_truth: List[bool]) -> Dict[str, float]:
        """
        Calculate detection metrics.

        Args:
            predictions: List of predicted anomaly flags
            ground_truth: List of true anomaly flags

        Returns:
            Dict with precision, recall, f1, accuracy
        """
        if not predictions:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}

        return {
            "precision": float(precision_score(ground_truth, predictions, zero_division=0)),
            "recall": float(recall_score(ground_truth, predictions, zero_division=0)),
            "f1": float(f1_score(ground_truth, predictions, zero_division=0)),
            "accuracy": float(accuracy_score(ground_truth, predictions))
        }

    @staticmethod
    def lead_time_metrics(lead_times: List[float]) -> Dict[str, float]:
        """
        Calculate lead time metrics.

        Args:
            lead_times: List of lead times in minutes

        Returns:
            Dict with mean, median, std, min, max
        """
        if not lead_times:
            return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

        return {
            "mean": float(np.mean(lead_times)),
            "median": float(np.median(lead_times)),
            "std": float(np.std(lead_times)),
            "min": float(np.min(lead_times)),
            "max": float(np.max(lead_times))
        }

    @staticmethod
    def resolution_metrics(resolution_times: List[float], costs: List[float]) -> Dict[str, float]:
        """
        Calculate resolution metrics.

        Args:
            resolution_times: List of resolution times in seconds
            costs: List of costs in USD

        Returns:
            Dict with mean resolution time, mean cost, etc.
        """
        if not resolution_times:
            return {"mean_resolution_time": 0.0, "mean_cost": 0.0, "success_rate": 0.0}

        # Assume success if resolution time < 300 seconds and cost > 0
        success_rate = sum(1 for t in resolution_times if t < 300) / len(resolution_times)

        return {
            "mean_resolution_time": float(np.mean(resolution_times)),
            "median_resolution_time": float(np.median(resolution_times)),
            "std_resolution_time": float(np.std(resolution_times)),
            "mean_cost": float(np.mean(costs)) if costs else 0.0,
            "success_rate": float(success_rate)
        }

    @staticmethod
    def cost_metrics(costs: List[float], baselines: Dict[str, List[float]]) -> Dict[str, float]:
        """
        Calculate cost metrics relative to baselines.

        Args:
            costs: List of costs for the proposed system
            baselines: Dict mapping baseline name to list of costs

        Returns:
            Dict with percentage improvement over each baseline
        """
        if not costs:
            return {}

        metrics = {
            "mean_cost": float(np.mean(costs)),
            "median_cost": float(np.median(costs)),
            "std_cost": float(np.std(costs))
        }

        for name, baseline_costs in baselines.items():
            if baseline_costs:
                baseline_mean = np.mean(baseline_costs)
                improvement = ((baseline_mean - np.mean(costs)) / baseline_mean) * 100
                metrics[f"improvement_vs_{name}_percent"] = float(improvement)

        return metrics

    @staticmethod
    def accuracy_metrics(predictions: List[Any], ground_truth: List[Any]) -> Dict[str, float]:
        """Calculate classification accuracy metrics."""
        if not predictions:
            return {"accuracy": 0.0, "correct": 0, "total": 0}

        correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
        return {
            "accuracy": float(correct / len(predictions)),
            "correct": correct,
            "total": len(predictions)
        }

    @staticmethod
    def calculate_all(y_true: List, y_pred: List, lead_times: List[float],
                      resolution_times: List[float], costs: List[float],
                      baseline_costs: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Calculate all metrics in one call.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            lead_times: Detection lead times
            resolution_times: Resolution times
            costs: System costs
            baseline_costs: Dict of baseline costs

        Returns:
            Complete metrics dictionary
        """
        metrics = {}

        # Detection metrics
        if y_true and y_pred:
            metrics["detection"] = MetricsCalculator.detection_metrics(y_pred, y_true)

        # Lead time metrics
        if lead_times:
            metrics["lead_time"] = MetricsCalculator.lead_time_metrics(lead_times)

        # Resolution metrics
        if resolution_times:
            metrics["resolution"] = MetricsCalculator.resolution_metrics(resolution_times, costs)

        # Cost metrics
        if costs:
            metrics["cost"] = MetricsCalculator.cost_metrics(costs, baseline_costs)

        return metrics