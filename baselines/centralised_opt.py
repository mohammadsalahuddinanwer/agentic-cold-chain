"""Baseline B2: Centralised optimisation (van Lon-style tabu search)."""
import random
from typing import Dict, List, Any
import numpy as np


class CentralisedOptimisationBaseline:
    """
    Centralised optimisation baseline using tabu search.
    Based on van Lon (2017) implementation.
    """

    def __init__(self, n_vehicles: int = 10, tabu_size: int = 10, max_iterations: int = 100, threshold: float = 1.0):
        self.n_vehicles = n_vehicles
        self.tabu_size = tabu_size
        self.max_iterations = max_iterations
        self.threshold = threshold  # <-- ADD THIS LINE
        self.tabu_list = []
        self.best_solution = None

    def generate_initial_solution(self, tasks: List[Dict]) -> List[int]:
        """Generate random assignment of tasks to vehicles."""
        return [random.randint(0, self.n_vehicles - 1) for _ in range(len(tasks))]

    def calculate_cost(self, assignment: List[int], tasks: List[Dict]) -> float:
        """Calculate total cost of an assignment."""
        total_cost = 0.0
        vehicle_loads = {i: [] for i in range(self.n_vehicles)}

        for task_idx, vehicle in enumerate(assignment):
            vehicle_loads[vehicle].append(tasks[task_idx])

        for vehicle, assigned_tasks in vehicle_loads.items():
            # Cost increases with number of tasks and their urgency
            if assigned_tasks:
                n_tasks = len(assigned_tasks)
                urgency_sum = sum(t.get('urgency', 0.5) for t in assigned_tasks)
                distance = random.uniform(10, 100)  # Simulated distance
                total_cost += n_tasks * 10 + urgency_sum * 5 + distance * 0.1

        return total_cost

    def get_neighbours(self, assignment: List[int]) -> List[List[int]]:
        """Generate neighbour solutions by moving one task to a different vehicle."""
        neighbours = []
        for i in range(len(assignment)):
            for v in range(self.n_vehicles):
                if assignment[i] != v:
                    new_assignment = assignment.copy()
                    new_assignment[i] = v
                    neighbours.append(new_assignment)
        return neighbours

    def process_anomaly(self, anomaly: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an anomaly using centralised optimisation.

        Args:
            anomaly: Dict with 'tasks', 'location', 'severity'

        Returns:
            Optimised assignment with cost and latency
        """
        tasks = anomaly.get('tasks', [{'urgency': anomaly.get('severity', 0.5)}])

        if not tasks:
            return {
                "assigned": False,
                "reason": "No tasks to assign",
                "cost": 0
            }

        # Generate initial solution
        current = self.generate_initial_solution(tasks)
        current_cost = self.calculate_cost(current, tasks)

        # Tabu search
        best = current.copy()
        best_cost = current_cost

        for iteration in range(self.max_iterations):
            neighbours = self.get_neighbours(current)

            # Filter tabu moves
            valid_neighbours = []
            for n in neighbours:
                if n not in self.tabu_list:
                    valid_neighbours.append(n)

            if not valid_neighbours:
                # Random restart
                current = self.generate_initial_solution(tasks)
                current_cost = self.calculate_cost(current, tasks)
                continue

            # Select best neighbour
            best_neighbour = valid_neighbours[0]
            best_neighbour_cost = self.calculate_cost(best_neighbour, tasks)

            for n in valid_neighbours[1:]:
                n_cost = self.calculate_cost(n, tasks)
                if n_cost < best_neighbour_cost:
                    best_neighbour = n
                    best_neighbour_cost = n_cost

            # Update tabu list
            self.tabu_list.append(current)
            if len(self.tabu_list) > self.tabu_size:
                self.tabu_list.pop(0)

            # Move to best neighbour
            current = best_neighbour
            current_cost = best_neighbour_cost

            # Update global best
            if current_cost < best_cost:
                best = current.copy()
                best_cost = current_cost

        self.best_solution = best

        # Calculate resolution latency (proportional to cost)
        resolution_latency = best_cost * 0.5 + random.uniform(10, 50)

        return {
            "assigned": True,
            "assignment": best,
            "cost": best_cost,
            "resolution_latency_seconds": resolution_latency,
            "iterations": self.max_iterations,
            "tabu_size": self.tabu_size,
            "n_vehicles": self.n_vehicles
        }

    def process(self, sensor_data: List[Dict]) -> Dict[str, Any]:
        """
        Process sensor data: detect anomaly (temperature spike) and run centralised optimisation only if anomaly detected.
        """
        # Extract temperature readings
        temps = []
        for record in sensor_data:
            # Try to find a temperature column
            for key, value in record.items():
                if 'temperature' in key.lower():
                    temps.append(value)
                    break
            else:
                # If no temperature column found, use the first value
                temps.append(list(record.values())[0])

        # Detect anomaly: if any temperature exceeds threshold
        is_anomaly = any(t > self.threshold for t in temps)

        if not is_anomaly:
            # No anomaly detected – return zero
            return {
                "cost_multiplier": 0.0,
                "resolution_time_seconds": 0.0,
                "anomalies_detected": 0
            }

        # Anomaly detected – run optimisation
        severity = max(temps) / 10.0
        severity = min(1.0, severity)
        anomaly = {
            "tasks": [{"urgency": severity}],
            "severity": severity
        }
        result = self.process_anomaly(anomaly)

        return {
            "cost_multiplier": result.get("cost", 100) / 100,
            "resolution_time_seconds": result.get("resolution_latency_seconds", 0),
            "anomalies_detected": 1 if result.get("assigned", False) else 0
        }