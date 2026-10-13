"""Simulated action execution for the Resolver Agent."""
import random
import datetime
from typing import Dict, Any, List, Optional


class ActionSimulator:
    """
    Simulates execution of mitigation actions.
    In production, this would call real APIs.
    """

    def __init__(self, success_rate: float = 0.9, delay_range: tuple = (5, 60)):
        """
        Args:
            success_rate: Probability of action success (0-1)
            delay_range: Min/max seconds for action execution
        """
        self.success_rate = success_rate
        self.delay_range = delay_range
        self.action_history = []

    def execute_notify_driver(self, driver_id: str, message: str) -> Dict[str, Any]:
        """Simulate notifying a driver."""
        success = random.random() < self.success_rate
        execution_time = random.randint(*self.delay_range)

        result = {
            "action": "notify_driver",
            "driver_id": driver_id,
            "message": message,
            "success": success,
            "execution_time_seconds": execution_time,
            "timestamp": str(datetime.datetime.now()),
            "details": "Driver notification sent via SMS" if success else "Driver unreachable, retrying..."
        }

        self.action_history.append(result)
        return result

    def execute_dispatch_backup(self, vehicle_id: str, location: str) -> Dict[str, Any]:
        """Simulate dispatching a backup vehicle."""
        success = random.random() < (self.success_rate * 0.95)  # Slightly lower for logistics
        execution_time = random.randint(30, 180)

        result = {
            "action": "dispatch_backup",
            "vehicle_id": vehicle_id,
            "location": location,
            "success": success,
            "execution_time_seconds": execution_time,
            "timestamp": str(datetime.datetime.now()),
            "details": f"Backup vehicle {vehicle_id} dispatched to {location}" if success else "No backup available"
        }

        self.action_history.append(result)
        return result

    def execute_repair_on_site(self, location: str) -> Dict[str, Any]:
        """Simulate on-site repair."""
        success = random.random() < (self.success_rate * 0.8)
        execution_time = random.randint(60, 300)

        result = {
            "action": "repair_on_site",
            "location": location,
            "success": success,
            "execution_time_seconds": execution_time,
            "timestamp": str(datetime.datetime.now()),
            "details": "Repair completed successfully" if success else "Repair failed, escalation required"
        }

        self.action_history.append(result)
        return result

    def execute_reroute_fastest(self, shipment_id: str, new_route: str) -> Dict[str, Any]:
        """Simulate fast rerouting."""
        success = random.random() < (self.success_rate * 0.9)
        execution_time = random.randint(10, 40)

        result = {
            "action": "reroute_fastest",
            "shipment_id": shipment_id,
            "new_route": new_route,
            "success": success,
            "execution_time_seconds": execution_time,
            "timestamp": str(datetime.datetime.now()),
            "details": f"Shipment {shipment_id} rerouted via {new_route}" if success else "Reroute failed"
        }

        self.action_history.append(result)
        return result

    def execute_escalate_to_human(self, issue: str) -> Dict[str, Any]:
        """Simulate escalation to human operator."""
        success = True  # Escalation always succeeds
        execution_time = random.randint(10, 60)

        result = {
            "action": "escalate_to_human",
            "issue": issue,
            "success": success,
            "execution_time_seconds": execution_time,
            "timestamp": str(datetime.datetime.now()),
            "details": f"Escalated to human operator: {issue}"
        }

        self.action_history.append(result)
        return result

    def execute_action(self, action_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute any action by name."""
        actions = {
            "notify_driver": self.execute_notify_driver,
            "dispatch_backup": self.execute_dispatch_backup,
            "repair_on_site": self.execute_repair_on_site,
            "reroute_fastest": self.execute_reroute_fastest,
            "escalate_to_human": self.execute_escalate_to_human,
        }

        if action_name in actions:
            return actions[action_name](**params)
        else:
            return {
                "action": action_name,
                "success": False,
                "details": f"Unknown action: {action_name}",
                "execution_time_seconds": 0,
                "timestamp": str(datetime.datetime.now())
            }

    def get_history(self) -> List[Dict]:
        """Get all executed actions."""
        return self.action_history

    def clear_history(self):
        """Clear action history."""
        self.action_history = []