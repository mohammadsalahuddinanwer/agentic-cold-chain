"""Main orchestrator wrapper for the cold-chain anomaly system."""
from typing import List, Dict, Any, Optional
from .langgraph_workflow import ColdChainOrchestrator, AgentState


class Orchestrator:
    """
    High-level orchestrator for the agent system.
    Provides simplified API and state management.
    """

    def __init__(self, monitor_agent, diagnosis_agent, resolver_agent, customer_agent):
        self.monitor = monitor_agent
        self.diagnosis = diagnosis_agent
        self.resolver = resolver_agent
        self.customer = customer_agent

        # Create LangGraph workflow
        self.workflow = ColdChainOrchestrator(
            monitor_agent, diagnosis_agent, resolver_agent, customer_agent
        )

        self.history = []

    def process(self, sensor_window: List[Dict], logs: List[str],
                context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single sensor window through the entire pipeline.

        Args:
            sensor_window: List of sensor readings
            logs: List of operational log messages
            context: Additional context (product_type, location, etc.)

        Returns:
            Complete result with predictions, diagnosis, plan, and messages
        """
        result = self.workflow.run(sensor_window, logs, context)

        # Store in history
        self.history.append({
            "timestamp": str(datetime.datetime.now()),
            "context": context,
            "result": result
        })

        return result

    def process_stream(self, sensor_stream: List[List[Dict]], logs_stream: List[List[str]],
                       contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a stream of sensor windows.

        Args:
            sensor_stream: List of sensor windows
            logs_stream: List of log lists
            contexts: List of contexts

        Returns:
            List of results for each window
        """
        results = []
        for window, logs, context in zip(sensor_stream, logs_stream, contexts):
            result = self.process(window, logs, context)
            results.append(result)
        return results

    def get_history(self) -> List[Dict]:
        """Get processing history."""
        return self.history

    def clear_history(self):
        """Clear processing history."""
        self.history = []

    def get_status(self) -> Dict[str, Any]:
        """Get system status."""
        return {
            "monitor_loaded": hasattr(self.monitor, 'model') and self.monitor.model is not None,
            "diagnosis_ready": self.diagnosis is not None,
            "resolver_ready": self.resolver is not None,
            "customer_ready": self.customer is not None,
            "history_length": len(self.history),
            "timestamp": str(datetime.datetime.now())
        }


import datetime