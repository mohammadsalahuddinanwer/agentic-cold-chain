"""LangGraph orchestration for the four agents."""
from typing import TypedDict, List, Dict, Any
import operator
from langgraph.graph import StateGraph, END


class AgentState(TypedDict):
    """Shared state across all agents."""
    sensor_window: List[Dict]
    operational_logs: List[str]
    context: Dict[str, Any]
    monitor_prediction: Dict[str, Any]
    diagnosis: Dict[str, Any]
    resolution_plan: Dict[str, Any]
    customer_messages: Dict[str, str]
    next_agent: str
    error: str
    iteration: int

class ColdChainOrchestrator:
    """
    LangGraph orchestrator for the cold‑chain anomaly system.
    Manages state, transitions, and error handling.
    """

    def __init__(self, monitor_agent, diagnosis_agent, resolver_agent, customer_agent):
        self.monitor = monitor_agent
        self.diagnosis = diagnosis_agent
        self.resolver = resolver_agent
        self.customer = customer_agent

        self.graph = self._build_graph()
        #self.memory = MemorySaver()
        #self.memory = InMemorySaver()

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(AgentState)

        graph.add_node("monitor", self._monitor_node)
        graph.add_node("diagnosis", self._diagnosis_node)
        graph.add_node("resolver", self._resolver_node)
        graph.add_node("customer", self._customer_node)
        graph.add_node("error_handler", self._error_node)

        graph.set_entry_point("monitor")

        graph.add_conditional_edges(
            "monitor",
            self._after_monitor,
            {
                "diagnosis": "diagnosis",
                "error": "error_handler",
                "end": END
            }
        )

        graph.add_conditional_edges(
            "diagnosis",
            self._after_diagnosis,
            {
                "resolver": "resolver",
                "customer": "customer",
                "error": "error_handler",
                "end": END
            }
        )

        graph.add_conditional_edges(
            "resolver",
            self._after_resolver,
            {
                "customer": "customer",
                "error": "error_handler",
                "end": END
            }
        )

        graph.add_edge("customer", END)
        graph.add_edge("error_handler", END)

        #return graph.compile(checkpointer=self.memory)
        #return graph.compile(checkpointer=InMemorySaver())
        return graph.compile()

    def _monitor_node(self, state: AgentState) -> Dict:
        try:
            from config import Config
            print(f"DEBUG _monitor_node: MONITOR_PROB_THRESHOLD = {Config.MONITOR_PROB_THRESHOLD}")
            prediction = self.monitor.predict_anomaly(
                state['sensor_window'],
                threshold=Config.MONITOR_PROB_THRESHOLD
            )
            # Use integer keys (30, 60, 90) instead of strings
            is_anomaly = prediction.get(30, {}).get("is_anomaly", False)
            temperature = prediction.get(30, {}).get("temperature", "N/A")
            print(f"DEBUG _monitor_node: is_anomaly = {is_anomaly}, temperature = {temperature}")
            next_agent = "diagnosis" if is_anomaly else "end"
            return {
                "monitor_prediction": prediction,
                "next_agent": next_agent,
                "iteration": state.get('iteration', 0) + 1
            }
        except Exception as e:
            print(f" _monitor_node error: {e}")
            return {"error": str(e), "next_agent": "error"}

    def _diagnosis_node(self, state: AgentState) -> Dict:
        try:
            context = {
                "monitor_prediction": state['monitor_prediction'],
                "sensor_data": state['sensor_window'],
                "operational_logs": state['operational_logs']
            }
            diagnosis = self.diagnosis.diagnose(context)
            return {
                "diagnosis": diagnosis,
                "next_agent": "resolver" if diagnosis.get('confidence', 0) > 0.5 else "customer"
            }
        except Exception as e:
            return {"error": str(e), "next_agent": "error"}

    def _resolver_node(self, state: AgentState) -> Dict:
        try:
            plan = self.resolver.generate_plan(state['diagnosis'], state['context'])
            return {
                "resolution_plan": plan,
                "next_agent": "customer"
            }
        except Exception as e:
            return {"error": str(e), "next_agent": "error"}

    def _customer_node(self, state: AgentState) -> Dict:
        try:
            messages = {}
            for role in ["shipper", "receiver", "carrier"]:
                msg = self.customer.generate_message(
                    role=role,
                    diagnosis=state.get('diagnosis', {}),
                    plan=state.get('resolution_plan', {}),
                    context=state['context']
                )
                messages[role] = msg

            return {
                "customer_messages": messages,
                "next_agent": "end"
            }
        except Exception as e:
            return {"error": str(e), "next_agent": "error"}

    def _error_node(self, state: AgentState) -> Dict:
        error_msg = state.get('error', 'Unknown error')
        return {
            "customer_messages": {
                "shipper": f"System encountered an error. {error_msg} Investigation underway.",
                "receiver": "Technical issue detected. Your shipment is being monitored manually.",
                "carrier": f"ERROR: {error_msg} – escalate to manual support."
            },
            "next_agent": "end"
        }

    def _after_monitor(self, state: AgentState) -> str:
        if state.get('error'):
            return "error"
        return state.get('next_agent', 'end')

    def _after_diagnosis(self, state: AgentState) -> str:
        if state.get('error'):
            return "error"
        return state.get('next_agent', 'customer')

    def _after_resolver(self, state: AgentState) -> str:
        if state.get('error'):
            return "error"
        return state.get('next_agent', 'customer')

    def run(self, sensor_window: List[Dict], logs: List[str], context: Dict[str, Any]) -> Dict[str, Any]:
        initial_state: AgentState = {
            "sensor_window": sensor_window,
            "operational_logs": logs,
            "context": context,
            "monitor_prediction": {},
            "diagnosis": {},
            "resolution_plan": {},
            "customer_messages": {},
            "next_agent": "monitor",
            "error": "",
            "iteration": 0
        }

        config = {"configurable": {"thread_id": "1"}}
        final_state = self.graph.invoke(initial_state, config)

        return final_state
