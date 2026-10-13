"""Resolver Agent – generates and selects optimal mitigation plans."""
from typing import Dict, Any, List
import random
import json

from .evolutionary_opt import EvolutionaryOptimiser


# ── Anomaly-type action guidance ─────────────────────────────────────────────
# Maps anomaly type to recommended primary and secondary actions.
# Based on cold-chain domain knowledge and Loisel et al. (2021).
ANOMALY_ACTION_GUIDANCE = {
    # primary   = action when severity >= threshold (serious case)
    # secondary = action when severity <  threshold (mild case)
    "compressor_failure": {
        "primary":   "dispatch_backup",  # serious: send backup vehicle
        "secondary": "repair_on_site",   # mild:    try on-site repair first
        "rationale": "Mechanical failure requires backup for high severity; repair for mild",
        "severity_threshold": 0.5,
    },
    "door_left_open": {
        "primary":   "dispatch_backup",  # serious: cargo compromised, send backup
        "secondary": "notify_driver",    # mild:    driver closes door immediately
        "rationale": "Minor door event: notify driver. Severe: cargo compromised, dispatch backup",
        "severity_threshold": 0.8,
    },
    "driver_error": {
        "primary":   "escalate_to_human",  # serious: human oversight needed
        "secondary": "notify_driver",      # mild:    remind driver of protocol
        "rationale": "Minor driver error: notify. Serious misconduct: escalate to human",
        "severity_threshold": 0.8,
    },
    "extreme_weather": {
        "primary":   "dispatch_backup",  # serious: temperature breached, send backup
        "secondary": "reroute_safest",   # mild:    reroute to avoid weather
        "rationale": "Mild weather: reroute. Severe weather/breach: dispatch backup",
        "severity_threshold": 0.7,
    },
    "power_outage": {
        "primary":   "dispatch_backup",  # serious: no cooling at all
        "secondary": "repair_on_site",   # mild:    brief glitch, try repair
        "rationale": "Power outage always serious; minor glitch may be repairable",
        "severity_threshold": 0.4,
    },
    "gps_blackout": {
        "primary":   "notify_driver",    # always: GPS only, cargo unaffected
        "secondary": "notify_driver",    # same: no escalation needed
        "rationale": "GPS loss is tracking only; cargo temperature always unaffected",
        "severity_threshold": 1.1,       # threshold > 1: always uses secondary (notify_driver)
    },
    "sensor_malfunction": {
        "primary":   "escalate_to_human",  # always: need human to verify
        "secondary": "notify_driver",      # mild: notify driver to check sensor
        "rationale": "Sensor fault always needs human verification before any action",
        "severity_threshold": 1.1,         # threshold > 1: always escalate
    },
}


class ResolverAgent:
    """
    Autonomous resolver that generates and selects mitigation actions.
    Combines LLM plan generation with evolutionary cost optimisation.
    Uses anomaly-type-aware prompting to ensure appropriate action selection.
    """

    def __init__(self, llm, optimiser: EvolutionaryOptimiser = None):
        self.llm = llm
        self.optimiser = optimiser or EvolutionaryOptimiser()
        self.action_templates = {
            "notify_driver":    {"cost_rerouting": 5,   "time": 10,  "carbon": 0},
            "dispatch_backup":  {"cost_rerouting": 50,  "time": 120, "carbon": 15},
            "repair_on_site":   {"cost_rerouting": 30,  "time": 60,  "carbon": 2},
            "reroute_fastest":  {"cost_rerouting": 20,  "time": 40,  "carbon": 5},
            "reroute_safest":   {"cost_rerouting": 15,  "time": 70,  "carbon": 3},
            "escalate_to_human":{"cost_rerouting": 100, "time": 300, "carbon": 0},
            "ignore":           {"cost_rerouting": 0,   "time": 0,   "carbon": 0},
        }

    def generate_plan(self, diagnosis: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate and select optimal mitigation plan."""
        candidates = self._llm_generate_candidates(diagnosis, context)

        evaluated = []
        for action in candidates:
            cost = self._compute_action_cost(action, diagnosis, context)
            evaluated.append({
                **action,
                "cost": cost,
                "metrics": cost["metrics"],
            })

        best_action = min(evaluated, key=lambda x: x["cost"]["total"])

        plan = {
            "action":                  best_action["name"],
            "description":             best_action["description"],
            "estimated_time_seconds":  best_action["estimated_time"],
            "cost_breakdown":          best_action["metrics"],
            "total_weighted_cost":     best_action["cost"]["total"],
            "success_probability":     self._estimate_success_probability(diagnosis, best_action),
        }
        return plan

    def _llm_generate_candidates(self, diagnosis: Dict, context: Dict) -> List[Dict]:
        """
        Generate action candidates using anomaly-type-aware prompting.
        The prompt explicitly guides the LLM to consider anomaly type
        and severity before proposing candidates, preventing the conservative
        bias of always selecting dispatch_backup for all anomaly types.
        """
        anomaly_type = diagnosis.get("diagnosis", {}).get("anomaly_type",
                       diagnosis.get("anomaly_type", "unknown"))
        severity     = diagnosis.get("diagnosis", {}).get("severity",
                       diagnosis.get("severity", 0.5))
        root_cause   = diagnosis.get("diagnosis", {}).get("root_cause",
                       diagnosis.get("root_cause", "unknown"))

        # Get domain guidance for this anomaly type
        guidance = ANOMALY_ACTION_GUIDANCE.get(anomaly_type, {})
        primary   = guidance.get("primary",   "dispatch_backup")
        secondary = guidance.get("secondary", "escalate_to_human")
        rationale = guidance.get("rationale", "")
        threshold = guidance.get("severity_threshold", 0.5)

        # Determine recommended action based on severity
        # Simple rule: severity >= threshold → primary action
        #              severity <  threshold → secondary action
        if severity >= threshold:
            recommended = primary
        else:
            recommended = secondary

        prompt = f"""You are a cold-chain logistics resolver. Select the most appropriate
mitigation action for this anomaly based on type, severity, and domain knowledge.

ANOMALY DETAILS:
- Type: {anomaly_type}
- Severity: {severity:.2f} (0=low, 1=critical)
- Root cause: {root_cause}

DOMAIN GUIDANCE FOR {anomaly_type.upper()}:
- Primary action: {primary}
- Rationale: {rationale}
- Recommended for severity {severity:.2f}: {recommended}

AVAILABLE ACTIONS (select appropriate ones for THIS anomaly type):
- notify_driver: Contact driver immediately (time=10s, low cost) — for human errors, door issues
- dispatch_backup: Send backup reefer vehicle (time=120s, high cost) — for mechanical failures, power loss
- repair_on_site: Arrange on-site repair (time=60s, medium cost) — for repairable faults
- reroute_fastest: Reroute to nearest cold storage (time=40s) — for weather, route issues
- reroute_safest: Reroute via safest path (time=70s) — for weather anomalies
- escalate_to_human: Human operator decision (time=300s) — for sensor faults, uncertainty
- ignore: No action (only for GPS blackout or trivial sensor noise)

Propose 3 candidates appropriate for {anomaly_type} with severity {severity:.2f}.
Include {recommended} as the primary candidate.
Do NOT always default to dispatch_backup unless the anomaly genuinely requires it.

Return ONLY a JSON list in this exact format:
[
  {{"name": "action_name", "description": "brief description", "estimated_time": seconds}},
  {{"name": "action_name", "description": "brief description", "estimated_time": seconds}},
  {{"name": "action_name", "description": "brief description", "estimated_time": seconds}}
]"""

        response = self.llm.generate(prompt)

        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if json_match:
                candidates = json.loads(json_match.group())
                if isinstance(candidates, list) and len(candidates) > 0:
                    # Validate action names
                    valid = [c for c in candidates
                             if c.get("name") in self.action_templates]
                    if valid:
                        return valid
        except (json.JSONDecodeError, AttributeError):
            pass

        # Fallback: use domain guidance directly
        fallback = [
            {"name": recommended,
             "description": f"{recommended.replace('_',' ').title()} for {anomaly_type}",
             "estimated_time": self.action_templates[recommended]["time"]},
            {"name": secondary,
             "description": f"{secondary.replace('_',' ').title()} as alternative",
             "estimated_time": self.action_templates[secondary]["time"]},
            {"name": "escalate_to_human",
             "description": "Escalate if above actions insufficient",
             "estimated_time": 300},
        ]
        return fallback

    def _compute_action_cost(self, action: Dict, diagnosis: Dict, context: Dict) -> Dict:
        template = self.action_templates.get(
            action["name"], self.action_templates["escalate_to_human"])

        anomaly_type = diagnosis.get("diagnosis", {}).get("anomaly_type",
                       diagnosis.get("anomaly_type", "unknown"))
        severity     = diagnosis.get("diagnosis", {}).get("severity",
                       diagnosis.get("severity", context.get("severity", 0.5)))
        product_value = context.get("product_value", 1000)

        # Effectiveness adjusted for anomaly type appropriateness
        base_effectiveness = self._action_effectiveness(action["name"])
        type_match = self._action_type_match(action["name"], anomaly_type)
        adjusted_effectiveness = base_effectiveness * type_match

        spoilage_cost  = product_value * severity * (1 - adjusted_effectiveness)
        rerouting_cost = template["cost_rerouting"] * context.get("distance_factor", 1.0)
        delay_cost     = template["time"] * context.get("delay_penalty_per_second", 0.01)
        carbon_cost    = template["carbon"] * context.get("carbon_price", 1.0)

        metrics = {
            "spoilage":  spoilage_cost,
            "rerouting": rerouting_cost,
            "delay":     delay_cost,
            "carbon":    carbon_cost,
        }
        total_cost = self.optimiser.compute_cost(metrics)
        return {"total": total_cost, "metrics": metrics}

    def _action_type_match(self, action_name: str, anomaly_type: str) -> float:
        """
        Returns effectiveness multiplier based on how appropriate the action
        is for the given anomaly type. This prevents dispatch_backup dominating
        for anomalies that only need driver notification.
        """
        guidance = ANOMALY_ACTION_GUIDANCE.get(anomaly_type, {})
        primary   = guidance.get("primary", "dispatch_backup")
        secondary = guidance.get("secondary", "escalate_to_human")

        if action_name == primary:
            return 1.0    # perfect match
        elif action_name == secondary:
            return 0.85   # good match
        elif action_name == "dispatch_backup" and anomaly_type in [
                "door_left_open", "driver_error", "gps_blackout", "sensor_malfunction"]:
            return 0.3    # overkill — strongly penalise dispatch for minor anomalies
        elif action_name == "reroute_safest" and anomaly_type in [
                "door_left_open", "driver_error"]:
            return 0.35   # rerouting unnecessary for human-caused anomalies
        else:
            return 0.7    # acceptable but not ideal

    def _action_effectiveness(self, action_name: str) -> float:
        effectiveness = {
            "notify_driver":     0.92,  # raised: effective for human-caused anomalies
            "dispatch_backup":   0.95,
            "repair_on_site":    0.80,
            "reroute_fastest":   0.85,
            "reroute_safest":    0.90,
            "escalate_to_human": 0.70,
            "ignore":            0.10,
        }
        return effectiveness.get(action_name, 0.5)

    def _estimate_success_probability(self, diagnosis: Dict, action: Dict) -> float:
        base = self._action_effectiveness(action["name"])
        severity = diagnosis.get("diagnosis", {}).get("severity",
                   diagnosis.get("severity", 0.5))
        if severity > 0.8:
            base *= 0.85
        return min(0.99, base)

    def execute_plan(self, plan: Dict) -> Dict:
        import datetime
        success = random.random() < plan["success_probability"]
        return {
            "plan":           plan,
            "executed":       success,
            "timestamp":      str(datetime.datetime.now()),
            "result_message": "Action executed successfully" if success
                              else "Action execution failed, retrying...",
        }
