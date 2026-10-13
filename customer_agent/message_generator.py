"""Message generation utilities for the Customer Agent."""
from typing import Dict, Any, List
import json
from .prompt_templates import PromptTemplates


class MessageGenerator:
    """
    Generates natural-language messages for stakeholders.
    Uses LLM with prompt templates.
    """

    def __init__(self, llm):
        self.llm = llm
        self.templates = PromptTemplates()

    def generate_update(self, role: str, diagnosis: Dict[str, Any],
                        plan: Dict[str, Any], context: Dict[str, Any]) -> str:
        """
        Generate a stakeholder-specific update.

        Args:
            role: 'shipper', 'receiver', 'carrier', 'maintenance'
            diagnosis: from Diagnosis Agent
            plan: from Resolver Agent
            context: shipment context
        """
        # Get template
        template = self.templates.get_template(role)

        # Prepare variables
        variables = {
            "anomaly_type": diagnosis.get('anomaly_type', 'temperature anomaly'),
            "severity": f"{diagnosis.get('severity', 0.5) * 100:.0f}%",
            "action_description": plan.get('description', 'mitigation action initiated'),
            "estimated_time": str(plan.get('estimated_time_seconds', 60)),
            "root_cause": diagnosis.get('root_cause', 'under investigation'),
            "product_type": context.get('product_type', 'temperature-sensitive goods'),
            "quality_assurance": "Quality remains assured" if plan.get('success_probability', 0.8) > 0.7 else "Quality may be compromised, inspection advised",
            "instructions": "Follow standard operating procedure for this anomaly type",
            "sensor_summary": str(context.get('sensor_data', [])[-5:]) if context.get('sensor_data') else "No recent sensor data",
            "location": context.get('current_location', 'en route')
        }

        # Fill template
        filled_template = template.format(**variables)

        # Use LLM to refine
        refinement_prompt = f"""
Original message draft:
{filled_template}

Refine this message to make it:
- Professional and clear
- Appropriate for the stakeholder ({role})
- Concise (max 4 sentences)
- Do NOT add any information not already present

Return only the final message, no commentary.
"""
        final_message = self.llm.generate(refinement_prompt)

        return final_message.strip()

    def generate_summary(self, diagnosis: Dict, plan: Dict) -> str:
        """Generate a concise incident summary."""
        prompt = f"""
Generate a one-sentence incident summary for an anomaly resolution:

Anomaly: {diagnosis.get('anomaly_type')}
Severity: {diagnosis.get('severity')}
Action: {plan.get('description')}
Outcome: {'Success' if plan.get('success_probability', 0) > 0.7 else 'Partial success'}
"""
        return self.llm.generate(prompt).strip()

    def generate_batch(self, roles: List[str], diagnosis: Dict[str, Any],
                       plan: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, str]:
        """Generate messages for multiple stakeholders."""
        return {
            role: self.generate_update(role, diagnosis, plan, context)
            for role in roles
        }