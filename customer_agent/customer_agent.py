"""Customer Agent – generates stakeholder‑specific natural‑language updates."""
from typing import Dict, Any
import json

class CustomerAgent:
    """
    Generates transparent, context‑aware updates for different stakeholders.
    """

    def __init__(self, llm):
        self.llm = llm

        self.templates = {
            "shipper": """
You are a customer service AI for a cold‑chain logistics company. Generate an update for the SHIPPER (who sent the goods).
Be professional, transparent about the issue, and provide actionable next steps.

Anomaly: {anomaly_type}
Severity: {severity}
Action taken: {action_description}
Estimated resolution time: {estimated_time} seconds
Root cause: {root_cause}
Product: {product_type}

Write a concise update (2‑3 sentences) that reassures the shipper while being honest.
""",
            "receiver": """
You are a customer service AI for a cold‑chain logistics company. Generate an update for the RECEIVER (awaiting the goods).
Focus on expected delivery timeline and quality assurance.

Anomaly detected: {anomaly_type}
Action taken: {action_description}
Estimated impact on delivery: {estimated_time} seconds delay
Product quality: {quality_assurance}

Write a reassuring, clear update (2 sentences).
""",
            "carrier": """
You are a logistics coordinator. Generate an OPERATIONAL UPDATE for the CARRIER (driver/dispatcher team).
Use clear, action‑oriented language.

Issue: {anomaly_type}
Required action: {action_description}
Timeline: {estimated_time} seconds
Additional instructions: {instructions}

Write a concise command or set of commands (2‑3 sentences).
""",
            "maintenance": """
You are a technical coordinator. Generate a TECHNICAL UPDATE for the MAINTENANCE CREW.

Equipment anomaly: {anomaly_type}
Sensor data: {sensor_summary}
Location: {location}
Suggested action: {action_description}

Write a technical report (2‑3 sentences) with actionable diagnostics.
"""
        }

    def generate_message(self, role: str, diagnosis: Dict[str, Any], plan: Dict[str, Any], context: Dict[str, Any]) -> str:
        template = self.templates.get(role.lower(), self.templates["shipper"])

        variables = {
            "anomaly_type": diagnosis.get('anomaly_type', 'temperature anomaly'),
            "severity": f"{diagnosis.get('severity', 0.5) * 100:.0f}%",
            "action_description": plan.get('description', 'mitigation action initiated'),
            "estimated_time": str(plan.get('estimated_time_seconds', 60)),
            "root_cause": diagnosis.get('root_cause', 'under investigation'),
            "product_type": context.get('product_type', 'temperature‑sensitive goods'),
            "quality_assurance": "Quality remains assured" if plan.get('success_probability', 0.8) > 0.7 else "Quality may be compromised, inspection advised",
            "instructions": "Follow standard operating procedure for this anomaly type",
            "sensor_summary": str(context.get('sensor_data', [])[-5:]),
            "location": context.get('current_location', 'en route')
        }

        filled_template = template.format(**variables)

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
        prompt = f"""
Generate a one‑sentence incident summary for an anomaly resolution:

Anomaly: {diagnosis.get('anomaly_type')}
Severity: {diagnosis.get('severity')}
Action: {plan.get('description')}
Outcome: {'Success' if plan.get('success_probability', 0) > 0.7 else 'Partial success'}
"""
        return self.llm.generate(prompt).strip()