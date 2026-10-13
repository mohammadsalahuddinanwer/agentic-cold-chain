"""Prompt templates for the Customer Agent."""


class PromptTemplates:
    """Collection of prompt templates for different stakeholders."""

    @staticmethod
    def shipper_template() -> str:
        return """You are a customer service AI for a cold-chain logistics company. Generate an update for the SHIPPER (who sent the goods).
Be professional, transparent about the issue, and provide actionable next steps.

Anomaly: {anomaly_type}
Severity: {severity}
Action taken: {action_description}
Estimated resolution time: {estimated_time} seconds
Root cause: {root_cause}
Product: {product_type}

Write a concise update (2-3 sentences) that reassures the shipper while being honest."""

    @staticmethod
    def receiver_template() -> str:
        return """You are a customer service AI for a cold-chain logistics company. Generate an update for the RECEIVER (awaiting the goods).
Focus on expected delivery timeline and quality assurance.

Anomaly detected: {anomaly_type}
Action taken: {action_description}
Estimated impact on delivery: {estimated_time} seconds delay
Product quality: {quality_assurance}

Write a reassuring, clear update (2 sentences)."""

    @staticmethod
    def carrier_template() -> str:
        return """You are a logistics coordinator. Generate an OPERATIONAL UPDATE for the CARRIER (driver/dispatcher team).
Use clear, action-oriented language.

Issue: {anomaly_type}
Required action: {action_description}
Timeline: {estimated_time} seconds
Additional instructions: {instructions}

Write a concise command or set of commands (2-3 sentences)."""

    @staticmethod
    def maintenance_template() -> str:
        return """You are a technical coordinator. Generate a TECHNICAL UPDATE for the MAINTENANCE CREW.

Equipment anomaly: {anomaly_type}
Sensor data: {sensor_summary}
Location: {location}
Suggested action: {action_description}

Write a technical report (2-3 sentences) with actionable diagnostics."""

    @staticmethod
    def default_template() -> str:
        return """Generate an update for the logistics team about an anomaly.

Anomaly: {anomaly_type}
Severity: {severity}
Action: {action_description}
Location: {location}

Write a concise update (2-3 sentences)."""

    def get_template(self, role: str) -> str:
        """Get template by role."""
        templates = {
            "shipper": self.shipper_template(),
            "receiver": self.receiver_template(),
            "carrier": self.carrier_template(),
            "maintenance": self.maintenance_template(),
        }
        return templates.get(role.lower(), self.default_template())