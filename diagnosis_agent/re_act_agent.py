"""ReAct (Reasoning + Acting) agent for anomaly diagnosis."""
from typing import Dict, Any, List, Optional
import json
import re

class ReActAgent:
    """
    Implements the ReAct reasoning loop: Thought → Action → Observation.
    Based on Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (ICLR 2023).
    """

    def __init__(self, llm, rag_retriever, max_iterations: int = 5):
        self.llm = llm
        self.rag = rag_retriever
        self.max_iterations = max_iterations
        self.tools = {
            "retrieve_similar_cases": self._retrieve_cases,
            "get_sensor_analysis": self._analyze_sensors,
            "get_environmental_data": self._get_environment,
            "final_answer": self._final_answer
        }

    def diagnose(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run ReAct loop to diagnose anomaly.
        """
        iterations = []
        current_context = context.copy()
        last_raw_response = ""

        for i in range(self.max_iterations):
            prompt = self._build_prompt(current_context, iterations)
            # FIX 1: stop the model as soon as it tries to write the tool's output
            # itself. Without a stop sequence the model continued past its first
            # action, emitted further Action blocks, and invented "Observation:"
            # text (which is where the phantom retrieval similarity of 0.821 in
            # the earlier write-up came from).
            # Stop only on "Observation": the model must not write the tool's own
            # output, which is what let it invent retrieval results.
            #
            # DO NOT add "\nThought:" here. It was tried and it breaks generation
            # completely: the model opens its completion with a newline followed by
            # "Thought:", so the stop string matches at position 0, every response
            # comes back as the single word "Thought:", and every case scores
            # "unknown". Rambling across several Thought/Action pairs is handled
            # instead by _extract_action (which prefers final_answer) and by the
            # truncation recovery in _json_after_action_input.
            raw_response = self.llm.generate(
                prompt, max_new_tokens=384,
                stop_strings=["Observation:", "\nObservation"])

            # Guard against a degenerate completion: too short to contain an
            # action at all. Retry once with no stop strings before giving up,
            # because scoring the case "unknown" on a generation artefact would
            # understate the model.
            if "Action:" not in raw_response and len(raw_response.strip()) < 40:
                print(f"DEBUG: degenerate completion {raw_response.strip()!r}; "
                      "retrying once without stop strings")
                raw_response = self.llm.generate(prompt, max_new_tokens=384)
            print(f"DEBUG: Raw LLM response:\n{raw_response}")
            import sys
            sys.stdout.flush()
            last_raw_response = raw_response

            thought = self._extract_thought(raw_response)
            action = self._extract_action(raw_response)

            if not action or action.get("action") == "final_answer":
                # Try to get diagnosis from action_input
                final = action.get("action_input", {}) if action else {}
                diagnosis = final.get("diagnosis", {})

                # --- FALLBACK: if diagnosis empty, parse raw response ---
                if not diagnosis:
                    print(f"DEBUG: Raw LLM response:\n{raw_response}")
                    # Try to find JSON in raw response
                    # FIX 5b. The previous pattern was r'\{.*\}' with DOTALL, which
                    # is greedy: it spanned from the FIRST brace in the response to
                    # the LAST, swallowing several separate JSON objects and failing
                    # with "Extra data". Scan each balanced object instead and take
                    # the first one that carries a diagnosis.
                    for candidate in self._balanced_objects(raw_response):
                        try:
                            parsed = json.loads(candidate)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(parsed, dict) and "diagnosis" in parsed:
                            diagnosis = parsed["diagnosis"]
                            break
                        if isinstance(parsed, dict) and "anomaly_type" in parsed:
                            diagnosis = parsed
                            break
                    # Last resort: the label alone, from the model's own text.
                    if not diagnosis:
                        m_any = re.search(r'"anomaly_type"\s*:\s*"([A-Za-z_]+)"',
                                          raw_response)
                        if m_any:
                            diagnosis = {"anomaly_type": m_any.group(1),
                                         "severity": 0.5,
                                         "root_cause": "recovered from truncated output",
                                         "recommended_action": "escalate_to_human"}
                            print("DEBUG: recovered anomaly_type="
                                  f"{m_any.group(1)} from the raw response")

                # If still empty, use a default diagnosis
                if not diagnosis:
                    diagnosis = {
                        "anomaly_type": "unknown",
                        "severity": 0.5,
                        "root_cause": "Could not parse LLM output",
                        "recommended_action": "escalate_to_human"
                    }

                return {
                    "status": "completed",
                    "diagnosis": diagnosis,
                    "iterations": i + 1,
                    "history": iterations,
                    "confidence": final.get("confidence", 0.8) if final else 0.8,
                    # FIX 4: record where the answer came from, so the evaluation
                    # can audit itself instead of the provenance being invisible.
                    # "llm_truncated" means the answer is the model's but its JSON
                    # was cut off by the token limit and the fields were recovered.
                    "used_fallback": False,
                    "answer_source": ("llm_truncated"
                                      if final.get("_truncated") else "llm"),
                }

            # Execute tool
            tool_name = action.get("action")
            tool_input = action.get("action_input", {})
            observation = self.tools.get(tool_name, lambda x: f"Error: tool {tool_name} not found")(tool_input)

            iterations.append({
                "thought": thought,
                "action": action,
                "observation": observation
            })

            current_context["last_observation"] = observation

        # ================================================================
        # HARDCODED FALLBACK (Called when max iterations reached)
        # ================================================================
        # This is a keyword match over operational_logs, NOT a model output. It
        # answered 54 of 60 cases in the original evaluation because a parser
        # defect (see _extract_action) stopped final_answer from ever being
        # executed. Any result that reaches here must be reported as produced by
        # the keyword rule, not by the LLM.
        print("DEBUG: max_iterations reached - answering from the KEYWORD FALLBACK, "
              "not the LLM")
        diagnosis = self._fallback_diagnosis_from_logs(current_context)

        return {
            "status": "completed",
            "diagnosis": diagnosis,
            "iterations": self.max_iterations,
            "history": iterations,
            "confidence": 0.7,
            "used_fallback": True,
            "answer_source": "keyword_fallback",
        }

    def _build_prompt(self, context: Dict[str, Any], history: List) -> str:
            monitor_pred = context.get("monitor_prediction", {})
            sensor_data  = context.get("sensor_data", [])
            op_logs      = context.get("operational_logs", [])
    
            prompt = f"""You are a cold-chain logistics anomaly diagnosis expert.
    
    ANOMALY CONTEXT:
    Monitor prediction: {json.dumps(monitor_pred, indent=2)}
    Operational logs: {op_logs}
    Sensor readings available: {len(sensor_data)} data points
    
    VALID ANOMALY TYPES (you MUST output one of these exactly):
    - compressor_failure: Internal mechanical failure of refrigeration unit (compressor, fan, valve, refrigerant leak). Rapid temperature rise >2 degrees in <15 minutes. Internal hardware fault.
    - power_outage: Total electrical power loss to vehicle or refrigeration. Generator/mains/fuse failure. All electrical systems affected. NOT a mechanical compressor fault.
    - door_left_open: Physical door open or door seal breach. Warm ambient air entering. Temperature rise is GRADUAL (0.1-0.5 degrees per minute). Human or loading bay action.
    - driver_error: Human operator mistake. Driver set wrong temperature, ignored alarm, turned off unit, or failed to close door properly. Deliberate or negligent human action.
    - extreme_weather: External environmental cause. Weather event (storm, heatwave, snow, rain) or extreme ambient temperature. Cooling unit working but overwhelmed. No internal fault.
    - gps_blackout: GPS signal loss ONLY. Temperature is STABLE and completely unaffected. Navigation issue only. No thermal anomaly.
    
    AVAILABLE TOOLS:
    1. retrieve_similar_cases(query): Search knowledge base for similar past cases.
       IMPORTANT: Use detailed descriptive queries, not single words.
       Example: instead of "snowstorm" use "external weather snowstorm temperature gradual rise no mechanical fault"
       Example: instead of "power" use "total electrical power loss generator failed all systems down"
    2. get_sensor_analysis(features): Analyse sensor patterns.
    3. get_environmental_data(): Check weather and route conditions.
    4. final_answer(diagnosis): Output final diagnosis in JSON format.
    
    {self._format_history(history)}
    
    INSTRUCTIONS:
    - Always call retrieve_similar_cases FIRST with a detailed query
    - The anomaly_type in your final_answer MUST be exactly one of the 6 valid types listed above
    - Do NOT invent new anomaly type names
    - Base your diagnosis on retrieved cases and the operational logs
    
    Now follow this format exactly:
    Thought: [your reasoning about what to do next]
    Action: tool_name
    Action Input: {{"parameter": "value"}}
    OR
    Thought: I have enough information to answer.
    Action: final_answer
    Action Input: {{"diagnosis": {{"anomaly_type": "one_of_the_6_valid_types", "severity": 0.0, "root_cause": "...", "recommended_action": "..."}}, "confidence": 0.0}}
    """
            return prompt

    def _format_history(self, history: List) -> str:
        if not history:
            return "No previous steps yet."
        lines = []
        for h in history:
            lines.append(f"Thought: {h['thought']}")
            lines.append(f"Action: {h['action']['action']}")
            lines.append(f"Action Input: {json.dumps(h['action']['action_input'])}")
            lines.append(f"Observation: {h['observation']}")
        return "\n".join(lines)

    def _extract_thought(self, response: str) -> str:
        match = re.search(r"Thought:\s*(.+?)(?=Action:|$)", response, re.DOTALL)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _balanced_objects(text: str):
        """Yield each balanced {...} block in the text, outermost only."""
        depth, start = 0, None
        for i, ch in enumerate(text):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}" and depth:
                depth -= 1
                if depth == 0 and start is not None:
                    yield text[start:i + 1]
                    start = None

    @staticmethod
    def _json_after_action_input(text: str) -> Dict:
        """Return the first balanced {...} block following 'Action Input:'.

        FIX 3a. The previous pattern, ({.+?}) with a lookahead, could not handle
        the nested JSON that final_answer returns:
            {"diagnosis": {"anomaly_type": ...}, "confidence": 0.8}
        Brace counting handles nesting, and returns {} when the completion was
        truncated mid-JSON rather than silently yielding a fragment.
        """
        m = re.search(r"Action Input:\s*", text)
        if not m:
            return {}
        s = text[m.end():].lstrip()
        if not s.startswith("{"):
            return {"text": s.split("\n")[0].strip()}
        depth = 0
        for i, ch in enumerate(s):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(s[:i + 1])
                    except json.JSONDecodeError:
                        return {"text": s[:i + 1]}
        # FIX 5: the completion was cut off mid-JSON by max_new_tokens. The model
        # had already committed to an answer, so salvage it rather than scoring the
        # case as "unknown" -- two of five smoke-test cases were lost this way, both
        # with the correct label plainly visible in the truncated text. The result
        # is still the model's own answer, and diagnose() marks it as truncated so
        # the provenance stays honest.
        try:                                        # (a) close any open braces
            return json.loads(s + "}" * depth)
        except json.JSONDecodeError:
            pass
        m_type = re.search(r'"anomaly_type"\s*:\s*"([A-Za-z_]+)"', s)
        if m_type:                                  # (b) read the fields directly
            diagnosis = {"anomaly_type": m_type.group(1)}
            m_root = re.search(r'"root_cause"\s*:\s*"([^"]*)"', s)
            if m_root:
                diagnosis["root_cause"] = m_root.group(1)
            m_sev = re.search(r'"severity"\s*:\s*([0-9.]+)', s)
            if m_sev:
                diagnosis["severity"] = float(m_sev.group(1))
            m_act = re.search(r'"recommended_action"\s*:\s*"([^"]*)"', s)
            if m_act:
                diagnosis["recommended_action"] = m_act.group(1)
            out = {"diagnosis": diagnosis, "_truncated": True}
            m_conf = re.search(r'"confidence"\s*:\s*([0-9.]+)', s)
            if m_conf:
                out["confidence"] = float(m_conf.group(1))
            print("DEBUG: final_answer JSON was truncated; recovered "
                  f"anomaly_type={diagnosis['anomaly_type']} from the model's own text")
            return out
        return {}                       # nothing recoverable

    def _extract_action(self, response: str) -> Optional[Dict]:
        """Choose which action to execute from one completion.

        FIX 3b. A single completion often contains an exploratory action and then
        a commitment to final_answer. The previous version used re.search, which
        returns the FIRST match, so final_answer was never executed: the loop ran
        to max_iterations and the hardcoded keyword fallback answered instead, in
        54 of 60 evaluation cases. final_answer now wins wherever it appears, and
        the Action Input is taken from the text that follows the chosen action
        rather than from the start of the response.
        """
        matches = list(re.finditer(r"Action:\s*(\w+)", response))
        if not matches:
            return None
        chosen = next((m for m in matches if m.group(1) == "final_answer"),
                      matches[0])
        return {"action": chosen.group(1),
                "action_input": self._json_after_action_input(response[chosen.end():])}

    def _retrieve_cases(self, query: Dict) -> str:
        query_text = query.get("text", query.get("query", ""))
        results = self.rag.retrieve(query_text, top_k=3)
        if results:
            return json.dumps([{"text": r["text"][:200], "similarity": r["similarity"]} for r in results], indent=2)
        return "No similar cases found."

    def _analyze_sensors(self, features: Dict) -> str:
        temp_trend = features.get("temperature_trend", "stable")
        door_events = features.get("door_events", 0)
        vibration = features.get("vibration_level", "normal")

        if door_events > 0:
            return "Pattern suggests door‑open event. Temperature rise correlates with door activity."
        elif vibration == "high":
            return "High vibration detected – possible mechanical issue or rough handling."
        elif temp_trend == "rising":
            return "Rising temperature trend without door events – likely compressor or power issue."
        return "No clear sensor pattern detected."

    def _get_environment(self, _: Dict = None) -> str:
        return "Weather: cloudy, ambient 15°C, no extreme events forecast for next 2 hours."

    def _final_answer(self, diagnosis_input: Dict) -> str:
        return "Diagnosis complete."

    def _fallback_diagnosis(self, context: Dict) -> Dict:
        return {
            "anomaly_type": "unknown",
            "severity": 0.5,
            "root_cause": "Cannot determine – insufficient information",
            "recommended_action": "escalate_to_human"
        }

    # ================================================================
    # HARDCODED FALLBACK FROM LOGS
    # ================================================================
    def _fallback_diagnosis_from_logs(self, context: Dict) -> Dict:
        """
        Hardcoded fallback: guesses the anomaly type based on keywords in operational logs.
        This is used when the ReAct loop fails to produce a diagnosis.
        """
        logs = context.get("operational_logs", [])
        text = " ".join(logs).lower()
        
        # Check for keywords in the operational logs
        if "compressor" in text or "cooling" in text or "refrigerat" in text or "cool" in text:
            return {
                "anomaly_type": "compressor_failure",
                "severity": 0.8,
                "root_cause": "Cooling system or compressor issue detected",
                "recommended_action": "dispatch_backup"
            }
        elif "door" in text or "seal" in text or "ajar" in text:
            return {
                "anomaly_type": "door_left_open",
                "severity": 0.6,
                "root_cause": "Door left open or seal failure",
                "recommended_action": "notify_driver"
            }
        elif "gps" in text or "signal" in text or "satellite" in text:
            return {
                "anomaly_type": "gps_blackout",
                "severity": 0.3,
                "root_cause": "GPS signal lost",
                "recommended_action": "manual_tracking"
            }
        elif "driver" in text or "operator" in text:
            return {
                "anomaly_type": "driver_error",
                "severity": 0.5,
                "root_cause": "Driver or operator error",
                "recommended_action": "contact_driver"
            }
        elif "weather" in text or "snow" in text or "rain" in text or "storm" in text or "heat" in text:
            return {
                "anomaly_type": "extreme_weather",
                "severity": 0.5,
                "root_cause": "Extreme weather conditions",
                "recommended_action": "adjust_cooling"
            }
        elif "power" in text or "electrical" in text or "outage" in text:
            return {
                "anomaly_type": "power_outage",
                "severity": 0.7,
                "root_cause": "Power failure",
                "recommended_action": "dispatch_backup"
            }
        elif "sensor" in text:
            return {
                "anomaly_type": "sensor_malfunction",
                "severity": 0.4,
                "root_cause": "Sensor malfunction",
                "recommended_action": "check_sensors"
            }
        else:
            return {
                "anomaly_type": "unknown",
                "severity": 0.5,
                "root_cause": "Insufficient information from logs",
                "recommended_action": "escalate_to_human"
            }