"""Specialized agent implementations for the multi-agent system."""

from __future__ import annotations

import ast
import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from travel_planner.data.context import row_to_prompt_context
from travel_planner.data.schemas import StructuredConstraints
from travel_planner.eval.parse_plan import parse_plan_text, sum_all_money
from travel_planner.graph.agent_system import Agent, AgentMessage, AgentType, MessageBus
from travel_planner.graph.llm import get_chat_llm
from travel_planner.graph.model_loader import get_solver_llm, should_use_local_model
from travel_planner.tools.rag_tools import rag_lookup
from travel_planner.tools.search import web_search

logger = logging.getLogger(__name__)

_RE_MONEY = re.compile(r"\$\s*([\d,]+(?:\.\d+)?)")
_RE_DAY_COUNT = re.compile(r"(\d+)\s*days?\b", re.IGNORECASE)
_RE_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_RE_TOTAL_COST = re.compile(
    r"(?:estimated|approx(?:imate)?|rough)?\s*total(?:\s+cost|\s+budget)?\s*[:=]?\s*\$\s*([\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)
_GENERIC_AUDIT_PREFIXES = (
    "the travel plan has",
    "the following violations",
    "specific violations include",
    "please review and revise",
    "please revise",
)
_AFFIRMATIVE_AUDIT_PHRASES = (
    "matches",
    "is correct",
    "are correct",
    "is sufficient",
    "are sufficient",
    "within budget",
    "meets all the requirements",
    "stays within the budget",
)


def _extract_failed_generation_payload(error: Exception) -> dict[str, Any] | None:
    """Recover Groq function-call payloads from tool_use_failed exceptions."""
    text = str(error)
    marker = re.search(r"<function=[^>]+>", text)
    if not marker:
        return None

    payload_src = text[marker.end():]
    start = payload_src.find("{")
    if start < 0:
        return None

    payload_text = payload_src[start:]
    depth = 0
    quote: str | None = None
    escape = False
    end_index: int | None = None
    for idx, ch in enumerate(payload_text):
        if quote is not None:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                quote = None
            continue

        if ch in ("'", '"'):
            quote = ch
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end_index = idx + 1
                break

    if end_index is None:
        return None
    payload_text = payload_text[:end_index]

    parse_attempts = [
        lambda s: json.loads(s),
        lambda s: json.loads(
            s.replace(": None", ": null")
             .replace(": True", ": true")
             .replace(": False", ": false")
        ),
        lambda s: ast.literal_eval(s),
    ]
    for parser in parse_attempts:
        try:
            parsed = parser(payload_text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    return None


def _normalize_constraints(data: dict[str, Any], metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Keep the structured fields while adding legacy keys used elsewhere in the app."""
    metadata = metadata or {}
    local_rules = data.get("local_rules")
    if isinstance(local_rules, dict):
        local_rules = ", ".join(
            f"{k}: {v}" for k, v in local_rules.items() if v not in (None, "", [], {})
        ) or None

    budget = data.get("budget_usd")
    days = data.get("num_days")

    return {
        **data,
        "origin_city": data.get("origin_city") or metadata.get("org"),
        "destination_city": data.get("destination_city") or metadata.get("dest") or "Unspecified",
        "days": days if days is not None else metadata.get("days"),
        "budget_lower": 0,
        "budget_upper": budget,
        "local_rules": local_rules,
    }


def _normalize_fragment(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = _RE_NON_ALNUM.sub(" ", text)
    return " ".join(text.split())


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = _normalize_fragment(item)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item.strip())
    return out


def _extract_money_values(text: str) -> list[float]:
    vals: list[float] = []
    for raw in _RE_MONEY.findall(text or ""):
        try:
            vals.append(float(raw.replace(",", "")))
        except Exception:
            continue
    return vals


def _estimate_total_cost(plan: str) -> float | None:
    explicit_totals = _RE_TOTAL_COST.findall(plan or "")
    if explicit_totals:
        try:
            return float(explicit_totals[-1].replace(",", ""))
        except Exception:
            pass
    return sum_all_money(plan)


def _extract_candidate_violation_lines(feedback: str, violations: list[str] | None) -> list[str]:
    candidates: list[str] = []
    parts = [feedback or ""]
    if violations:
        parts.extend(str(v or "") for v in violations)

    for block in parts:
        normalized = (
            block.replace("•", "\n")
            .replace("|", "\n")
            .replace("\t", " ")
        )
        for line in normalized.splitlines():
            clean = line.strip(" -*")
            if clean:
                candidates.append(clean)
    return _dedupe_keep_order(candidates)


def _compact_text_block(text: str, *, max_chars: int) -> str:
    cleaned = str(text or "").strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 32].rstrip() + "\n[truncated for token budget]"


def _heuristic_local_rule_violation(plan_norm: str, local_rules: str | None) -> str | None:
    rule_text = str(local_rules or "").strip()
    if not rule_text:
        return None

    rule_norm = _normalize_fragment(rule_text)
    if not rule_norm:
        return None

    if "shared room" in rule_norm and not any(
        token in plan_norm for token in ("private room", "single room", "solo room", "non shared room")
    ):
        return "Explicit room preference is not addressed in the plan."

    if "shared meal" in rule_norm and not any(
        token in plan_norm for token in ("solo dining", "individual meal", "private meal", "non shared meal")
    ):
        return "Explicit meal preference is not addressed in the plan."

    tokens = [token for token in rule_norm.split() if len(token) > 3]
    if tokens and not any(token in plan_norm for token in tokens):
        return "Explicit local rules/preferences are not clearly addressed in the plan."

    return None


def _heuristic_audit_violations(plan: str, constraints: dict[str, Any]) -> list[str]:
    parsed = parse_plan_text(plan)
    normalized_plan = _normalize_fragment(plan)
    violations: list[str] = []

    if not plan.strip():
        return ["Plan is empty."]

    if len(plan.strip()) < 160:
        violations.append("Plan is too short to be a complete itinerary.")

    expected_days = constraints.get("days") or constraints.get("num_days")
    try:
        expected_days_int = int(expected_days) if expected_days is not None else None
    except Exception:
        expected_days_int = None

    if parsed.day_sections < 1:
        violations.append("Plan is missing a clear day-by-day itinerary structure.")
    elif expected_days_int and parsed.day_sections < expected_days_int:
        violations.append(
            f"Plan covers only {parsed.day_sections} day sections but {expected_days_int} were requested."
        )

    if expected_days_int and f"day {expected_days_int}" not in normalized_plan and parsed.day_sections < expected_days_int:
        violations.append(f"Requested trip length is {expected_days_int} days, but the itinerary does not fully cover it.")

    destination = _normalize_fragment(constraints.get("destination_city"))
    if destination and destination not in normalized_plan:
        violations.append("Destination city is not clearly reflected in the itinerary.")

    origin = _normalize_fragment(constraints.get("origin_city"))
    if origin and origin not in normalized_plan:
        violations.append("Origin city/departure point is not clearly acknowledged in the itinerary.")

    budget_upper = constraints.get("budget_upper") or constraints.get("budget_usd")
    try:
        budget_float = float(budget_upper) if budget_upper is not None else None
    except Exception:
        budget_float = None

    if budget_float is not None:
        total = _estimate_total_cost(plan)
        if total is not None and total > budget_float + 1e-6:
            violations.append(
                f"Estimated total cost ${total:,.2f} exceeds the budget ceiling of ${budget_float:,.2f}."
            )

    local_rule_violation = _heuristic_local_rule_violation(
        normalized_plan,
        str(constraints.get("local_rules") or "").strip() or None,
    )
    if local_rule_violation:
        violations.append(local_rule_violation)

    if parsed.day_sections == 0 and any(token in normalized_plan for token in ("constraints", "hard requirements", "query", "api")):
        violations.append("Output reads like a constraint summary instead of a usable itinerary.")

    return _dedupe_keep_order(violations)


def _line_contradicts_constraints(line: str, constraints: dict[str, Any]) -> bool:
    normalized = _normalize_fragment(line)
    if not normalized:
        return True

    if any(phrase in normalized for phrase in _AFFIRMATIVE_AUDIT_PHRASES):
        return True

    if any(normalized.startswith(prefix) for prefix in _GENERIC_AUDIT_PREFIXES):
        return True

    if "mismatch" in normalized or "wrong" in normalized or "does not match" in normalized:
        origin = _normalize_fragment(constraints.get("origin_city"))
        destination = _normalize_fragment(constraints.get("destination_city"))
        if origin and "origin" in normalized and normalized.count(origin) >= 2:
            return True
        if destination and "destination" in normalized and normalized.count(destination) >= 2:
            return True

    day_counts = [int(x) for x in _RE_DAY_COUNT.findall(line)]
    lowered = line.lower()
    if len(day_counts) >= 2 and day_counts[0] == day_counts[1]:
        if any(token in lowered for token in ("mismatch", "insufficient", "coverage", "requested duration", "<", "vs")):
            return True

    budget_upper = constraints.get("budget_upper") or constraints.get("budget_usd")
    try:
        budget_float = float(budget_upper) if budget_upper is not None else None
    except Exception:
        budget_float = None
    if budget_float is not None and "budget" in normalized and any(
        token in normalized for token in ("exceed", "over", "above")
    ):
        amounts = _extract_money_values(line)
        if amounts and max(amounts) <= budget_float + 1e-6:
            return True

    local_rules = str(constraints.get("local_rules") or "").strip()
    if not local_rules and any(token in normalized for token in ("local rules", "local_rules", "dietary", "room constraints")):
        return True

    return False


def _finalize_audit_result(
    *,
    plan: str,
    constraints: dict[str, Any],
    feedback: str,
    violations: list[str] | None,
) -> dict[str, Any]:
    heuristic_violations = _heuristic_audit_violations(plan, constraints)
    llm_violations = [
        line for line in _extract_candidate_violation_lines(feedback, violations)
        if not _line_contradicts_constraints(line, constraints)
    ]
    combined = _dedupe_keep_order(heuristic_violations + llm_violations)
    passed = len(combined) == 0
    final_feedback = (
        "Plan satisfies the checked origin/destination, day-count, budget, and local-rule constraints."
        if passed
        else "\n".join(f"- {line}" for line in combined)
    )
    return {"passed": passed, "feedback": final_feedback}


class ConstraintExtractorAgent(Agent):
    """Agent responsible for extracting structured constraints from query."""
    
    def __init__(self, bus: MessageBus):
        super().__init__("constraint_extractor", AgentType.CONSTRAINT_EXTRACTOR, bus)
    
    async def process(self, message: AgentMessage) -> dict[str, Any]:
        """Extract constraints from query."""
        if message.action != "extract_constraints":
            return {}
        
        try:
            query = message.data.get("query", "")
            metadata = message.data.get("metadata", {})
            
            logger.info(f"Extracting constraints from: {query[:60]}...")
            
            # Require Groq or OpenAI for structured extraction
            from travel_planner.config import has_chat_llm_credentials

            if not has_chat_llm_credentials():
                logger.warning("GROQ_API_KEY / OPENAI_API_KEY not set; using minimal constraints")
                # Graceful degradation: extract what we can from query
                return {"constraints": {
                    "destination_city": metadata.get("dest", "").strip() or "Unspecified",
                    "days": int(metadata.get("days", 3)) if metadata.get("days") else 3,
                    "budget_lower": 0,
                    "budget_upper": None,
                    "local_rules": [],
                }}
            
            llm = get_chat_llm()
            structured = llm.with_structured_output(StructuredConstraints)
            
            ctx = row_to_prompt_context({"query": query, **metadata})
            msg = HumanMessage(
                content=(
                    "Extract structured travel constraints as JSON fields. "
                    "Infer budget only if clearly implied. "
                    "Use local_rules for house/cuisine/room/transport constraints.\n\n"
                    f"{ctx}"
                )
            )
            
            out: StructuredConstraints = structured.invoke([msg])
            constraints = _normalize_constraints(
                out.model_dump(mode="json"),
                metadata,
            )
            
            logger.info(f"Constraints extracted: {json.dumps(constraints, indent=2)[:200]}...")
            
            return {"constraints": constraints}
        
        except Exception as e:
            recovered = _extract_failed_generation_payload(e)
            if recovered:
                logger.warning("Recovered structured constraints from Groq tool_use_failed payload")
                return {
                    "constraints": _normalize_constraints(recovered, metadata),
                }
            logger.error(f"Constraint extraction failed: {e}", exc_info=True)
            # Graceful degradation: return minimal constraints
            return {
                "constraints": {
                    "destination_city": "Unspecified",
                    "days": 3,
                    "budget_lower": 0,
                    "budget_upper": None,
                    "local_rules": [],
                }
            }


class ResearcherAgent(Agent):
    """Agent responsible for researching flights, hotels, attractions."""
    
    def __init__(self, bus: MessageBus):
        super().__init__("researcher", AgentType.RESEARCHER, bus)
    
    async def process(self, message: AgentMessage) -> dict[str, Any]:
        """Perform research via web search and RAG."""
        if message.action != "research":
            return {}
        
        try:
            from travel_planner.tools.basic_tools import SearchTool
            
            query = message.data.get("query", "")
            metadata = message.data.get("metadata", {})
            
            logger.info(f"Researching for: {query[:60]}...")
            
            # Generate research queries
            constraints = metadata.get("constraints", {})
            dest = str(constraints.get("destination_city") or metadata.get("dest") or "").strip()
            org = str(constraints.get("origin_city") or metadata.get("org") or "").strip()
            
            if not dest:
                q = str(query or "")[:240]
                research_queries = [f"travel planning {q}"]
            else:
                research_queries = [
                    f"flights from {org} to {dest} price",
                    f"hotels in {dest} booking price",
                    f"attractions restaurants itinerary {dest}",
                ]
            
            notes: list[str] = []
            for q in research_queries:
                logger.info(f"Searching: {q}")
                # Use SearchTool explicitly to show tool usage
                web_result = SearchTool.run(q)
                notes.append(f"### Web: {q}\n{web_result}")
            
            # RAG lookup
            if dest:
                logger.info(f"RAG lookup for: {dest}")
                rag_query = f"{query} {dest} attractions hotels transport"
                try:
                    rag_text = rag_lookup(rag_query, city=dest or None, k=5)
                    if rag_text:
                        notes.append(f"### POI RAG ({dest})\n{rag_text}")
                except Exception as e:
                    logger.warning(f"RAG lookup failed: {e}")
                    notes.append(f"### POI RAG error\n{e}")
            
            research_notes = "\n\n".join(notes)
            logger.info(f"Research complete: {len(research_notes)} chars")
            
            return {"research_notes": research_notes}
        
        except Exception as e:
            logger.error(f"Research failed: {e}", exc_info=True)
            raise


class FlightAgent(Agent):
    """Searches for flight information via Tavily web search."""

    def __init__(self, bus: MessageBus):
        super().__init__("flight_agent", AgentType.FLIGHT_AGENT, bus)

    async def process(self, message: AgentMessage) -> dict[str, Any]:
        if message.action != "search_flights":
            return {}

        meta = message.data.get("metadata") or {}
        org  = str(meta.get("org")  or "").strip()
        dest = str(meta.get("dest") or "").strip()
        dep  = meta.get("departure_date") or meta.get("check_in") or ""

        if not org or not dest:
            return {"flight_notes": ""}

        date_hint = f" {dep}" if dep else ""
        query = f"cheap flights from {org} to {dest}{date_hint} price airline schedule"
        logger.info("[FlightAgent] searching: %s", query)
        text = web_search(query)
        logger.info("FlightAgent finished (%s chars)", len(text))
        return {"flight_notes": f"### Flight search: {org} → {dest}\n{text}"}


class HotelAgent(Agent):
    """Searches for hotel information via Tavily web search."""

    def __init__(self, bus: MessageBus):
        super().__init__("hotel_agent", AgentType.HOTEL_AGENT, bus)

    async def process(self, message: AgentMessage) -> dict[str, Any]:
        if message.action != "search_hotels":
            return {}

        meta = message.data.get("metadata") or {}
        dest     = str(meta.get("dest") or "").strip()
        check_in = meta.get("check_in") or meta.get("departure_date") or ""

        if not dest:
            return {"hotel_notes": ""}

        date_hint = f" {check_in}" if check_in else ""
        query = f"best hotels in {dest}{date_hint} price per night booking"
        logger.info("[HotelAgent] searching: %s", query)
        text = web_search(query)
        logger.info("HotelAgent finished (%s chars)", len(text))
        return {"hotel_notes": f"### Hotel search: {dest}\n{text}"}


class WeatherAgent(Agent):
    """Fetches Open-Meteo forecast for destination city."""

    def __init__(self, bus: MessageBus):
        super().__init__("weather_agent", AgentType.WEATHER_AGENT, bus)

    async def process(self, message: AgentMessage) -> dict[str, Any]:
        if message.action != "fetch_weather":
            return {}

        from travel_planner.tools.weather_tool import get_weather

        meta = message.data.get("metadata") or {}
        dest = str(meta.get("dest") or "").strip()
        days = meta.get("days")
        try:
            fdays = int(days) if days is not None else 7
        except Exception:
            fdays = 7

        text = get_weather(dest, forecast_days=min(max(fdays, 1), 14))
        logger.info("WeatherAgent finished (%s chars)", len(text))
        return {"weather_notes": text}


class SolverAgent(Agent):
    """Agent responsible for generating travel plans."""
    
    def __init__(self, bus: MessageBus, temperature: float = 0.4):
        super().__init__("solver", AgentType.SOLVER, bus)
        self.temperature = temperature
    
    async def process(self, message: AgentMessage) -> dict[str, Any]:
        """Generate travel plan with tool support."""
        if message.action != "generate_plan":
            return {}
        
        try:
            from travel_planner.tools.basic_tools import CalculatorTool
            
            constraints = message.data.get("constraints", {})
            research = message.data.get("research_notes", "")
            feedback = message.data.get("feedback", "")
            
            # Check if local model should be used
            if should_use_local_model():
                model_type = "LoRA"
            else:
                from travel_planner.config import has_chat_llm_credentials

                if not has_chat_llm_credentials():
                    logger.warning("GROQ_API_KEY / OPENAI_API_KEY not set; generating stub plan")
                    # Graceful degradation: create a structured stub plan
                    dest = constraints.get("destination_city", "Destination")
                    days = int(constraints.get("days", 3)) if constraints.get("days") else 3
                    budget = constraints.get("budget_upper") or "Unspecified"
                    
                    # Build middle days
                    middle_days = ""
                    for i in range(2, int(days) + 1):
                        middle_days += f"\nDay {i}: Exploration\n- Visit local attractions\n- Enjoy dining\n- Rest\n"
                    
                    plan_text = f"""TRAVEL PLAN FOR {dest}
Duration: {days} days
Budget: ${budget} USD

[Note: This is a stub plan generated without API access. Set GROQ_API_KEY (recommended) or OPENAI_API_KEY.]

Day 1: Arrival
- Arrive in {dest}
- Check into accommodation
- Explore local area
{middle_days}
Day {days}: Departure
- Last-minute sightseeing
- Depart to airport
- Safe travels!

Research Notes Summary:
{research[:500]}"""
                    return {"plan": plan_text}
                model_type = "API"
            
            logger.info(f"Solver generating plan (using {model_type} model, temperature={self.temperature})...")
            
            # TOOL USAGE: Detect when to use calculator
            query_text = message.data.get("query", "").lower()
            tool_calls_log = []
            
            if any(word in query_text for word in ["budget", "calculate", "cost", "price", "per day"]):
                logger.info("Detected budget-related query, attempting calculator tool")
                # Try to extract a calculation from constraints
                budget_upper = constraints.get("budget_upper")
                days = constraints.get("days", 3)
                if budget_upper and days:
                    try:
                        expr = f"{budget_upper}/{days}"
                        result = CalculatorTool.run(expr)
                        tool_calls_log.append(f"Calculator: {expr} = {result}")
                        logger.info(f"Budget per day: {result}")
                    except Exception as e:
                        logger.warning(f"Calculator failed: {e}")
            
            # TOOL USAGE: SearchTool explicitly called in research phase
            # (but log it here for transparency in solver)
            if research:
                logger.info("[TOOL] search already performed in research phase")
                tool_calls_log.append("Search: Completed in research phase")
            
            lk = message.data.get("solver_llm_kwargs") or {}

            def _as_float(v: Any) -> float | None:
                if v is None:
                    return None
                try:
                    return float(v)
                except Exception:
                    return None

            def _as_int(v: Any) -> int | None:
                if v is None:
                    return None
                try:
                    return int(v)
                except Exception:
                    return None

            llm = get_solver_llm(
                self.temperature,
                top_p=_as_float(lk.get("top_p")),
                max_tokens=_as_int(lk.get("max_tokens")),
                frequency_penalty=_as_float(lk.get("frequency_penalty")),
            )
            
            extra = ""
            if feedback:
                feedback = _compact_text_block(feedback, max_chars=1200)
                extra = (
                    "\nThis is a revision pass. Fix every audit issue below while preserving any correct parts "
                    "of the itinerary:\n"
                    f"{feedback}\n"
                )
            
            # Include tool call summary in context
            tool_context = ""
            if tool_calls_log:
                tool_context = "\n\nTools used during planning:\n" + "\n".join(f"  - {tc}" for tc in tool_calls_log)

            research = _compact_text_block(research, max_chars=3500)
            
            prompt = (
                "You are an expert travel planner. Produce a usable itinerary, not a constraint summary.\n"
                "Hard requirements you must satisfy:\n"
                "- Never swap the origin city and destination city.\n"
                "- Create exactly the requested number of day sections when constraints specify trip length.\n"
                "- Keep hotels, attractions, and meals anchored in the destination city unless transit requires otherwise.\n"
                "- Mention departure from the origin city and arrival in the destination city.\n"
                "- Include approximate USD costs for flight, hotel, meals, activities, and local transport.\n"
                "- End with a compact cost breakdown and an estimated total that stays within budget when a ceiling is provided.\n"
                "- Explicitly honor local rules such as private rooms or non-shared meals when present.\n"
                "- If start_date/end_date are provided, label each day with the matching travel date span.\n"
                "Use the research context as evidence, but prefer cheaper and simpler options when there is any budget tension. "
                "If a specialist tool returned a credential or lookup error, note it briefly and continue with best-effort planning.\n"
                f"{extra}\n"
                f"Structured constraints:\n{json.dumps(constraints, indent=2)}\n\n"
                f"Research notes:\n{research}\n"
                f"{tool_context}\n"
                "Output format:\n"
                "1. Trip Summary\n"
                "2. Day 1 / Day 2 / ... day-by-day itinerary\n"
                "3. Budget Breakdown with a final Estimated Total in USD\n"
            )
            
            msg_obj = HumanMessage(content=prompt)
            resp = llm.invoke([msg_obj])
            text = getattr(resp, "content", str(resp))
            
            logger.info(f"Plan generated: {len(text)} chars")
            
            return {"plan": text}
        
        except Exception as e:
            logger.error(f"Plan generation failed: {e}", exc_info=True)
            # Graceful degradation: return a minimal stub
            constraints = message.data.get("constraints", {})
            dest = constraints.get("destination_city", "Destination")
            days = constraints.get("days", 3)
            return {
                "plan": f"""MINIMAL TRAVEL PLAN FOR {dest}
Duration: {days} days

[Error: {str(e)[:100]}. Please check configuration.]

Day 1: Arrival & Exploration
Day 2-{days-1}: Activities
Day {days}: Departure"""
            }


class AuditorAgent(Agent):
    """Agent responsible for auditing plans against constraints."""
    
    def __init__(self, bus: MessageBus):
        super().__init__("auditor", AgentType.AUDITOR, bus)
    
    async def process(self, message: AgentMessage) -> dict[str, Any]:
        """Audit plan against constraints."""
        if message.action != "audit_plan":
            return {}
        
        try:
            from pydantic import BaseModel, Field
            
            class AuditResult(BaseModel):
                passed: bool
                feedback: str = ""
                violations: list[str] = Field(default_factory=list)
            
            plan = message.data.get("plan", "")
            constraints = message.data.get("constraints", {})
            
            logger.info("Auditor validating plan...")
            
            from travel_planner.config import has_chat_llm_credentials

            if not has_chat_llm_credentials():
                logger.warning("GROQ_API_KEY / OPENAI_API_KEY not set; skipping detailed audit")
                # Graceful degradation: basic heuristic checks
                is_valid = True
                feedback = ""
                violations = []
                
                # Check if plan is empty/stub
                if "[Note:" in plan or "[Error:" in plan or "MINIMAL" in plan:
                    is_valid = False
                    violations.append("Plan appears to be a stub/error plan")
                
                # Check for basic structure
                if len(plan) < 100:
                    is_valid = False
                    violations.append("Plan is too short")
                
                return {
                    "passed": is_valid,
                    "feedback": feedback + ("\n" + "\n".join(violations) if violations else ""),
                }
            
            llm = get_chat_llm(temperature=0.0)
            structured = llm.with_structured_output(AuditResult)
            
            msg_obj = HumanMessage(
                content=(
                    "Audit the travel plan against the constraints. "
                    "Fail if budget is clearly exceeded, required cities/days mismatch, or obvious impossible timing. "
                    "List specific violations.\n\n"
                    f"Constraints:\n{json.dumps(constraints, indent=2)}\n\nPlan:\n{plan}"
                )
            )
            
            out: AuditResult = structured.invoke(
                [
                    SystemMessage(
                        content=(
                            "You are a strict but evidence-based travel plan auditor. "
                            "Only fail the plan when you can point to a concrete contradiction in the text. "
                            "Do not mark a field as mismatched if the plan and constraints agree. "
                            "Do not claim the budget is exceeded unless the plan includes numeric evidence above the ceiling. "
                            "Do not call day coverage insufficient when the requested and provided day counts match. "
                            "Return concise, concrete violations only."
                        )
                    ),
                    msg_obj,
                ]
            )

            result = _finalize_audit_result(
                plan=plan,
                constraints=constraints,
                feedback=out.feedback,
                violations=out.violations,
            )

            status = "PASSED" if result["passed"] else "FAILED"
            logger.info(f"Audit {status}")
            
            return result
        
        except Exception as e:
            recovered = _extract_failed_generation_payload(e)
            if recovered:
                logger.warning("Recovered audit result from Groq tool_use_failed payload")
                return _finalize_audit_result(
                    plan=plan,
                    constraints=constraints,
                    feedback=str(recovered.get("feedback") or ""),
                    violations=recovered.get("violations") or [],
                )
            logger.error(f"Audit failed: {e}", exc_info=True)
            # Graceful degradation: pass the audit (better to have a plan than none)
            return {
                "passed": True,
                "feedback": f"Audit skipped due to error: {str(e)[:100]}",
            }
