"""Multi-agent TravelPlanner workflow with message-based communication and parallel execution.

KEY IMPROVEMENTS:
1. Message-based communication between agents (not state dict)
2. Parallel execution: constraints + research + flights + hotels + weather concurrently
3. Coordinator agent orchestrates workflow
4. Async/await for true concurrency
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _solver_llm_kwargs(
    *,
    top_p: float | None,
    max_tokens: int | None,
    frequency_penalty: float | None,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if top_p is not None:
        out["top_p"] = top_p
    if max_tokens is not None:
        out["max_tokens"] = max_tokens
    if frequency_penalty is not None:
        out["frequency_penalty"] = frequency_penalty
    return out


async def run_travel_planner_async(
    query: str,
    row_metadata: dict[str, Any] | None = None,
    *,
    row_id: int | str | None = None,
    temperature: float = 0.4,
    top_p: float | None = None,
    max_tokens: int | None = None,
    frequency_penalty: float | None = None,
) -> dict[str, Any]:
    """Run multi-agent system asynchronously.

    Uses Groq (or OpenAI fallback) for chat agents; parallel specialists include
    web+RAG research (Tavily), flight/hotel search (Tavily), and Open-Meteo weather.
    """
    from travel_planner.graph.agent_system import CoordinatorAgent, MessageBus
    from travel_planner.graph.agents import (
        AuditorAgent,
        ConstraintExtractorAgent,
        FlightAgent,
        HotelAgent,
        ResearcherAgent,
        SolverAgent,
        WeatherAgent,
    )

    logger.info("Starting multi-agent travel planner")

    bus = MessageBus()

    coordinator = CoordinatorAgent(bus)
    constraint_extractor = ConstraintExtractorAgent(bus)
    researcher = ResearcherAgent(bus)
    flight_agent = FlightAgent(bus)
    hotel_agent = HotelAgent(bus)
    weather_agent = WeatherAgent(bus)
    solver = SolverAgent(bus, temperature=temperature)
    auditor = AuditorAgent(bus)

    logger.info("Agents created, starting orchestration")

    agent_tasks = [
        asyncio.create_task(constraint_extractor.handle_messages()),
        asyncio.create_task(researcher.handle_messages()),
        asyncio.create_task(flight_agent.handle_messages()),
        asyncio.create_task(hotel_agent.handle_messages()),
        asyncio.create_task(weather_agent.handle_messages()),
        asyncio.create_task(solver.handle_messages()),
        asyncio.create_task(auditor.handle_messages()),
    ]

    llm_kw = _solver_llm_kwargs(
        top_p=top_p,
        max_tokens=max_tokens,
        frequency_penalty=frequency_penalty,
    )

    try:
        result = await asyncio.wait_for(
            coordinator.orchestrate(query, row_metadata, solver_llm_kwargs=llm_kw),
            timeout=180.0,
        )

        logger.info("Orchestration complete")

        result["message_history"] = bus.get_history()

        if row_id is not None:
            result["row_id"] = row_id

        result["temperature"] = temperature
        result["top_p"] = top_p
        result["max_tokens"] = max_tokens
        result["frequency_penalty"] = frequency_penalty

        return result

    finally:
        for task in agent_tasks:
            task.cancel()

        await asyncio.gather(*agent_tasks, return_exceptions=True)


def run_travel_planner(
    query: str,
    row_metadata: dict[str, Any] | None = None,
    *,
    row_id: int | str | None = None,
    temperature: float = 0.4,
    top_p: float | None = None,
    max_tokens: int | None = None,
    frequency_penalty: float | None = None,
) -> dict[str, Any]:
    """Synchronous wrapper for the async multi-agent system."""
    try:
        asyncio.get_running_loop()
        raise RuntimeError("Use run_travel_planner_async() in async context")
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                run_travel_planner_async(
                    query,
                    row_metadata,
                    row_id=row_id,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    frequency_penalty=frequency_penalty,
                )
            )
        finally:
            loop.close()


def build_graph():
    """Legacy function for backward compatibility."""
    logger.warning(
        "build_graph() is deprecated. Use run_travel_planner() or run_travel_planner_async()"
    )
    raise NotImplementedError(
        "Old LangGraph workflow replaced with multi-agent system. "
        "Use run_travel_planner() instead."
    )


__all__ = ["run_travel_planner", "run_travel_planner_async", "build_graph"]
