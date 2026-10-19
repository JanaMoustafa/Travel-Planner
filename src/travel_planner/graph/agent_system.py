"""True multi-agent system with message-based communication and async execution.

This module replaces the sequential pipeline with a proper multi-agent architecture featuring:
1. Message-based communication between agents (not state dict passing)
2. Async/await for parallel execution
3. Agent registry and central message bus
4. Coordinator agent for orchestration
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Types of agents in the system."""
    COORDINATOR = "coordinator"
    CONSTRAINT_EXTRACTOR = "constraint_extractor"
    RESEARCHER = "researcher"
    FLIGHT_AGENT = "flight_agent"
    HOTEL_AGENT = "hotel_agent"
    WEATHER_AGENT = "weather_agent"
    SOLVER = "solver"
    AUDITOR = "auditor"


@dataclass
class AgentMessage:
    """Message passed between agents (CRITICAL: replaces state dict).
    
    This is the KEY to true multi-agent communication.
    Instead of agents reading/writing to shared state,
    they send/receive explicit messages.
    """
    sender: str  # Agent name that sent this message
    receiver: str  # Target agent name (or "broadcast")
    action: str  # What the sender wants (e.g., "extract_constraints", "research")
    data: dict[str, Any] = field(default_factory=dict)
    
    message_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    status: str = "pending"  # "pending", "completed", "failed"
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    
    def __repr__(self) -> str:
        return f"[{self.message_id}] {self.sender}->{self.receiver}: {self.action}"


class MessageBus:
    """Central message broker for agent communication.
    
    Agents send messages through the bus instead of directly modifying state.
    """
    
    def __init__(self):
        self.messages: list[AgentMessage] = []
        self._lock = asyncio.Lock()
    
    async def send(self, message: AgentMessage) -> None:
        """Agent sends a message."""
        async with self._lock:
            self.messages.append(message)
            logger.info(f"Message sent: {message}")
    
    async def receive(self, receiver: str) -> list[AgentMessage]:
        """Agent receives messages addressed to them."""
        async with self._lock:
            messages = [m for m in self.messages if m.receiver == receiver and m.status in ("pending", "completed")]
            # Don't remove - messages stay in history for auditing
            return messages
    
    async def get_latest_from(self, sender: str, action: str | None = None) -> AgentMessage | None:
        """Get latest message from specific sender."""
        async with self._lock:
            matching = [
                m for m in reversed(self.messages)
                if m.sender == sender and (action is None or m.action == action)
            ]
            return matching[0] if matching else None
    
    async def get_message_by_id(self, message_id: str) -> AgentMessage | None:
        """Get message by ID."""
        async with self._lock:
            for m in self.messages:
                if m.message_id == message_id:
                    return m
            return None
    
    async def update_status(self, message_id: str, status: str, result: dict | None = None, error: str | None = None) -> None:
        """Update message status after processing."""
        async with self._lock:
            for m in self.messages:
                if m.message_id == message_id:
                    m.status = status
                    if result:
                        m.result = result
                    if error:
                        m.error = error
                    logger.info(f"Message {message_id} status={status}")
                    return
    
    def get_history(self) -> str:
        """Get formatted message history for auditing."""
        lines = ["Agent Communication History:"]
        for m in self.messages:
            lines.append(f"  {m.timestamp} | {m.sender}->{m.receiver} | {m.action} | {m.status}")
        return "\n".join(lines)


class Agent:
    """Base agent class.
    
    All agents inherit from this and implement the process() method
    to handle specific actions.
    """
    
    def __init__(self, name: str, agent_type: AgentType, bus: MessageBus):
        self.name = name
        self.agent_type = agent_type
        self.bus = bus
        self.processed_messages = 0
    
    async def send_message(
        self,
        receiver: str,
        action: str,
        data: dict[str, Any] | None = None,
    ) -> AgentMessage:
        """Send message to another agent."""
        message = AgentMessage(
            sender=self.name,
            receiver=receiver,
            action=action,
            data=data or {},
        )
        await self.bus.send(message)
        return message
    
    async def broadcast(
        self,
        action: str,
        data: dict[str, Any] | None = None,
    ) -> AgentMessage:
        """Send message to all agents."""
        message = AgentMessage(
            sender=self.name,
            receiver="broadcast",
            action=action,
            data=data or {},
        )
        await self.bus.send(message)
        return message
    
    async def listen(self) -> list[AgentMessage]:
        """Receive messages addressed to this agent."""
        return await self.bus.receive(self.name)
    
    async def listen_to(self, sender: str, action: str | None = None) -> AgentMessage | None:
        """Wait for specific message from specific sender."""
        return await self.bus.get_latest_from(sender, action)
    
    async def process(self, message: AgentMessage) -> dict[str, Any]:
        """Process a message. To be implemented by subclasses."""
        raise NotImplementedError
    
    async def handle_messages(self):
        """Main loop: listen for messages and process them."""
        while True:
            messages = await self.listen()
            for msg in messages:
                if msg.status == "pending":
                    try:
                        logger.info(f"{self.name} processing: {msg.action}")
                        result = await self.process(msg)
                        await self.bus.update_status(
                            msg.message_id,
                            "completed",
                            result=result
                        )
                        self.processed_messages += 1
                    except Exception as e:
                        logger.error(f"{self.name} failed: {e}")
                        await self.bus.update_status(
                            msg.message_id,
                            "failed",
                            error=str(e)
                        )
            await asyncio.sleep(0.1)  # Prevent busy loop


class CoordinatorAgent(Agent):
    """Main orchestrator agent.
    
    Decides which agents to call and in what order.
    Coordinates parallel execution.
    """
    
    def __init__(self, bus: MessageBus):
        super().__init__("coordinator", AgentType.COORDINATOR, bus)
    
    async def orchestrate(
        self,
        query: str,
        row_metadata: dict[str, Any] | None = None,
        *,
        solver_llm_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Orchestrate the multi-agent system.
        
        This is the main entry point that coordinates all agents.
        Key improvement: Parallel execution where possible.
        Graceful degradation when agents fail.
        """
        logger.info("Coordinator: Starting orchestration")
        llm_kw = solver_llm_kwargs or {}

        # Step 1: Extract constraints, research, flights, hotels, weather in parallel
        logger.info("Step 1: Running constraint extraction, research, flights, hotels, and weather in parallel")

        constraint_task = self._request_constraints(query, row_metadata)
        research_task = self._request_research(query, row_metadata)
        flight_task = self._request_flights(query, row_metadata)
        hotel_task = self._request_hotels(query, row_metadata)
        weather_task = self._request_weather(query, row_metadata)

        results = await asyncio.gather(
            constraint_task,
            research_task,
            flight_task,
            hotel_task,
            weather_task,
            return_exceptions=True,
        )

        raw_constraints = {} if isinstance(results[0], Exception) else (results[0] or {})
        if isinstance(raw_constraints, dict) and "constraints" in raw_constraints:
            constraints = raw_constraints.get("constraints") or {}
        else:
            constraints = raw_constraints if isinstance(raw_constraints, dict) else {}
        research_notes = "" if isinstance(results[1], Exception) else (results[1] or "")
        flight_notes = "" if isinstance(results[2], Exception) else (results[2] or "")
        hotel_notes = "" if isinstance(results[3], Exception) else (results[3] or "")
        weather_notes = "" if isinstance(results[4], Exception) else (results[4] or "")

        if isinstance(results[0], Exception):
            logger.warning("Constraint extraction failed (graceful): %s", results[0])
        if isinstance(results[1], Exception):
            logger.warning("Research failed (graceful): %s", results[1])
        if isinstance(results[2], Exception):
            logger.warning("Flight search failed (graceful): %s", results[2])
        if isinstance(results[3], Exception):
            logger.warning("Hotel search failed (graceful): %s", results[3])
        if isinstance(results[4], Exception):
            logger.warning("Weather lookup failed (graceful): %s", results[4])

        parts: list[str] = []
        if research_notes:
            parts.append(research_notes)
        if flight_notes:
            parts.append(flight_notes)
        if hotel_notes:
            parts.append(hotel_notes)
        if weather_notes:
            parts.append(weather_notes)
        combined_research = "\n\n".join(parts)

        logger.info("Step 1 complete: parallel specialist tasks finished")
        
        # Step 2: Generate plan using constraints and combined research
        logger.info("Step 2: Solver generating plan")
        draft_plan = await self._request_plan(
            constraints,
            combined_research,
            query=query,
            solver_llm_kwargs=llm_kw,
        )
        
        # Step 3: Audit plan
        logger.info("Step 3: Auditor validating plan")
        audit_result = await self._request_audit(draft_plan, constraints)
        
        # Step 4: If audit failed, loop back to solver (adaptive feedback)
        from travel_planner.config import MAX_AUDIT_ITERATIONS

        iterations = 1
        max_iterations = max(1, int(MAX_AUDIT_ITERATIONS))
        while not audit_result.get("passed") and iterations < max_iterations:
            logger.info(f"Audit iteration {iterations}: Solver revising")
            iterations += 1
            draft_plan = await self._request_plan(
                constraints,
                combined_research,
                query=query,
                feedback=audit_result.get("feedback", ""),
                solver_llm_kwargs=llm_kw,
            )
            audit_result = await self._request_audit(draft_plan, constraints)
        
        logger.info("Orchestration complete")
        
        return {
            "query": query,
            "constraints": constraints,
            "research_notes": combined_research,
            "draft_plan": draft_plan,
            "audit_passed": audit_result.get("passed", False),
            "audit_feedback": audit_result.get("feedback", ""),
            "solver_calls": iterations,
        }
    
    async def _request_constraints(self, query: str, metadata: dict | None) -> dict:
        """Request constraint extraction from extractor agent."""
        msg = await self.send_message(
            "constraint_extractor",
            "extract_constraints",
            {"query": query, "metadata": metadata or {}},
        )
        
        # Wait for response
        for _ in range(100):  # Timeout after 10 seconds
            response = await self.bus.get_message_by_id(msg.message_id)
            if response and response.status == "completed":
                return response.result
            await asyncio.sleep(0.1)
        
        logger.error("Constraint extraction timeout")
        return {}
    
    async def _request_research(self, query: str, metadata: dict | None) -> str:
        """Request research from researcher agent."""
        msg = await self.send_message(
            "researcher",
            "research",
            {"query": query, "metadata": metadata or {}},
        )
        
        # Wait for response
        for _ in range(100):
            response = await self.bus.get_message_by_id(msg.message_id)
            if response and response.status == "completed":
                return response.result.get("research_notes", "")
            await asyncio.sleep(0.1)
        
        logger.error("Research timeout")
        return ""

    async def _request_flights(self, query: str, metadata: dict | None) -> str:
        msg = await self.send_message(
            "flight_agent",
            "search_flights",
            {"query": query, "metadata": metadata or {}},
        )
        for _ in range(100):
            response = await self.bus.get_message_by_id(msg.message_id)
            if response and response.status == "completed":
                return response.result.get("flight_notes", "")
            await asyncio.sleep(0.1)
        logger.error("Flight search timeout")
        return ""

    async def _request_hotels(self, query: str, metadata: dict | None) -> str:
        msg = await self.send_message(
            "hotel_agent",
            "search_hotels",
            {"query": query, "metadata": metadata or {}},
        )
        for _ in range(100):
            response = await self.bus.get_message_by_id(msg.message_id)
            if response and response.status == "completed":
                return response.result.get("hotel_notes", "")
            await asyncio.sleep(0.1)
        logger.error("Hotel search timeout")
        return ""

    async def _request_weather(self, query: str, metadata: dict | None) -> str:
        msg = await self.send_message(
            "weather_agent",
            "fetch_weather",
            {"query": query, "metadata": metadata or {}},
        )
        for _ in range(100):
            response = await self.bus.get_message_by_id(msg.message_id)
            if response and response.status == "completed":
                return response.result.get("weather_notes", "")
            await asyncio.sleep(0.1)
        logger.error("Weather lookup timeout")
        return ""
    
    async def _request_plan(
        self,
        constraints: dict,
        research_notes: str,
        *,
        query: str = "",
        feedback: str = "",
        solver_llm_kwargs: dict[str, Any] | None = None,
    ) -> str:
        """Request plan generation from solver agent."""
        msg = await self.send_message(
            "solver",
            "generate_plan",
            {
                "constraints": constraints,
                "research_notes": research_notes,
                "feedback": feedback,
                "query": query,
                "solver_llm_kwargs": solver_llm_kwargs or {},
            },
        )
        
        # Wait for response
        for _ in range(100):
            response = await self.bus.get_message_by_id(msg.message_id)
            if response and response.status == "completed":
                return response.result.get("plan", "")
            await asyncio.sleep(0.1)
        
        logger.error("Plan generation timeout")
        return ""
    
    async def _request_audit(self, plan: str, constraints: dict) -> dict:
        """Request audit from auditor agent."""
        msg = await self.send_message(
            "auditor",
            "audit_plan",
            {"plan": plan, "constraints": constraints},
        )
        
        # Wait for response
        for _ in range(100):
            response = await self.bus.get_message_by_id(msg.message_id)
            if response and response.status == "completed":
                return response.result
            await asyncio.sleep(0.1)
        
        logger.error("Audit timeout")
        return {"passed": False, "feedback": "Audit timeout"}
    
    async def process(self, message: AgentMessage) -> dict[str, Any]:
        """Not used - coordinator uses orchestrate() directly."""
        return {}
