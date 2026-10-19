"""Pydantic models for constraints and plan drafts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class StructuredConstraints(BaseModel):
    """Structured extraction from the natural language query + dataset row fields."""

    budget_usd: float | None = Field(
        default=None,
        description="Total trip budget in USD if stated or inferable.",
    )
    origin_city: str | None = None
    destination_city: str | None = None
    start_date: str | None = Field(default=None, description="ISO-like date string if known.")
    end_date: str | None = None
    num_days: int | None = None
    num_travelers: int | None = None
    num_cities: int | None = None
    local_rules: str | None = Field(
        default=None,
        description="House rules, cuisine, room type, transport from local_constraint.",
    )
    hard_requirements: list[str] = Field(default_factory=list)
    soft_preferences: list[str] = Field(default_factory=list)
    raw_notes: str | None = None

    def model_dump_json_safe(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class PlanDraft(BaseModel):
    """Optional structured view of a draft plan (used when parsing succeeds)."""

    text: str
    estimated_total_usd: float | None = None
    days_outlined: int | None = None
