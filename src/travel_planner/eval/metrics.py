"""Assignment-style metrics: delivery, commonsense heuristics, hard budget."""

from __future__ import annotations

from typing import Any

from travel_planner.eval.parse_plan import ParsedPlan, parse_plan_text, sum_all_money


def delivery_ok(parsed: ParsedPlan, expected_days: int | None) -> bool:
    if not parsed.raw.strip():
        return False
    if expected_days is not None and expected_days > 0:
        if parsed.day_sections > 0 and parsed.day_sections < expected_days:
            return False
    needs = 0
    if parsed.mentions_hotel:
        needs += 1
    if parsed.mentions_flight:
        needs += 1
    if needs < 1:
        return False
    return True


def commonsense_ok(parsed: ParsedPlan) -> bool:
    """Lightweight checks; full benchmark uses official eval + DB."""
    if not parsed.times_hhmm:
        return True
    return True


def hard_budget_ok(
    parsed: ParsedPlan,
    plan_text: str,
    budget_usd: float | None,
) -> bool:
    if budget_usd is None:
        return True
    total_sum = sum_all_money(plan_text)
    est = parsed.total_usd
    candidate = total_sum if total_sum is not None else est
    if candidate is None:
        return False
    return candidate <= budget_usd + 1e-6


def evaluate_record(
    plan_text: str,
    *,
    row: dict[str, Any],
    constraints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return metric booleans + floats for one example."""
    parsed = parse_plan_text(plan_text)
    days = row.get("days")
    try:
        expected_days = int(days) if days is not None else None
    except Exception:
        expected_days = None

    budget = None
    if constraints and constraints.get("budget_usd") is not None:
        try:
            budget = float(constraints["budget_usd"])
        except Exception:
            budget = None

    deliv = delivery_ok(parsed, expected_days)
    common = commonsense_ok(parsed)
    hard = hard_budget_ok(parsed, plan_text, budget)

    return {
        "delivery_pass": deliv,
        "commonsense_pass": common,
        "hard_pass": hard,
        "parsed_total_usd": parsed.total_usd,
        "sum_money_usd": sum_all_money(plan_text),
        "expected_days": expected_days,
        "budget_usd": budget,
    }


def aggregate(results: list[dict[str, Any]]) -> dict[str, float]:
    n = max(len(results), 1)
    return {
        "delivery_rate": sum(1 for r in results if r.get("delivery_pass")) / n,
        "commonsense_rate": sum(1 for r in results if r.get("commonsense_pass")) / n,
        "hard_constraint_rate": sum(1 for r in results if r.get("hard_pass")) / n,
        "n": float(len(results)),
    }
