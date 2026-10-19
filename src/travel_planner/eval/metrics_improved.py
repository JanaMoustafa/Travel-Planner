"""Improved evaluation metrics with ground truth comparison.

This module was added to fix the weak evaluation requirement.

Previously, evaluation was heuristic-based (regex parsing, no ground truth).
This adds:
- ROUGE-L comparison to annotated_plan (reference)
- Constraint adherence scoring
- Better structured validation
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any

from travel_planner.eval.parse_plan import ParsedPlan, parse_plan_text, sum_all_money

logger = logging.getLogger(__name__)

try:
    from rouge_score import rouge_scorer

    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    logger.warning("rouge_score not installed. ROUGE metrics will be skipped.")

try:
    from sentence_transformers import SentenceTransformer, util as st_util

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not installed. Semantic similarity will be skipped.")


def compute_bleu_score(generated: str, reference: str) -> float:
    """BLEU score vs reference plan (Hugging Face ``evaluate``)."""
    if not generated.strip() or not reference.strip():
        return 0.0
    try:
        import evaluate

        bleu = evaluate.load("bleu")
        out = bleu.compute(predictions=[generated.strip()], references=[[reference.strip()]])
        return float(out.get("bleu") or 0.0)
    except ImportError:
        logger.warning("evaluate not installed. BLEU will be 0.")
        return 0.0
    except Exception as exc:
        logger.warning("BLEU computation failed: %s", exc)
        return 0.0


_RE_MULTI_SPACE = re.compile(r"\s+")
_RE_NUMBER = re.compile(r"\b(\d{1,3})(?:,(\d{3}))+\b")
_RE_PUNCT = re.compile(r"[\u2013\u2014\u2022•·,;:!?()\[\]{}<>]")


def normalize_text(text: str) -> str:
    """Normalize text before metric computation."""
    if not text:
        return ""
    normalized = text.lower().strip()
    normalized = _RE_NUMBER.sub(lambda match: match.group(0).replace(",", ""), normalized)
    normalized = normalized.replace("\n", " ")
    normalized = _RE_PUNCT.sub(" ", normalized)
    normalized = _RE_MULTI_SPACE.sub(" ", normalized)
    return normalized.strip()


@lru_cache(maxsize=1)
def _get_sentence_encoder() -> "SentenceTransformer":
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        raise RuntimeError("sentence-transformers is not available")
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def delivery_ok(parsed: ParsedPlan, expected_days: int | None) -> bool:
    """Check if plan has basic delivery structure."""
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
    """Check if plan has reasonable structure (not just stub)."""
    if not parsed.raw.strip():
        return False
    
    # Must have some content (not just a title)
    if len(parsed.raw.strip()) < 100:
        return False
    
    # Must have day sections
    if parsed.day_sections < 1:
        return False
    
    # Must mention at least one transport OR accommodation option
    has_transport = parsed.mentions_flight or "train" in parsed.raw.lower() or "bus" in parsed.raw.lower()
    has_accommodation = parsed.mentions_hotel
    
    if not (has_transport or has_accommodation):
        return False
    
    return True


def compute_structure_score(parsed: ParsedPlan, plan_text: str, expected_days: int | None = None) -> float:
    """Score itinerary structure based on day sections and expected detail."""
    if not parsed.raw.strip():
        return 0.0

    components: list[float] = []
    components.append(1.0 if parsed.day_sections >= 1 else 0.0)
    if expected_days and expected_days > 0:
        components.append(min(parsed.day_sections / expected_days, 1.0))
    components.append(1.0 if parsed.mentions_hotel else 0.0)
    components.append(1.0 if parsed.mentions_flight else 0.0)
    components.append(1.0 if sum_all_money(plan_text) is not None else 0.0)
    return sum(components) / len(components)


def compute_semantic_similarity(generated: str, reference: str) -> float:
    """Compute cosine similarity between plan texts using sentence embeddings."""
    if not generated or not reference:
        return 0.0
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        return 0.0
    try:
        model = _get_sentence_encoder()
        emb_gen = model.encode(normalize_text(generated), convert_to_tensor=True, normalize_embeddings=True)
        emb_ref = model.encode(normalize_text(reference), convert_to_tensor=True, normalize_embeddings=True)
        return float(st_util.cos_sim(emb_gen, emb_ref).item())
    except Exception as exc:
        logger.warning(f"Semantic similarity failed: {exc}")
        return 0.0


def compute_rouge_scores(generated: str, reference: str) -> dict[str, float]:
    """Compute reference-based ROUGE-1 and ROUGE-L."""
    if not ROUGE_AVAILABLE or not reference or not generated:
        return {"rouge1_fmeasure": 0.0, "rougeL_fmeasure": 0.0}

    try:
        scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
        scores = scorer.score(normalize_text(reference), normalize_text(generated))
        return {
            "rouge1_fmeasure": float(scores["rouge1"].fmeasure),
            "rougeL_fmeasure": float(scores["rougeL"].fmeasure),
        }
    except Exception as exc:
        logger.warning(f"ROUGE computation failed: {exc}")
        return {"rouge1_fmeasure": 0.0, "rougeL_fmeasure": 0.0}


def hard_budget_ok(
    parsed: ParsedPlan,
    plan_text: str,
    budget_usd: float | None,
) -> bool:
    """Check if plan respects hard budget constraint."""
    if budget_usd is None:
        return True
    total_sum = sum_all_money(plan_text)
    est = parsed.total_usd
    candidate = total_sum if total_sum is not None else est
    if candidate is None:
        return False
    return candidate <= budget_usd + 1e-6


def compute_rouge_similarity(generated: str, reference: str) -> float:
    """Compute ROUGE-L F1 score between generated and reference plans.
    
    Args:
        generated: Generated travel plan
        reference: Reference/annotated travel plan
        
    Returns:
        ROUGE-L F1 score (0-1), or 0.0 if ROUGE unavailable
    """
    if not ROUGE_AVAILABLE or not reference or not generated:
        return 0.0
    
    try:
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = scorer.score(reference, generated)
        return scores["rougeL"].fmeasure
    except Exception as e:
        logger.warning(f"ROUGE computation failed: {e}")
        return 0.0


def compute_constraint_adherence(
    generated: str, constraints: dict[str, Any] | None
) -> float:
    """Compute % of constraints that appear in generated plan.
    
    Args:
        generated: Generated travel plan text
        constraints: Constraint dict with keys like budget_usd, destination_city, etc.
        
    Returns:
        Adherence rate (0-1)
    """
    if not constraints:
        return 1.0
    
    constraint_mentions = 0
    total_constraints = 0
    generated_lower = generated.lower()
    
    for key, val in constraints.items():
        if val is None or val == "":
            continue
        
        total_constraints += 1
        val_str = str(val).lower()
        
        # For destination_city, check if it appears
        if key == "destination_city" and val_str in generated_lower:
            constraint_mentions += 1
        # For origin_city
        elif key == "origin_city" and val_str in generated_lower:
            constraint_mentions += 1
        # For budget, check if costs mentioned
        elif key == "budget_usd":
            total = sum_all_money(generated)
            if total is not None and total <= float(val) + 100:
                constraint_mentions += 1
        # For generic keywords
        elif key == "local_rules" and val_str in generated_lower:
            constraint_mentions += 1
    
    return (constraint_mentions / total_constraints) if total_constraints > 0 else 1.0


def compute_constraint_score(
    generated: str,
    *,
    row: dict[str, Any],
    constraints: dict[str, Any] | None = None,
) -> float:
    """Score how well the plan preserves destination, origin, days, budget, and local rules."""
    normalized = normalize_text(generated)
    values: list[float] = []

    destination = str((constraints or {}).get("destination_city") or row.get("dest") or "").strip().lower()
    if destination:
        values.append(1.0 if destination in normalized else 0.0)

    origin = str((constraints or {}).get("origin_city") or row.get("org") or "").strip().lower()
    if origin:
        values.append(1.0 if origin in normalized else 0.0)

    expected_days = row.get("days") or (constraints or {}).get("days")
    if expected_days:
        try:
            expected_days_int = int(expected_days)
            values.append(1.0 if f"day {expected_days_int}" in normalized or expected_days_int <= max(1, len(re.findall(r'(?i)\bday\s*\d+\b', generated))) else 0.0)
        except Exception:
            pass

    budget = (constraints or {}).get("budget_usd") or row.get("budget")
    if budget is not None:
        try:
            budget_float = float(budget)
            spent = sum_all_money(generated)
            if spent is not None:
                values.append(1.0 if spent <= budget_float + 100 else 0.0)
        except Exception:
            pass

    local_rules = str((constraints or {}).get("local_rules") or row.get("local_constraint") or "").strip().lower()
    if local_rules:
        values.append(1.0 if any(token in normalized for token in local_rules.split() if len(token) > 3) else 0.0)

    return sum(values) / len(values) if values else 0.0


def compute_generalization_score(plans: list[str]) -> float | None:
    """Score consistency across temperature variants for the same example."""
    plans = [p for p in plans if p and p.strip()]
    if len(plans) < 2:
        return None
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        return None
    try:
        model = _get_sentence_encoder()
        vectors = model.encode([normalize_text(p) for p in plans], convert_to_tensor=True, normalize_embeddings=True)
        pair_scores: list[float] = []
        for i in range(len(plans)):
            for j in range(i + 1, len(plans)):
                pair_scores.append(float(st_util.cos_sim(vectors[i], vectors[j]).item()))
        return sum(pair_scores) / len(pair_scores) if pair_scores else None
    except Exception as exc:
        logger.warning(f"Generalization score failed: {exc}")
        return None


def evaluate_record(
    plan_text: str,
    *,
    row: dict[str, Any],
    constraints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate one record against heuristic metrics only (backward compatible).
    
    Args:
        plan_text: Generated travel plan
        row: Original dataset row
        constraints: Extracted constraints
        
    Returns:
        Dict with metrics
    """
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
    structure_score = compute_structure_score(parsed, plan_text, expected_days)
    constraint_score = compute_constraint_score(plan_text, row=row, constraints=constraints)

    return {
        "delivery_pass": deliv,
        "commonsense_pass": common,
        "hard_pass": hard,
        "structure_score": structure_score,
        "constraint_score": constraint_score,
        "parsed_total_usd": parsed.total_usd,
        "sum_money_usd": sum_all_money(plan_text),
        "expected_days": expected_days,
        "budget_usd": budget,
    }


def evaluate_record_with_reference(
    generated_plan: str,
    reference_plan: str,
    *,
    row: dict[str, Any],
    constraints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate record with ground truth comparison (improved metrics).
    
    This is the KEY FIX for evaluation. It adds:
    - ROUGE-L comparison to reference
    - Constraint adherence scoring
    - Better commonsense check
    
    Args:
        generated_plan: Generated travel plan
        reference_plan: Reference annotated_plan from dataset
        row: Original dataset row
        constraints: Extracted constraints
        
    Returns:
        Dict with basic + improved metrics
    """
    # Get basic metrics
    basic = evaluate_record(generated_plan, row=row, constraints=constraints)
    
    # Add improved metrics
    rouge_scores = compute_rouge_scores(generated_plan, reference_plan)
    improved = {
        **basic,
        "rouge1_fmeasure": rouge_scores["rouge1_fmeasure"],
        "rouge_l_fmeasure": rouge_scores["rougeL_fmeasure"],
        "bleu_score": compute_bleu_score(generated_plan, reference_plan),
        "semantic_similarity": compute_semantic_similarity(generated_plan, reference_plan),
        "constraint_adherence": compute_constraint_adherence(generated_plan, constraints),
    }
    
    return improved


def aggregate(results: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate metrics (heuristic only)."""
    n = max(len(results), 1)
    return {
        "delivery_rate": sum(1 for r in results if r.get("delivery_pass")) / n,
        "commonsense_rate": sum(1 for r in results if r.get("commonsense_pass")) / n,
        "hard_constraint_rate": sum(1 for r in results if r.get("hard_pass")) / n,
        "n": float(len(results)),
    }


def aggregate_improved(
    results: list[dict[str, Any]],
    *,
    generalization_score: float | None = None,
) -> dict[str, float]:
    """Aggregate metrics including new ground truth comparisons.
    
    KEY FIX: This computes better aggregate metrics including ROUGE
    and constraint adherence.
    """
    n = max(len(results), 1)
    
    agg: dict[str, float] = {
        "delivery_rate": sum(1 for r in results if r.get("delivery_pass")) / n,
        "commonsense_rate": sum(1 for r in results if r.get("commonsense_pass")) / n,
        "hard_constraint_rate": sum(1 for r in results if r.get("hard_pass")) / n,
        "n": float(len(results)),
    }

    rouge1_scores = [float(r.get("rouge1_fmeasure") or 0.0) for r in results]
    rouge_l_scores = [float(r.get("rouge_l_fmeasure") or 0.0) for r in results]
    semantic_scores = [float(r.get("semantic_similarity") or 0.0) for r in results]
    constraint_scores = [float(r.get("constraint_score") or r.get("constraint_adherence") or 0.0) for r in results]
    structure_scores = [float(r.get("structure_score") or 0.0) for r in results]
    bleu_scores = [float(r.get("bleu_score") or 0.0) for r in results]

    agg["avg_rouge1"] = sum(rouge1_scores) / n
    agg["avg_rougeL"] = sum(rouge_l_scores) / n
    agg["avg_bleu"] = sum(bleu_scores) / n
    agg["avg_semantic_similarity"] = sum(semantic_scores) / n
    agg["avg_constraint_score"] = sum(constraint_scores) / n
    agg["avg_structure_score"] = sum(structure_scores) / n

    if generalization_score is not None:
        agg["generalization_score"] = generalization_score
    else:
        agg["generalization_score"] = 0.0

    # Backward-compatible aliases.
    agg["avg_rouge_l"] = agg["avg_rougeL"]
    agg["avg_constraint_adherence"] = agg["avg_constraint_score"]
    
    return agg
