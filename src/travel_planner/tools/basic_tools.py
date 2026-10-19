"""Basic tools for agents: Calculator (arithmetic + Frankfurter FX) and Search."""

from __future__ import annotations

import logging
import re

import httpx

from travel_planner.tools.search import web_search

logger = logging.getLogger(__name__)

# frankfurter.app redirects to frankfurter.dev (see https://www.frankfurter.app/)
FRANKFURTER_LATEST = "https://api.frankfurter.dev/v1/latest"

# "450 USD to EGP" or "100 usd into EUR"
_RE_CONVERT = re.compile(
    r"(?i)^\s*(\d+(?:\.\d+)?)\s+([a-z]{3})\s+(?:to|into)\s+([a-z]{3})\s*$"
)


def frankfurter_convert(amount: float, from_currency: str, to_currency: str) -> tuple[float, str] | tuple[None, str]:
    """Convert amount using ECB rates via Frankfurter (no API key)."""
    fc = from_currency.upper().strip()
    tc = to_currency.upper().strip()
    if fc == tc:
        return amount, tc

    try:
        r = httpx.get(
            FRANKFURTER_LATEST,
            params={"from": fc, "to": tc},
            timeout=30.0,
            follow_redirects=True,
        )
        r.raise_for_status()
        data = r.json()
        rates = data.get("rates") or {}
        if tc not in rates:
            return None, f"{tc} not supported by Frankfurter (ECB currencies only)"
        converted = round(amount * float(rates[tc]), 2)
        return converted, tc
    except Exception as e:
        logger.warning("Frankfurter conversion failed: %s", e)
        return None, str(e)


class CalculatorTool:
    """Tool for arithmetic and currency conversion (Frankfurter API)."""

    name = "calculator"
    description = "Perform arithmetic calculations or currency conversion via Frankfurter"

    @staticmethod
    def run(expression: str) -> str:
        """Evaluate arithmetic or convert currencies (e.g. `450 USD to EGP`)."""
        expr = (expression or "").strip()
        try:
            logger.info("[TOOL] calculator: %s", expr)

            m = _RE_CONVERT.match(expr)
            if m:
                amount = float(m.group(1))
                fr = m.group(2).upper()
                to = m.group(3).upper()
                val, info = frankfurter_convert(amount, fr, to)
                if val is None:
                    msg = f"currency error: {info}"
                    logger.warning(msg)
                    return msg
                result = f"{val} {to}"
                logger.info("Frankfurter result: %s", result)
                return result

            result = str(eval(expr))
            logger.info("Result: %s", result)
            return result
        except Exception as e:
            error_msg = f"error: {type(e).__name__}: {e}"
            logger.warning(error_msg)
            return error_msg


class SearchTool:
    """Tool for searching travel information."""

    name = "search"
    description = "Search for travel information"

    @staticmethod
    def run(query: str) -> str:
        """Execute a web search for travel information."""
        try:
            logger.info("[TOOL] search: %s", query)
            result = web_search(query)
            logger.info("Found %s chars of results", len(result))
            return result
        except Exception as e:
            error_msg = f"search error: {e}"
            logger.warning(error_msg)
            return error_msg
