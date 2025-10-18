"""Utilities for automated model promotion decisions.

This module centralises the statistical tests and ranking logic used by the
post-training auto-promotion pipeline.  Keeping the maths in one place makes it
simpler to unit-test the decision rules and reuse them in both batch scripts and
FastAPI endpoints.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import math


@dataclass(frozen=True)
class AliasGameStats:
    """Observed win/loss counts for a specific model alias."""

    wins: int
    total: int

    @property
    def win_rate(self) -> float:
        if self.total <= 0:
            return 0.0
        return self.wins / self.total


@dataclass(frozen=True)
class AliasAccuracyStats:
    """Correct / total action predictions for a specific model alias."""

    correct: int
    total: int

    @property
    def accuracy(self) -> float:
        if self.total <= 0:
            return 0.0
        return self.correct / self.total


@dataclass(frozen=True)
class PromotionTestResult:
    """Outcome of the Production vs B hypothesis test."""

    z_statistic: Optional[float]
    p_value: Optional[float]
    decision: str
    reason: str

    def should_swap(self) -> bool:
        return self.decision == "swap_production_b"


@dataclass(frozen=True)
class RankedAlias:
    """Represents an alias ordered by observed accuracy."""

    alias: str
    accuracy: float
    total_predictions: int
    source_alias: str


_ALLOWED_MODEL_TYPES: Sequence[str] = ("xgboost", "feedforward_nn", "multinomial_logistic")
_ALIAS_REORDER_TARGETS: Sequence[str] = ("B", "shadow1", "shadow2")


def two_sided_to_one_sided_p(z_value: float) -> float:
    """Convert a z-score to a one-sided (upper tail) p-value."""
    # Survival function of standard normal without SciPy: 0.5 * erfc(z / sqrt(2))
    return 0.5 * math.erfc(z_value / math.sqrt(2.0))


def evaluate_promotion_decision(
    b_stats: AliasGameStats,
    production_stats: AliasGameStats,
    *,
    alpha: float = 0.2,
    minimum_total_games: int = 3,
) -> PromotionTestResult:
    """Compare Production and B aliases using a simplified win-rate rule.

    We still compute the classical z-test for telemetry, but the decision logic
    now follows the proof-of-concept requirement: once both aliases have logged
    at least ``minimum_total_games`` completed games, whichever alias has the
    higher empirical win rate is selected. Ties break in favour of Production.
    """

    total_observations = b_stats.total + production_stats.total
    if b_stats.total < minimum_total_games or production_stats.total < minimum_total_games:
        return PromotionTestResult(
            z_statistic=None,
            p_value=None,
            decision="insufficient_data",
            reason=(
                "Not enough completed games for promotion comparison "
                f"(Production={production_stats.total}, B={b_stats.total}, "
                f"minimum={minimum_total_games})"
            ),
        )

    if total_observations == 0:
        return PromotionTestResult(
            z_statistic=None,
            p_value=None,
            decision="insufficient_data",
            reason="No completed games observed since last restart",
        )

    pooled_successes = b_stats.wins + production_stats.wins
    pooled_rate = pooled_successes / total_observations if total_observations > 0 else 0.0
    variance_term = pooled_rate * (1.0 - pooled_rate)
    z_value: Optional[float] = None
    p_value: Optional[float] = None
    variance_note: Optional[str] = None

    if variance_term > 0:
        std_error = math.sqrt(variance_term * ((1.0 / b_stats.total) + (1.0 / production_stats.total)))
        if std_error > 0:
            z_value = (b_stats.win_rate - production_stats.win_rate) / std_error
            p_value = two_sided_to_one_sided_p(z_value)
        else:
            variance_note = "standard error non-positive"
    else:
        variance_note = "pooled variance is zero (all wins or losses)"

    b_win_rate = b_stats.win_rate
    production_win_rate = production_stats.win_rate

    if b_win_rate > production_win_rate:
        decision = "swap_production_b"
        headline = "B win rate higher than Production"
    elif b_win_rate < production_win_rate:
        decision = "retain_production"
        headline = "Production win rate is higher than B"
    else:
        decision = "retain_production"
        headline = "Win rates tied; defaulting to Production"

    reason_parts = [
        headline,
        (
            "PoC win-rate rule applied after at least "
            f"{minimum_total_games} games per alias"
        ),
        (
            f"B={b_win_rate:.1%} ({b_stats.wins}/{b_stats.total}), "
            f"Production={production_win_rate:.1%} ({production_stats.wins}/{production_stats.total})"
        ),
    ]

    if z_value is not None and p_value is not None:
        reason_parts.append(f"z={z_value:.2f}, p={p_value:.4f}")
        if p_value < alpha:
            reason_parts.append(f"p<{alpha} historically flagged swaps")
    elif variance_note:
        reason_parts.append(f"z/p unavailable: {variance_note}")

    reason = " | ".join(reason_parts)

    return PromotionTestResult(
        z_statistic=z_value,
        p_value=p_value,
        decision=decision,
        reason=reason,
    )


def rank_aliases_by_accuracy(
    accuracy_map: Dict[str, AliasAccuracyStats],
    *,
    preferred_order: Sequence[str] = _ALIAS_REORDER_TARGETS,
) -> List[RankedAlias]:
    """Return aliases ordered by action accuracy (highest first).

    ``accuracy_map`` should contain entries for at least the aliases named in
    ``preferred_order``.  Entries missing from the map are treated as having zero
    accuracy and zero predictions, ensuring deterministic ordering.
    """

    ranked: List[RankedAlias] = []
    for alias in preferred_order:
        stats = accuracy_map.get(alias, AliasAccuracyStats(correct=0, total=0))
        ranked.append(
            RankedAlias(
                alias=alias,
                accuracy=stats.accuracy,
                total_predictions=stats.total,
                source_alias=alias,
            )
        )

    # Allow callers to pass in additional aliases (e.g., Production) and still
    # surface them in the ordering when explicitly referenced in preferred_order.

    ranked.sort(
        key=lambda item: (
            item.accuracy,
            item.total_predictions,
            -preferred_order.index(item.alias) if item.alias in preferred_order else -len(preferred_order),
        ),
        reverse=True,
    )
    return ranked


def normalise_model_type(model_type: str) -> str:
    value = model_type.strip().lower()
    if value not in _ALLOWED_MODEL_TYPES:
        raise ValueError(f"Unsupported model_type '{model_type}'")
    return value


def calculate_accuracy(correct: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return correct / total


__all__ = [
    "AliasAccuracyStats",
    "AliasGameStats",
    "PromotionTestResult",
    "RankedAlias",
    "calculate_accuracy",
    "evaluate_promotion_decision",
    "normalise_model_type",
    "rank_aliases_by_accuracy",
]
