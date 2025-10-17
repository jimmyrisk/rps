"""Shared Prometheus metric helpers for the RPS project.

This module centralises metric registration so repeated imports across the
application, trainers, and scripts reuse the same collectors without raising
"Collector already registered" errors.
"""
from __future__ import annotations

from typing import Optional, Sequence
import logging

try:  # pragma: no cover - optional dependency in some environments
    from prometheus_client import Counter, Gauge, Histogram, Summary, REGISTRY  # type: ignore
except Exception:  # pragma: no cover - Prometheus not installed
    Counter = Gauge = Histogram = Summary = None  # type: ignore
    REGISTRY = None  # type: ignore


class _NoopMetric:
    """Fallback metric that safely ignores all operations."""

    def labels(self, *args, **kwargs):  # pragma: no cover - trivial
        return self

    def inc(self, *args, **kwargs):  # pragma: no cover - trivial
        return self

    def observe(self, *args, **kwargs):  # pragma: no cover - trivial
        return self

    def set(self, *args, **kwargs):  # pragma: no cover - trivial
        return self


_logger = logging.getLogger(__name__)


_STANDARD_DIFFICULTY_LABELS = {"normal", "standard", "std", ""}
_EASY_DIFFICULTY_LABELS = {"easy", "easy_mode"}  # Legacy labels; retained for historical/regression data only
_UNKNOWN_DIFFICULTY_LABELS_LOGGED: set[str] = set()


def _canonical_difficulty_label(raw_value: Optional[str]) -> str:
    """Normalize difficulty labels for metrics.

    Historically we accepted both "normal" and "standard" (from policy metadata).
    This helper collapses those into the "standard" label used by Grafana panels
    while keeping "easy" intact. Unknown values default to "standard" but are
    logged once to aid debugging.
    """

    normalized = (raw_value or "").strip().lower()
    if normalized in _STANDARD_DIFFICULTY_LABELS:
        return "standard"
    if normalized in _EASY_DIFFICULTY_LABELS:
        return "easy"

    if normalized not in _UNKNOWN_DIFFICULTY_LABELS_LOGGED:
        _UNKNOWN_DIFFICULTY_LABELS_LOGGED.add(normalized)
        _logger.warning("Unknown difficulty_mode '%s'; defaulting to 'standard' for metrics", raw_value)
    return "standard"


def _get_or_create_metric(metric_cls, name: str, documentation: str, *, labelnames: Optional[Sequence[str]] = None, buckets: Optional[Sequence[float]] = None):
    """Register (or retrieve) a metric, falling back to a no-op when unavailable."""
    if metric_cls is None or REGISTRY is None:  # Prometheus dependency missing
        return _NoopMetric()

    try:
        if metric_cls is Histogram and buckets is not None:
            return metric_cls(name, documentation, labelnames=labelnames or (), buckets=buckets)  # type: ignore[call-arg]
        return metric_cls(name, documentation, labelnames=labelnames or ())  # type: ignore[call-arg]
    except ValueError:
        existing = getattr(REGISTRY, "_names_to_collectors", {}).get(name)
        if existing is None:
            raise
        return existing  # type: ignore[return-value]


def get_counter(name: str, documentation: str, labelnames: Optional[Sequence[str]] = None):
    return _get_or_create_metric(Counter, name, documentation, labelnames=labelnames)


def get_gauge(name: str, documentation: str, labelnames: Optional[Sequence[str]] = None):
    return _get_or_create_metric(Gauge, name, documentation, labelnames=labelnames)


def get_histogram(
    name: str,
    documentation: str,
    labelnames: Optional[Sequence[str]] = None,
    *,
    buckets: Optional[Sequence[float]] = None,
):
    return _get_or_create_metric(Histogram, name, documentation, labelnames=labelnames, buckets=buckets)


def get_summary(name: str, documentation: str, labelnames: Optional[Sequence[str]] = None):
    return _get_or_create_metric(Summary, name, documentation, labelnames=labelnames)


# Project-wide gauges and helpers -------------------------------------------------

_DATA_DRIFT_GAUGE = get_gauge(
    "rps_data_drift_score",
    "Composite drift score (e.g., PSI) for recent gameplay data",
    ["scope"],
)

# Action-level metrics (per round/move)
_ACTION_PREDICTIONS_COUNTER = get_counter(
    "rps_action_predictions_total",
    "Total action predictions made",
    ["policy", "model_type", "difficulty", "move"]
)

_ACTION_WINS_COUNTER = get_counter(
    "rps_action_wins_total",
    "Total actions won by bot",
    ["policy", "model_type", "difficulty"]
)

_ACTION_LOSSES_COUNTER = get_counter(
    "rps_action_losses_total", 
    "Total actions lost by bot",
    ["policy", "model_type", "difficulty"]
)

_ACTION_TIES_COUNTER = get_counter(
    "rps_action_ties_total",
    "Total action ties", 
    ["policy", "model_type", "difficulty"]
)

# Game-level metrics (complete games to 10 points)
_GAME_WINS_COUNTER = get_counter(
    "rps_game_wins_total",
    "Total complete games won by bot",
    ["policy", "model_type", "difficulty"]
)

_GAME_LOSSES_COUNTER = get_counter(
    "rps_game_losses_total",
    "Total complete games lost by bot", 
    ["policy", "model_type", "difficulty"]
)

_GAMES_TOTAL_COUNTER = get_counter(
    "rps_games_total",
    "Total games completed",
    ["policy", "model_type", "difficulty"]
)

# Training event tracking
_TRAINING_COMPLETED_COUNTER = get_counter(
    "rps_training_completed_total",
    "Training runs completed",
    ["model_type", "status"]
)

_TRAINING_DURATION_HISTOGRAM = get_histogram(
    "rps_training_duration_seconds",
    "Training duration in seconds",
    ["model_type"],
    buckets=[10, 30, 60, 120, 300, 600, 1800, 3600]
)

# Model alias-specific metrics (for 12 models: 3 types × 4 aliases)
_MODEL_ALIAS_ACCURACY_GAUGE = get_gauge(
    "rps_model_accuracy",
    "Action prediction accuracy by model and alias",
    ["model", "alias"]
)

_MODEL_ALIAS_PREDICTIONS_COUNTER = get_counter(
    "rps_model_predictions_by_alias_total",
    "Total predictions by model and alias",
    ["model", "alias"]
)

_MODEL_ALIAS_CORRECT_PREDICTIONS_COUNTER = get_counter(
    "rps_model_correct_predictions_by_alias_total",
    "Correct predictions by model and alias",
    ["model", "alias"]
)

_MODEL_ALIAS_GAME_WINS_COUNTER = get_counter(
    "rps_model_game_wins_by_alias_total",
    "Game wins by model, alias, and difficulty (Production/B only)",
    ["model", "alias", "difficulty"]
)

_MODEL_ALIAS_GAME_LOSSES_COUNTER = get_counter(
    "rps_model_game_losses_by_alias_total",
    "Game losses by model, alias, and difficulty (Production/B only)",
    ["model", "alias", "difficulty"]
)

_MODEL_ALIAS_GAMES_TOTAL_COUNTER = get_counter(
    "rps_model_games_by_alias_total",
    "Total games by model, alias, and difficulty (Production/B only)",
    ["model", "alias", "difficulty"]
)


# Auto-promotion metrics ------------------------------------------------------
_PROMOTION_TEST_STAT_GAUGE = get_gauge(
    "rps_model_promotion_z_statistic",
    "Z statistic from the Production vs B hypothesis test",
    ["model"],
)

_PROMOTION_P_VALUE_GAUGE = get_gauge(
    "rps_model_promotion_p_value",
    "One-sided p-value from the Production vs B hypothesis test",
    ["model"],
)

_PROMOTION_DECISIONS_COUNTER = get_counter(
    "rps_model_promotion_events_total",
    "Auto-promotion events grouped by decision type",
    ["model", "decision"],
)

_PROMOTION_ALIAS_REORDER_COUNTER = get_counter(
    "rps_model_alias_reorders_total",
    "Count of alias reorder operations executed for challenger slots",
    ["model"],
)

_PROMOTION_ALIAS_RANK_GAUGE = get_gauge(
    "rps_model_alias_rank",
    "Current rank assignment for non-production aliases (1=best)",
    ["model", "alias"],
)

_PROMOTION_DECISION_STATE_GAUGE = get_gauge(
    "rps_model_promotion_decision_state",
    "Latest recorded auto-promotion decision (1 active, 0 otherwise)",
    ["model", "decision"],
)

_PROMOTION_GAMES_SINCE_SWAP_GAUGE = get_gauge(
    "rps_model_games_since_production_swap",
    "Completed games for Production/B since the most recent swap",
    ["model", "alias"],
)

_PROMOTION_CYCLES_SINCE_SWAP_GAUGE = get_gauge(
    "rps_model_promotion_cycles_since_swap",
    "Auto-promotion cycles evaluated since the last Production/B swap",
    ["model"],
)


def set_data_drift_score(score: float, scope: str = "overall") -> None:
    """Update the shared data drift gauge if Prometheus is available."""
    try:
        _DATA_DRIFT_GAUGE.labels(scope=scope).set(float(score))
    except Exception:  # pragma: no cover - safe fallback when prom missing
        pass

def record_policy_prediction(policy: str, model_type: str, difficulty_mode: str, move: str) -> None:
    """Record a policy prediction for Grafana metrics."""
    try:
        difficulty = _canonical_difficulty_label(difficulty_mode)
        _ACTION_PREDICTIONS_COUNTER.labels(
            policy=policy,
            model_type=model_type,
            difficulty=difficulty,
            move=move
        ).inc()
    except Exception:
        pass

def record_policy_action_result(policy: str, model_type: str, difficulty_mode: str, result: str) -> None:
    """Record action-level result (win/loss/tie) for a policy.
    
    Maps difficulty_mode to difficulty for dashboard compatibility.
    """
    import logging
    logger = logging.getLogger(__name__)
    try:
        # Map to dashboard-compatible difficulty label
        difficulty = _canonical_difficulty_label(difficulty_mode)
        
        if result == "win":
            _ACTION_WINS_COUNTER.labels(
                policy=policy,
                model_type=model_type,
                difficulty=difficulty
            ).inc()
            logger.info(f"Incremented action win counter: {policy}/{model_type}/{difficulty}")
        elif result == "loss":
            _ACTION_LOSSES_COUNTER.labels(
                policy=policy,
                model_type=model_type,
                difficulty=difficulty
            ).inc()
            logger.info(f"Incremented action loss counter: {policy}/{model_type}/{difficulty}")
        elif result == "tie":
            _ACTION_TIES_COUNTER.labels(
                policy=policy,
                model_type=model_type,
                difficulty=difficulty
            ).inc()
            logger.info(f"Incremented action tie counter: {policy}/{model_type}/{difficulty}")
    except Exception as e:
        logger.error(f"Failed to record action result: {e}", exc_info=True)

def record_policy_game_result(policy: str, model_type: str, difficulty_mode: str, result: str) -> None:
    """Record game-level result (win/loss) for a complete game to 10 points.
    
    Maps difficulty_mode to difficulty for dashboard compatibility.
    """
    try:
        # Map to dashboard-compatible difficulty label
        difficulty = _canonical_difficulty_label(difficulty_mode)
        
        if result == "win":
            _GAME_WINS_COUNTER.labels(
                policy=policy,
                model_type=model_type,
                difficulty=difficulty
            ).inc()
        elif result == "loss":
            _GAME_LOSSES_COUNTER.labels(
                policy=policy,
                model_type=model_type,
                difficulty=difficulty
            ).inc()
    except Exception:
        pass


def record_game_completion(policy: str, model_type: str, difficulty_mode: str) -> None:
    """Record that a game was completed (regardless of outcome).
    
    Maps difficulty_mode ('normal'/'easy') to difficulty label ('standard'/'easy')
    for dashboard compatibility.
    """
    try:
        # Map difficulty_mode to difficulty for dashboard compatibility
        # Internal code uses "normal" but dashboard expects "standard"
        difficulty = _canonical_difficulty_label(difficulty_mode)
        
        _GAMES_TOTAL_COUNTER.labels(
            policy=policy,
            model_type=model_type,
            difficulty=difficulty
        ).inc()
    except Exception:
        pass


def record_training_completion(model_type: str, status: str, duration_seconds: float = None) -> None:
    """Record training job completion event."""
    try:
        _TRAINING_COMPLETED_COUNTER.labels(
            model_type=model_type,
            status=status  # "success" or "failure"
        ).inc()
        if duration_seconds is not None:
            _TRAINING_DURATION_HISTOGRAM.labels(model_type=model_type).observe(duration_seconds)
    except Exception:
        pass


def record_model_alias_prediction(model: str, alias: str, correct: bool = None) -> None:
    """Record a prediction for a specific model alias (e.g., xgboost@Production)."""
    try:
        _MODEL_ALIAS_PREDICTIONS_COUNTER.labels(model=model, alias=alias).inc()
        if correct is not None:
            # Initialize the counter with 0 if needed (first time), then increment if correct
            # This ensures the metric is exported even if all predictions are wrong
            if correct:
                _MODEL_ALIAS_CORRECT_PREDICTIONS_COUNTER.labels(model=model, alias=alias).inc()
            else:
                # Initialize counter at 0 if this is first wrong prediction
                # (Prometheus won't export metrics that were never touched)
                _MODEL_ALIAS_CORRECT_PREDICTIONS_COUNTER.labels(model=model, alias=alias).inc(0)
    except Exception:
        pass


def record_model_alias_game_result(model: str, alias: str, difficulty_mode: str, won: bool) -> None:
    """Record game result for Production/B aliases only."""
    try:
        if alias in ["Production", "B"]:
            difficulty = _canonical_difficulty_label(difficulty_mode)
            # Always increment total games
            _MODEL_ALIAS_GAMES_TOTAL_COUNTER.labels(model=model, alias=alias, difficulty=difficulty).inc()
            
            # Increment wins or losses
            if won:
                _MODEL_ALIAS_GAME_WINS_COUNTER.labels(model=model, alias=alias, difficulty=difficulty).inc()
            else:
                _MODEL_ALIAS_GAME_LOSSES_COUNTER.labels(model=model, alias=alias, difficulty=difficulty).inc()
    except Exception:
        pass


def update_model_alias_accuracy(model: str, alias: str, accuracy: float) -> None:
    """Update accuracy gauge for a specific model alias."""
    try:
        _MODEL_ALIAS_ACCURACY_GAUGE.labels(model=model, alias=alias).set(accuracy)
    except Exception:
        pass


def record_promotion_evaluation(
    *,
    model: str,
    decision: str,
    z_statistic: float | None,
    p_value: float | None,
) -> None:
    """Persist the outcome of an auto-promotion hypothesis test."""

    try:
        if z_statistic is not None:
            _PROMOTION_TEST_STAT_GAUGE.labels(model=model).set(float(z_statistic))
        if p_value is not None:
            _PROMOTION_P_VALUE_GAUGE.labels(model=model).set(float(p_value))
        _PROMOTION_DECISIONS_COUNTER.labels(model=model, decision=decision).inc()

        for label in ["swap_production_b", "retain_production", "insufficient_data", "alias_reorder"]:
            value = 1.0 if label == decision else 0.0
            _PROMOTION_DECISION_STATE_GAUGE.labels(model=model, decision=label).set(value)
    except Exception:
        pass


def record_alias_reorder(model: str, performed: bool) -> None:
    """Increment the reorder counter when challenger aliases are reshuffled."""

    if not performed:
        return
    try:
        _PROMOTION_ALIAS_REORDER_COUNTER.labels(model=model).inc()
    except Exception:
        pass


def update_alias_rank(model: str, alias: str, rank: int) -> None:
    """Set the rank gauge for a challenger alias (1=best)."""

    try:
        _PROMOTION_ALIAS_RANK_GAUGE.labels(model=model, alias=alias).set(float(rank))
    except Exception:
        pass


def update_games_since_swap(model: str, alias: str, games_since_swap: int) -> None:
    """Update the ticker tracking games since the last Production/B swap."""

    try:
        _PROMOTION_GAMES_SINCE_SWAP_GAUGE.labels(model=model, alias=alias).set(float(max(games_since_swap, 0)))
    except Exception:
        pass


def update_cycles_since_swap(model: str, cycles_since_swap: int) -> None:
    """Update the ticker for auto-promotion cycles since the last swap."""

    try:
        _PROMOTION_CYCLES_SINCE_SWAP_GAUGE.labels(model=model).set(float(max(cycles_since_swap, 0)))
    except Exception:
        pass


__all__ = [
    "get_counter",
    "get_gauge", 
    "get_histogram",
    "get_summary",
    "set_data_drift_score",
    "record_policy_prediction",
    "record_policy_action_result", 
    "record_policy_game_result",
    "record_game_completion",
    "record_training_completion",
    "record_model_alias_prediction",
    "record_model_alias_game_result",
    "update_model_alias_accuracy",
    "record_alias_reorder",
    "record_promotion_evaluation",
    "update_alias_rank",
    "update_games_since_swap",
    "update_cycles_since_swap",
]
