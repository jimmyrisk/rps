"""Centralized configuration helpers for the RPS Quest project.

This module provides a single place to resolve environment-dependent values,
so that the app service, trainers, tests, and scripts all interpret
configuration flags consistently.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

_PROJECT_ROOT_ENV = "RPS_PROJECT_ROOT"
_DEFAULT_TRACKING_SUBDIR = "mlruns"
_DEFAULT_PRODUCTION_ALIAS = "Production"

# Metrics and training data filtering configuration
# Set to a date to exclude older data, or None to include all data
_DEFAULT_METRICS_SINCE = "2025-10-10T08:00:00Z"  # Leaderboard reset (Oct 10 2025 01:00 PDT)
_DEFAULT_TRAINING_SINCE = "2025-10-01T00:00:00Z"  # Include all historical training data (October 1+)
_DEFAULT_ROLLING_WINDOW = 500  # Last N actions/games for rolling metrics

# Easy mode difficulty tuning parameters (legacy; retained for backward compatibility)
# Controls move selection based on expected value rankings for historical sessions
_DEFAULT_EASY_WORST_PROB = 0.7   # Probability of picking worst EV move in deprecated easy mode
_DEFAULT_EASY_MIDDLE_PROB = 0.2  # Probability of picking middle EV move in deprecated easy mode
_DEFAULT_EASY_BEST_PROB = 0.1    # Probability of picking best EV move in deprecated easy mode


def _project_root() -> Path:
    """Resolve the project root directory.

    The value can be overridden with the ``RPS_PROJECT_ROOT`` environment
    variable. Otherwise it is inferred from the location of this file.
    """
    root_override = os.getenv(_PROJECT_ROOT_ENV)
    if root_override:
        return Path(root_override).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def get_mlflow_tracking_uri() -> str:
    """Return the MLflow tracking URI with a shared fallback.

    The priority order is:
    1. ``MLFLOW_TRACKING_URI`` environment variable (already respected by
       external deployments such as Kubernetes or CI).
    2. A local file-based store at ``<project-root>/mlruns``. This keeps
       local development consistent and avoids the previous mix of SQLite
       defaults vs. file stores.
    """
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if uri:
        return uri
    return f"file://{_project_root() / _DEFAULT_TRACKING_SUBDIR}"


@lru_cache(maxsize=1)
def get_mlflow_production_alias() -> str:
    """Return the registered-model alias used for production lookups."""
    return os.getenv("MLFLOW_PRODUCTION_ALIAS", _DEFAULT_PRODUCTION_ALIAS)


def set_mlflow_tracking_uri_if_needed():
    """Configure MLflow's tracking URI if it hasn't been set already."""
    uri = get_mlflow_tracking_uri()
    import mlflow  # Local import to avoid hard dependency during module import

    mlflow.set_tracking_uri(uri)
    return uri


@lru_cache(maxsize=1)
def get_metrics_since_date() -> str:
    """Return the cutoff date for metrics display.
    
    Metrics before this date are excluded from analytics queries.
    """
    return os.getenv("METRICS_SINCE_DATE", _DEFAULT_METRICS_SINCE)


@lru_cache(maxsize=1)
def get_training_data_since_date() -> str:
    """Return the cutoff date for training data.
    
    Training data before this date is excluded from model training.
    """
    return os.getenv("TRAINING_DATA_SINCE_DATE", _DEFAULT_TRAINING_SINCE)


@lru_cache(maxsize=1)
def get_rolling_window_size() -> int:
    """Return the window size for rolling metrics (e.g., last N actions/games)."""
    return int(os.getenv("METRICS_ROLLING_WINDOW", str(_DEFAULT_ROLLING_WINDOW)))


@lru_cache(maxsize=1)
def get_easy_mode_probabilities() -> tuple[float, float, float]:
    """Return legacy easy-mode probabilities for backward compatibility.

    Easy mode is no longer surfaced in the UI or automated scripts, but the
    weighting remains available so historical games and regression tests can
    replay the original behaviour without divergence.
    """
    worst = float(os.getenv("EASY_MODE_WORST_PROB", str(_DEFAULT_EASY_WORST_PROB)))
    middle = float(os.getenv("EASY_MODE_MIDDLE_PROB", str(_DEFAULT_EASY_MIDDLE_PROB)))
    best = float(os.getenv("EASY_MODE_BEST_PROB", str(_DEFAULT_EASY_BEST_PROB)))
    return (worst, middle, best)

