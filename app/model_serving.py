#!/usr/bin/env python3
"""
Model serving utilities for RPS application.
Supports loading and serving multiple model types with MinIO and MLflow integration.
"""
import os, json, logging, random, re, time, sys, uuid
import threading
import importlib
from pathlib import Path
from collections import defaultdict
from typing import Optional, Dict, List, Any, Tuple, Union

# Add trainer to sys.path to fix feedforward_nn model loading
trainer_path = Path(__file__).parent.parent / "trainer"
if str(trainer_path) not in sys.path:
    sys.path.insert(0, str(trainer_path))

# MinIO configuration for fast local model storage
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio.mlops-poc.svc.cluster.local:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin123")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "mlflow-artifacts")
USE_MINIO = os.getenv("USE_MINIO", "true").lower() == "true"

logger = logging.getLogger(__name__)

# Try to import boto3 for MinIO S3 access (lazily if needed)
try:
    import boto3  # type: ignore[import-not-found]
    from botocore.exceptions import ClientError, NoCredentialsError  # type: ignore[import-not-found]
except ImportError:  # noqa: F401 - we fall back to lazy import later
    boto3 = None  # type: ignore[assignment]
    ClientError = NoCredentialsError = Exception  # type: ignore[assignment]


def _ensure_boto3_loaded() -> bool:
    """Attempt to load boto3 at runtime if it was missing at import time."""
    global boto3, ClientError, NoCredentialsError

    if boto3 is not None:  # Already available
        return True

    try:
        boto3 = importlib.import_module("boto3")  # type: ignore[assignment]
        exceptions_mod = importlib.import_module("botocore.exceptions")
        ClientError = getattr(exceptions_mod, "ClientError")  # type: ignore[assignment]
        NoCredentialsError = getattr(exceptions_mod, "NoCredentialsError")  # type: ignore[assignment]
        logger.info("Successfully imported boto3 at runtime for MinIO support")
        return True
    except ImportError as exc:
        logger.warning("boto3 not available for MinIO integration: %s", exc)
        return False

# Configure MLflow model caching BEFORE any MLflow imports
# This ensures models are cached locally instead of re-downloaded from DagsHub
_MLFLOW_CACHE_DIR = os.getenv("MLFLOW_CACHE_DIR", "/data/mlflow_cache")
try:
    Path(_MLFLOW_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    os.environ["MLFLOW_CACHE_DIR"] = _MLFLOW_CACHE_DIR
    logging.info(f"MLflow cache directory set to: {_MLFLOW_CACHE_DIR}")
except Exception as e:
    logging.warning(f"Could not create MLflow cache directory {_MLFLOW_CACHE_DIR}: {e}")
    # Fallback to user's home directory
    _MLFLOW_CACHE_DIR = str(Path.home() / ".mlflow" / "cache")
    os.environ["MLFLOW_CACHE_DIR"] = _MLFLOW_CACHE_DIR
    logging.info(f"Using fallback cache directory: {_MLFLOW_CACHE_DIR}")

import mlflow
import mlflow.pyfunc
import mlflow.xgboost
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration - MLflow registry with DagsHub experiment tracking
from app.config import (
    get_mlflow_production_alias,
    set_mlflow_tracking_uri_if_needed,
)
from app.metrics import get_counter, get_gauge, get_histogram

MODEL_CACHE_DIR = Path(os.getenv("MODEL_CACHE_DIR", "/tmp/models"))
DEFAULT_MODEL_TYPE = os.getenv("DEFAULT_MODEL_TYPE", "xgboost")  # Use best performing model
PRODUCTION_ALIAS = get_mlflow_production_alias()

# A/B Testing configuration - alias-based within model types
# Each policy (brian/forrest/logan) A/B tests between Production and B aliases
# Example: brian games use feedforward_nn@Production or feedforward_nn@B
ENABLE_AB_TESTING = os.getenv("ENABLE_AB_TESTING", "true").lower() == "true"
AB_SPLIT_RATIO = float(os.getenv("AB_SPLIT_RATIO", "0.5"))  # 50/50 split between Production and B aliases


def _resolve_model_split_ratio(model_type: str) -> float:
    """Resolve AB split ratio for a specific model type.

    Environment variables may override the defaults using the pattern
    ``AB_SPLIT_RATIO_<MODEL_TYPE>`` (uppercased). If no override is found, we
    selectively fall back to safer defaults. All model types inherit the global
    ``AB_SPLIT_RATIO`` unless an explicit override is provided for the model
    family.
    """

    env_key = f"AB_SPLIT_RATIO_{model_type.upper()}"
    env_value = os.getenv(env_key)
    if env_value is not None:
        try:
            return float(env_value)
        except ValueError:
            logger.warning("Invalid %s override: %s", env_key, env_value)

    return AB_SPLIT_RATIO


MODEL_AB_SPLIT_RATIOS = {
    model_type: _resolve_model_split_ratio(model_type)
    for model_type in ["xgboost", "feedforward_nn", "multinomial_logistic"]
}

FALLBACK_SOURCES = {
    "fallback",
    "forced_fallback",
    "legacy_fallback",
    "mlflow_error",
    "load_failed",
}


_AB_ENABLED_GAUGE = get_gauge(
    "rps_ab_enabled",
    "Whether A/B testing is enabled (1=yes)",
)
_AB_SPLIT_RATIO_GAUGE = get_gauge(
    "rps_ab_split_ratio",
    "Traffic ratio assigned to model A",
)
_AB_REQUESTS_TOTAL_GAUGE = get_gauge(
    "rps_ab_requests_total",
    "Total prediction requests served by each A/B model",
    ["model"],
)
_AB_ACTIONS_TOTAL_GAUGE = get_gauge(
    "rps_ab_actions_total",
    "Total actions evaluated for each A/B model",
    ["model"],
)
_AB_ACTIONS_CORRECT_GAUGE = get_gauge(
    "rps_ab_actions_correct_total",
    "Number of correctly predicted actions for each A/B model",
    ["model"],
)
_AB_GAMES_TOTAL_GAUGE = get_gauge(
    "rps_ab_games_total",
    "Total games completed for each A/B model",
    ["model"],
)
_AB_GAMES_WON_GAUGE = get_gauge(
    "rps_ab_games_won",
    "Number of games won by the bot for each A/B model",
    ["model"],
)
_AB_ACTION_ACCURACY_GAUGE = get_gauge(
    "rps_ab_action_accuracy",
    "Action-level accuracy for each A/B model",
    ["model"],
)
_AB_GAME_WIN_RATE_GAUGE = get_gauge(
    "rps_ab_game_win_rate",
    "Game win rate for each A/B model",
    ["model"],
)
_AB_DISAGREE_RATIO_GAUGE = get_gauge(
    "rps_ab_disagree_ratio",
    "Fraction of requests where active and shadow predictions differ",
    ["active_model", "shadow_model"],
)
_AB_WIN_LIFT_GAUGE = get_gauge(
    "rps_ab_win_lift",
    "Online win-rate delta versus control variant",
    ["variant"],
)
_AB_EXPOSURE_COUNTER = get_counter(
    "rps_ab_exposure_total",
    "Total prediction exposures served by each AB variant",
    ["variant"],
)
_AB_CALIBRATION_ERROR_GAUGE = get_gauge(
    "rps_ab_calibration_error",
    "Rolling Brier score for AB variants",
    ["variant"],
)
_MODEL_INFO_GAUGE = get_gauge(
    "rps_model_loaded_info",
    "Metadata about currently loaded models",
    ["model", "version", "stage", "alias"],
)
_MODEL_SOURCE_COUNTER = get_counter(
    "rps_model_prediction_sources_total",
    "Prediction sources observed per model",
    ["model", "source"],
)
_MODEL_FALLBACK_COUNTER = get_counter(
    "rps_model_fallback_total",
    "Number of times gameplay fell back to heuristics or legacy models",
    ["model", "reason"],
)
_MODEL_WIN_RATE_GAUGE = get_gauge(
    "rps_model_win_rate",
    "Bot win rate by model (bot perspective)",
    ["model"],
)
_MODEL_FALLBACK_RATIO_GAUGE = get_gauge(
    "rps_model_fallback_ratio",
    "Share of predictions that required fallbacks",
    ["model"],
)
# Per-model prediction and game tracking metrics
_MODEL_PREDICTIONS_TOTAL_COUNTER = get_counter(
    "rps_model_predictions_total",
    "Total predictions made by each model",
    ["model"],
)
_MODEL_PREDICTIONS_CORRECT_COUNTER = get_counter(
    "rps_model_predictions_correct_total",
    "Number of correct predictions by each model",
    ["model"],
)
_MODEL_PREDICTION_ACCURACY_GAUGE = get_gauge(
    "rps_model_prediction_accuracy",
    "Prediction accuracy for each model",
    ["model"],
)
_MODEL_GAMES_TOTAL_COUNTER = get_counter(
    "rps_model_games_total",
    "Total games completed by each model",
    ["model"],
)
_MODEL_GAMES_WON_COUNTER = get_counter(
    "rps_model_games_won_total",
    "Number of games won by each model",
    ["model"],
)
_MODEL_GAME_WIN_RATE_GAUGE = get_gauge(
    "rps_model_game_win_rate",
    "Game win rate for each model",
    ["model"],
)
_MODEL_LOAD_FAILURE_COUNTER = get_counter(
    "rps_model_load_failures_total",
    "Count of model load failures",
    ["model"],
)
_MODEL_LOAD_SKIP_COUNTER = get_counter(
    "rps_model_load_skips_total",
    "Count of skipped load attempts due to cooldown/backoff",
    ["model"],
)
_MODEL_LOAD_SECONDS = get_histogram(
    "rps_model_load_seconds",
    "Time spent loading ML models from the registry",
    ["model", "version"],
)
_REGISTRY_RESOLUTION_COUNTER = get_counter(
    "rps_registry_resolution_total",
    "Outcomes when resolving registry references (alias/stage/fallback)",
    ["result"],
)

# Available model configurations - MLflow registry models
# Note: These define which stage to load from for each model type
AVAILABLE_MODELS = {
    "multinomial_logistic": {
        "name": "rps_bot_mnlogit",
        "stage": "Production",  # Load from Production stage
        "fallback_stage": "Staging",  # Fallback if Production doesn't exist
        "alias": PRODUCTION_ALIAS,
        "description": "Multinomial Logistic Regression",
        "performance": 38.9  # Latest known test accuracy
    },
    "xgboost": {
        "name": "rps_bot_xgboost", 
        "stage": "Production",  # Load from Production stage
        "fallback_stage": "Staging",  # Fallback if Production doesn't exist
        "alias": PRODUCTION_ALIAS,
        "description": "XGBoost Classifier",
        "performance": 52.2  # Latest known test accuracy
    },
    "feedforward_nn": {
        "name": "rps_bot_feedforward",
        "stage": "Production",  # Load from Production stage
        "fallback_stage": "Staging",  # Fallback if Production doesn't exist
        "alias": PRODUCTION_ALIAS,
        "description": "Feedforward Neural Network",
        "performance": 36.7  # Latest known test accuracy
    }
}

MOVES: Tuple[str, str, str] = ("rock", "paper", "scissors")


class _ModelAdapter:
    """Normalise heterogeneous MLflow model interfaces into a common shape."""

    def __init__(self, model_type: str, raw_model: Any):
        self.model_type = model_type
        self.raw_model = raw_model

    def _call_predict(self, features: pd.DataFrame):
        try:
            return self.raw_model.predict(features)
        except TypeError:
            # Some pyfunc wrappers expect (context, model_input)
            return self.raw_model.predict(None, features)

    def _normalise_probabilities(
        self,
        probs_like: Union[List[float], Tuple[float, ...], np.ndarray, pd.Series, Dict[str, Any]],
    ) -> np.ndarray:
        if isinstance(probs_like, dict):
            if {"rock", "paper", "scissors"}.issubset(probs_like.keys()):
                arr = [probs_like["rock"], probs_like["paper"], probs_like["scissors"]]
            elif "probabilities" in probs_like:
                arr = probs_like["probabilities"]
            elif "probs" in probs_like:
                arr = probs_like["probs"]
            else:
                arr = list(probs_like.values())
        elif isinstance(probs_like, pd.Series):
            arr = probs_like.to_numpy()
        elif isinstance(probs_like, pd.DataFrame):
            arr = probs_like.iloc[0].to_numpy()
        else:
            arr = np.asarray(probs_like)

        arr = arr.astype(float).ravel()
        if arr.size < 3:
            arr = np.pad(arr, (0, 3 - arr.size), "constant")
        arr = arr[:3]
        arr = np.clip(arr, 0.0, None)
        total = float(arr.sum())
        if total <= 0.0:
            return np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
        return arr / total

    def predict(self, features: pd.DataFrame) -> Dict[str, Any]:
        if hasattr(self.raw_model, "predict_proba"):
            try:
                probs = self.raw_model.predict_proba(features)[0]
                probs = self._normalise_probabilities(probs)
                pick = MOVES[int(np.argmax(probs))]
                probs_list = probs.tolist()
                return {
                    "pick": pick,
                    "classes": list(MOVES),
                    "probabilities": probs_list,
                    "probs": probs_list,
                    "model_type": self.model_type,
                    "source": "mlflow",
                }
            except Exception as exc:
                logger.warning("predict_proba for %s failed (%s); retrying generic path", self.model_type, exc)

        raw = self._call_predict(features)

        if isinstance(raw, dict):
            probs_obj: Optional[np.ndarray] = None
            if "probabilities" in raw:
                probs_obj = self._normalise_probabilities(raw["probabilities"])
            elif "probs" in raw:
                probs_obj = self._normalise_probabilities(raw["probs"])
            elif {"rock", "paper", "scissors"}.issubset(raw.keys()):
                probs_obj = self._normalise_probabilities(raw)

            if probs_obj is not None:
                pick = raw.get("pick") or MOVES[int(np.argmax(probs_obj))]
                classes = raw.get("classes", list(MOVES))
                probs_list = probs_obj.tolist()
                return {
                    "pick": pick,
                    "classes": list(classes),
                    "probabilities": probs_list,
                    "probs": probs_list,
                    "model_type": self.model_type,
                    "source": raw.get("source", "mlflow"),
                }

            if "pick" in raw:
                pick = raw["pick"]
                probs_obj = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
                probs_list = probs_obj.tolist()
                return {
                    "pick": pick,
                    "classes": list(MOVES),
                    "probabilities": probs_list,
                    "probs": probs_list,
                    "model_type": self.model_type,
                    "source": raw.get("source", "mlflow"),
                }

        if isinstance(raw, (pd.Series, pd.DataFrame, np.ndarray, list, tuple)):
            arr = self._normalise_probabilities(raw)
            pick = MOVES[int(np.argmax(arr))]
            probs_list = arr.tolist()
            return {
                "pick": pick,
                "classes": list(MOVES),
                "probabilities": probs_list,
                "probs": probs_list,
                "model_type": self.model_type,
                "source": "mlflow",
            }

        if isinstance(raw, str) and raw in MOVES:
            probs = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
            probs_list = probs.tolist()
            return {
                "pick": raw,
                "classes": list(MOVES),
                "probabilities": probs_list,
                "probs": probs_list,
                "model_type": self.model_type,
                "source": "mlflow",
            }

        if isinstance(raw, (int, np.integer)) and 0 <= int(raw) < len(MOVES):
            idx = int(raw)
            probs = np.array([0.0, 0.0, 0.0])
            probs[idx] = 1.0
            probs_list = probs.tolist()
            return {
                "pick": MOVES[idx],
                "classes": list(MOVES),
                "probabilities": probs_list,
                "probs": probs_list,
                "model_type": self.model_type,
                "source": "mlflow",
            }

        raise ValueError(f"Unhandled prediction format from {self.model_type}: {type(raw)}")


class ModelManager:
    """Manages loading and serving multiple ML models with A/B testing support"""
    
    def __init__(self):
        self.loaded_models: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Dict] = {}
        self.ab_test_stats: Dict[str, Dict] = {
            "total_requests": 0,
            "model_a_requests": 0,
            "model_b_requests": 0,
            "model_a_wins": 0,
            "model_b_wins": 0
        }
        self.model_usage: Dict[str, Dict[str, Any]] = {}
        self.ab_assignments: Dict[str, str] = {}
        self.ab_alias_assignments: Dict[str, str] = {}
        self.ab_model_stats: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {
                "actions_total": 0,
                "actions_correct": 0,
                "games_total": 0,
                "games_won": 0,
                "last_action_ts": 0.0,
                "last_game_ts": 0.0,
            }
        )
        self.model_game_stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"games_total": 0, "games_won": 0}
        )
        self._model_prediction_stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"total": 0, "fallback": 0}
        )
        self._model_accuracy_stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"total": 0, "correct": 0}
        )
        self.shadow_stats: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(
            lambda: {"total": 0, "disagree": 0}
        )
        self.ab_calibration_stats: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"count": 0, "brier_sum": 0.0}
        )
        self._model_info_labels: Dict[str, Dict[str, str]] = {}
        self._failed_model_loads: Dict[str, float] = {}
        self._reload_backoff_seconds = float(os.getenv("MODEL_RELOAD_BACKOFF_SECONDS", "30"))
        self._metrics_lock = threading.RLock()
        self._metrics_snapshot: Dict[str, Tuple[int, int, int, int]] = {}
        self._metrics_rollup_interval = max(5, int(os.getenv("METRICS_ROLLUP_SECONDS", "60")))
        self._metrics_rollup_stop = threading.Event()
        self._metrics_rollup_thread: Optional[threading.Thread] = None
        self._alias_info_cache: Dict[str, Dict[str, Any]] = {}
        
        # Local model storage configuration
        self.local_models_dir = Path(os.getenv("LOCAL_MODELS_DIR", "/data/models"))
        self.prefer_local_models = os.getenv("PREFER_LOCAL_MODELS", "true").lower() == "true"
        logger.info(f"Local model storage: dir={self.local_models_dir}, prefer_local={self.prefer_local_models}")
        
        # MinIO S3 client for fast model loading
        self.s3_client = None
        self._ensure_s3_client()
        
        self._setup_mlflow()
        logger.info(f"A/B Testing: {'Enabled' if ENABLE_AB_TESTING else 'Disabled'}")
        if ENABLE_AB_TESTING:
            self.model_ab_split_ratios = MODEL_AB_SPLIT_RATIOS.copy()
            logger.info(
                "A/B testing mode: alias-based (Production vs B within each model type, splits: %s)",
                self.model_ab_split_ratios,
            )
        else:
            self.model_ab_split_ratios = MODEL_AB_SPLIT_RATIOS.copy()
        self._update_ab_metrics()
        self._start_metrics_rollup_thread()

    @property
    def ab_testing_enabled(self) -> bool:
        """Report whether alias-based A/B testing is active."""
        return ENABLE_AB_TESTING

    def _ensure_s3_client(self) -> bool:
        """Initialise the MinIO S3 client if available."""
        if not USE_MINIO:
            return False

        if self.s3_client is not None:
            return True

        if not _ensure_boto3_loaded():
            return False

        try:
            self.s3_client = boto3.client(  # type: ignore[arg-type]
                's3',
                endpoint_url=MINIO_ENDPOINT,
                aws_access_key_id=MINIO_ACCESS_KEY,
                aws_secret_access_key=MINIO_SECRET_KEY,
                region_name='us-east-1'
            )
            self.s3_client.head_bucket(Bucket=MINIO_BUCKET)
            logger.info(f"MinIO S3 client initialised: {MINIO_ENDPOINT}/{MINIO_BUCKET}")
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to initialise MinIO S3 client: {exc}")
            self.s3_client = None
            return False

    def _update_model_info_metric(
        self,
        model_type: str,
        version: Optional[str],
        stage: Optional[str],
        alias: Optional[str],
    ) -> None:
        """Refresh Prometheus gauge tracking loaded model metadata."""
        labels = {
            "model": model_type,
            "version": version or "unknown",
            "stage": stage or "unassigned",
            "alias": alias or "none",
        }
        current = self._model_info_labels.get(model_type)
        if current and current != labels:
            _MODEL_INFO_GAUGE.labels(**current).set(0)

        _MODEL_INFO_GAUGE.labels(**labels).set(1)
        self._model_info_labels[model_type] = labels

    def _update_ab_metrics(self):
        """Push the latest A/B statistics into Prometheus gauges.
        
        NOTE: This function is deprecated in favor of per-model@alias metrics.
        A/B testing now compares Production vs B aliases within each model type,
        not cross-class comparisons like xgboost vs feedforward_nn.
        
        Metrics are now tracked via:
        - record_action_accuracy() for action-level tracking (all aliases)
        - record_game_result() for game-level tracking (Production/B only)
        """
        _AB_ENABLED_GAUGE.set(1 if ENABLE_AB_TESTING else 0)
        _AB_SPLIT_RATIO_GAUGE.set(AB_SPLIT_RATIO if ENABLE_AB_TESTING else 0)

    def _compose_model_key(self, model_type: str, alias: str) -> str:
        return f"{model_type}@{alias}"

    def _start_metrics_rollup_thread(self) -> None:
        if self._metrics_rollup_interval <= 0:
            logger.info("Metrics rollup disabled (interval <= 0)")
            return

        # Ensure gauges reflect current snapshot before starting background loop
        self._refresh_alias_metrics(force=True)

        if self._metrics_rollup_thread and self._metrics_rollup_thread.is_alive():
            return

        self._metrics_rollup_thread = threading.Thread(
            target=self._metrics_rollup_worker,
            name="metrics-rollup",
            daemon=True,
        )
        self._metrics_rollup_thread.start()
        logger.info(
            "Metrics rollup thread started (interval=%ss)",
            self._metrics_rollup_interval,
        )

    def _metrics_rollup_worker(self) -> None:
        while not self._metrics_rollup_stop.wait(self._metrics_rollup_interval):
            try:
                updated = self._refresh_alias_metrics()
                if updated:
                    logger.debug("Alias metrics rollup refreshed")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Metrics rollup failed: %s", exc)
        logger.info("Metrics rollup thread exiting")

    def _refresh_alias_metrics(self, *, force: bool = False) -> bool:
        with self._metrics_lock:
            snapshot = {key: value.copy() for key, value in self.ab_model_stats.items()}

        if not snapshot:
            if force:
                self._metrics_snapshot = {}
            return False

        changed = force
        new_snapshot: Dict[str, Tuple[int, int, int, int]] = {}

        for model_key, stats in snapshot.items():
            actions_total = int(stats.get("actions_total", 0))
            actions_correct = int(stats.get("actions_correct", 0))
            games_total = int(stats.get("games_total", 0))
            games_won = int(stats.get("games_won", 0))

            new_snapshot[model_key] = (actions_total, actions_correct, games_total, games_won)

            if self._metrics_snapshot.get(model_key) != new_snapshot[model_key]:
                changed = True
                self._set_alias_metrics(
                    model_key,
                    actions_total,
                    actions_correct,
                    games_total,
                    games_won,
                )

        # Handle keys that disappeared (e.g., reset)
        removed_keys = set(self._metrics_snapshot.keys()) - set(new_snapshot.keys())
        for model_key in removed_keys:
            changed = True
            self._set_alias_metrics(model_key, 0, 0, 0, 0)

        if changed:
            self._metrics_snapshot = new_snapshot

        return changed

    def _set_alias_metrics(
        self,
        model_key: str,
        actions_total: int,
        actions_correct: int,
        games_total: int,
        games_won: int,
    ) -> None:
        try:
            _AB_ACTIONS_TOTAL_GAUGE.labels(model=model_key).set(float(actions_total))
            _AB_ACTIONS_CORRECT_GAUGE.labels(model=model_key).set(float(actions_correct))
            accuracy = (actions_correct / actions_total) if actions_total else 0.0
            _AB_ACTION_ACCURACY_GAUGE.labels(model=model_key).set(accuracy)

            _AB_GAMES_TOTAL_GAUGE.labels(model=model_key).set(float(games_total))
            _AB_GAMES_WON_GAUGE.labels(model=model_key).set(float(games_won))
            win_rate = (games_won / games_total) if games_total else 0.0
            _AB_GAME_WIN_RATE_GAUGE.labels(model=model_key).set(win_rate)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to set alias metrics for %s: %s", model_key, exc)

    def stop_metrics_rollup(self) -> None:
        self._metrics_rollup_stop.set()
        if self._metrics_rollup_thread and self._metrics_rollup_thread.is_alive():
            self._metrics_rollup_thread.join(timeout=5)

    def record_alias_action(
        self,
        model_type: str,
        alias: str,
        predicted_move: str,
        actual_user_move: str,
        round_no: int,
    ) -> None:
        if round_no <= 3:
            return

        model_key = self._compose_model_key(model_type, alias)
        is_correct = predicted_move.lower() == actual_user_move.lower()

        with self._metrics_lock:
            stats = self.ab_model_stats[model_key]
            stats["actions_total"] += 1
            if is_correct:
                stats["actions_correct"] += 1
            stats["last_action_ts"] = time.time()

        try:
            _MODEL_PREDICTIONS_TOTAL_COUNTER.labels(model=model_key).inc()
            if is_correct:
                _MODEL_PREDICTIONS_CORRECT_COUNTER.labels(model=model_key).inc()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to update prediction counters for %s: %s", model_key, exc)

    def record_alias_game_result(
        self,
        model_type: str,
        alias: str,
        bot_won: bool,
        policy: str,
        difficulty_mode: str,
    ) -> None:
        model_key = self._compose_model_key(model_type, alias)

        with self._metrics_lock:
            stats = self.ab_model_stats[model_key]
            stats["games_total"] += 1
            if bot_won:
                stats["games_won"] += 1
            stats["last_game_ts"] = time.time()

        try:
            _MODEL_GAMES_TOTAL_COUNTER.labels(model=model_key).inc()
            if bot_won:
                _MODEL_GAMES_WON_COUNTER.labels(model=model_key).inc()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to update game counters for %s: %s", model_key, exc)

        # Record alias-specific game result for Grafana dashboard (Production/B only)
        try:
            from app.metrics import record_model_alias_game_result
            record_model_alias_game_result(
                model=model_type,
                alias=alias,
                difficulty_mode=difficulty_mode,
                won=bot_won,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to record alias game result for %s@%s: %s", model_type, alias, exc)

        # NOTE: Policy-level game metrics are recorded in app/routes/game.py
        # to avoid double-counting. This function only records model-level metrics.
    
    def _setup_mlflow(self):
        """Configure MLflow client"""
        uri = set_mlflow_tracking_uri_if_needed()
        logger.info(f"MLflow tracking URI set to: {uri}")
        
        MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    def list_available_models(self) -> Dict[str, Dict]:
        """Get list of available models and their metadata"""
        models = AVAILABLE_MODELS.copy()
        # All models support A/B testing via Production vs B aliases
        if ENABLE_AB_TESTING:
            for model_type in models:
                models[model_type]["ab_enabled"] = True
                models[model_type]["aliases"] = ["Production", "B", "shadow1", "shadow2"]
        return models
    
    def _resolve_alias_to_run_id(self, model_name: str, alias: str) -> Optional[str]:
        """Resolve alias to run ID - try MinIO first, then MLflow as fallback"""
        # Try MinIO first (no dependency on DagsHub)
        if USE_MINIO and self.s3_client:
            try:
                response = self.s3_client.list_objects_v2(Bucket=MINIO_BUCKET)
                if 'Contents' in response:
                    for obj in response['Contents']:
                        key = obj['Key']
                        if f'/.alias_{alias}' in key:
                            run_id = key.split('/')[0]
                            logger.info(f"Resolved alias '{alias}' to run_id {run_id[:8]}... from MinIO")
                            return run_id
            except Exception as e:
                logger.debug(f"MinIO alias resolution failed for {alias}: {e}")
        
        # Fallback to MLflow only if MinIO didn't have it
        try:
            set_mlflow_tracking_uri_if_needed()
            client = mlflow.tracking.MlflowClient()
            aliased = client.get_model_version_by_alias(model_name, alias)
            logger.info(f"Resolved alias '{alias}' to run_id {aliased.run_id[:8]}... from MLflow")
            return aliased.run_id
        except Exception as e:
            logger.warning(f"Could not resolve alias '{alias}' for {model_name} from MLflow: {e}")
            return None
    
    def _load_model_from_local(self, model_name: str, run_id: str, model_type: str) -> Optional[Any]:
        """Load model from local filesystem storage"""
        import json
        import torch
        import xgboost as xgb
        
        model_dir = self.local_models_dir / model_name / run_id
        metadata_file = model_dir / "metadata.json"
        
        if not model_dir.exists() or not metadata_file.exists():
            logger.debug(f"Local model not found: {model_dir}")
            return None
        
        # Load and validate metadata
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            if metadata.get("model_type") != model_type:
                logger.warning(f"Model type mismatch in local metadata: expected {model_type}, got {metadata.get('model_type')}")
                return None
            
            logger.info(f"Loading model from local storage: {model_dir}")
            
            # Load model based on type
            if model_type == "xgboost":
                model_file = model_dir / "model.xgb"
                if not model_file.exists():
                    logger.warning(f"XGBoost model file not found: {model_file}")
                    return None
                model = xgb.Booster()
                model.load_model(str(model_file))
                logger.info(f"✅ Loaded XGBoost model from local storage (<1s)")
                return model
            else:
                # Load PyTorch model
                model_file = model_dir / "model.pt"
                if not model_file.exists():
                    logger.warning(f"PyTorch model file not found: {model_file}")
                    return None
                
                checkpoint = torch.load(model_file, map_location='cpu')
                
                # Reconstruct model architecture
                if model_type in ["feedforward", "feedforward_nn"]:
                    from trainer.model_defs import FeedForwardNN
                    model = FeedForwardNN(input_dim=50, hidden_dim=128, dropout=0.3)
                elif model_type in ["mnlogit", "multinomial_logistic"]:
                    from trainer.model_defs import MultinomialLogit
                    model = MultinomialLogit(input_dim=50)
                else:
                    logger.error(f"Unknown PyTorch model type: {model_type}")
                    return None
                
                model.load_state_dict(checkpoint['state_dict'])
                model.eval()
                logger.info(f"✅ Loaded {model_type} model from local storage (<1s)")
                return model
                
        except Exception as e:
            logger.warning(f"Failed to load model from local storage: {e}")
            return None
    
    def get_ab_test_stats(self) -> Dict[str, Any]:
        """Get A/B testing statistics.
        
        NOTE: This returns simplified stats. Full per-model@alias metrics
        are available via Prometheus metrics and analytics endpoints.
        """
        if not ENABLE_AB_TESTING:
            return {"ab_testing_enabled": False}
            
        stats = self.ab_test_stats.copy()
        stats["ab_testing_enabled"] = True
        stats["split_ratio"] = AB_SPLIT_RATIO
        stats["per_model_split_ratios"] = self.model_ab_split_ratios.copy()
        stats["mode"] = "alias_based"  # New: Production vs B within model types
        stats["description"] = "A/B testing compares Production vs B aliases within each model type"
        
        self._update_ab_metrics()
        return stats
    
    def select_model_for_ab_test(self, user_id: str = None) -> str:
        """Select model for A/B testing based on user ID or random selection.
        
        DEPRECATED: This function is deprecated in the new alias-based A/B testing.
        Model type is now determined by policy (brian→feedforward_nn, forrest→xgboost, logan→multinomial_logistic).
        A/B testing selects between Production and B aliases via select_model_alias_for_policy().
        
        Returns DEFAULT_MODEL_TYPE for backward compatibility.
        """
        return DEFAULT_MODEL_TYPE

    def assign_model_for_game(self, game_id: str, user_id: Optional[str] = None) -> Optional[str]:
        """Assign and cache an A/B testing model for the lifetime of a game."""
        if not ENABLE_AB_TESTING:
            return None

        if game_id in self.ab_assignments:
            return self.ab_assignments[game_id]

        model = self.select_model_for_ab_test(user_id)
        self.ab_assignments[game_id] = model
        return model

    def clear_ab_assignment(self, game_id: str):
        """Remove cached A/B assignment when a game concludes."""
        self.ab_assignments.pop(game_id, None)

        prefix = f"{game_id}|"
        with self._metrics_lock:
            keys_to_remove = [key for key in self.ab_alias_assignments if key.startswith(prefix)]
            for key in keys_to_remove:
                self.ab_alias_assignments.pop(key, None)
    
    def select_model_alias_for_policy(self, model_type: str, game_id: str = None, user_id: str = None) -> str:
        """Select model alias (Production vs B) for policy-based A/B testing.

        Alias choice is random per game/session to honour the configured split ratio.
        The first call for a given game caches the assignment so subsequent requests
        for the same game reuse the original variant.
        """
        if not ENABLE_AB_TESTING:
            return "Production"

        assignment_key = game_id or user_id
        cache_key = None
        existing_alias: Optional[str] = None

        if assignment_key:
            cache_key = f"{assignment_key}|{model_type}"
            with self._metrics_lock:
                existing_alias = self.ab_alias_assignments.get(cache_key)

        if existing_alias:
            logger.debug("Reusing cached A/B alias %s for %s", existing_alias, cache_key)
            return existing_alias

        ratio = self.model_ab_split_ratios.get(model_type, AB_SPLIT_RATIO)

        if ratio <= 0:
            selected_alias = "B"
        elif ratio >= 1:
            selected_alias = "Production"
        else:
            selected_alias = "Production" if random.random() < ratio else "B"

        with self._metrics_lock:
            if cache_key:
                self.ab_alias_assignments[cache_key] = selected_alias

            try:
                if selected_alias == "Production":
                    self.ab_test_stats["model_a_requests"] += 1
                else:
                    self.ab_test_stats["model_b_requests"] += 1
                self.ab_test_stats["total_requests"] += 1
            except Exception:
                logger.debug("Failed updating AB request counters", exc_info=True)

        variant_key = self._compose_model_key(model_type, selected_alias)

        try:
            _AB_EXPOSURE_COUNTER.labels(variant=variant_key).inc()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to increment AB exposure counter for %s: %s", variant_key, exc)

        # Ensure stats structure exists
        with self._metrics_lock:
            _ = self.ab_model_stats[variant_key]

        logger.debug("A/B test alias selection for %s: %s (key=%s)", model_type, selected_alias, cache_key)
        return selected_alias
    
    def _load_model_from_minio(
        self,
        model_type: str,
        alias: str,
    ) -> Tuple[Optional[Any], Optional[str], Optional[str], Dict[str, Any]]:
        """Load model from MinIO S3 storage without requiring MLflow/DagsHub.

        Returns a tuple of (model, run_id, version). Run/version may be None if
        they cannot be determined from marker files.
        """
        if not self._ensure_s3_client():
            logger.debug("MinIO S3 client not available, skipping MinIO load")
            return None, None, None, {}
            
        try:
            config = AVAILABLE_MODELS[model_type]
            model_name = config["name"]
            
            # Look for alias marker files in MinIO to find the run_id
            # Format: {run_id}/.alias_{alias} (e.g., "abc123/.alias_Production")
            run_id = None
            version_number = None
            
            try:
                # List all objects to find alias markers
                response = self.s3_client.list_objects_v2(Bucket=MINIO_BUCKET)
                
                if 'Contents' in response:
                    for obj in response['Contents']:
                        key = obj['Key']
                        # Check if this is an alias marker for our model type and alias
                        if f'/.alias_{alias}' in key:
                            # Extract run_id from path
                            parts = key.split('/')
                            if len(parts) >= 2:
                                potential_run_id = parts[0]
                                # Verify this run_id has the model files we need
                                model_prefix = f"{potential_run_id}/model/"
                                check_response = self.s3_client.list_objects_v2(
                                    Bucket=MINIO_BUCKET,
                                    Prefix=model_prefix,
                                    MaxKeys=1
                                )
                                if 'Contents' in check_response:
                                    # Read the marker file to verify it's for the correct model type
                                    # Format: "rps_bot_xgboost@Production v94"
                                    try:
                                        marker_obj = self.s3_client.get_object(Bucket=MINIO_BUCKET, Key=key)
                                        marker_content = marker_obj['Body'].read().decode('utf-8').strip()
                                        
                                        # Verify this marker is for the requested model name
                                        # Extract model name from marker content
                                        if '@' in marker_content:
                                            marker_model_name = marker_content.split('@')[0]
                                            if marker_model_name != model_name:
                                                logger.debug(f"Skipping marker {key}: expected {model_name}, found {marker_model_name}")
                                                continue  # Wrong model type, keep searching
                                        
                                        # Extract version from "model_name@alias vXX" format
                                        if ' v' in marker_content:
                                            version_str = marker_content.split(' v')[-1]
                                            version_number = version_str
                                        
                                        # This is the correct marker
                                        run_id = potential_run_id
                                        logger.info(f"Found {model_type}@{alias} in MinIO via alias marker: run_id={run_id[:8]}... version={version_number}")
                                        break
                                        
                                    except Exception as e:
                                        logger.debug(f"Could not read/parse marker {key}: {e}")
                                        continue
                
                if not run_id:
                    logger.warning(f"No alias marker found in MinIO for {model_type}@{alias}")
                    return None, None, None, {}
                    
            except Exception as e:
                logger.warning(f"Failed to find alias in MinIO for {model_type}@{alias}: {e}")
                return None, None, None, {}
            
            # Download model artifacts from MinIO
            model_prefix = f"{run_id}/model/"
            local_model_dir = Path(f"/tmp/minio_models/{model_type}/{run_id}")
            local_model_dir.mkdir(parents=True, exist_ok=True)
            
            # List objects in MinIO with this prefix
            response = self.s3_client.list_objects_v2(
                Bucket=MINIO_BUCKET,
                Prefix=model_prefix
            )
            
            if 'Contents' not in response:
                logger.warning(f"No model artifacts found in MinIO for {model_type}@{alias} (run: {run_id[:8]})")
                return None, run_id, version_number, {}
            
            # Download all model files
            artifact_count = 0
            for obj in response['Contents']:
                key = obj['Key']
                local_file = local_model_dir / key.replace(model_prefix, '')
                local_file.parent.mkdir(parents=True, exist_ok=True)
                
                self.s3_client.download_file(MINIO_BUCKET, key, str(local_file))
                artifact_count += 1
            
            logger.info(f"Downloaded {artifact_count} artifacts from MinIO for {model_type}@{alias}")
            
            # Load the model using MLflow
            model_path = f"file://{local_model_dir}"
            start_time = time.time()
            
            # All models are now logged as pyfunc with custom wrappers
            model = mlflow.pyfunc.load_model(model_path)
            
            load_time = time.time() - start_time
            logger.info(f"Loaded {model_type}@{alias} from MinIO in {load_time:.3f}s")
            metadata = {
                "run_id": run_id,
                "version": version_number,
                "load_time_seconds": load_time,
                "artifact_count": artifact_count,
                "loading_method": "minio",
                "source": "minio",
            }
            return model, run_id, version_number, metadata
            
        except Exception as e:
            logger.error(f"Failed to load {model_type}@{alias} from MinIO: {e}")
            return None, None, None, {}
    
    def _get_alias_info(
        self,
        model_type: str,
        alias: str,
        *,
        ttl_seconds: float = 60.0,
    ) -> Optional[Dict[str, Any]]:
        """Resolve the current run/version for a model alias via MLflow with caching."""

        cache_key = f"{model_type}@{alias}"
        now = time.time()
        cached = self._alias_info_cache.get(cache_key)
        if cached and (now - cached.get("resolved_at", 0.0)) < ttl_seconds:
            return cached

        config = AVAILABLE_MODELS.get(model_type)
        if not config:
            logger.error("Unknown model_type when resolving alias info: %s", model_type)
            return None

        try:
            client = mlflow.tracking.MlflowClient()
            mv = client.get_model_version_by_alias(config["name"], alias)
            info = {
                "run_id": mv.run_id,
                "version": str(mv.version),
                "resolved_at": now,
            }
            self._alias_info_cache[cache_key] = info
            return info
        except Exception as exc:  # noqa: BLE001
            logger.warning("Unable to resolve alias info for %s: %s", cache_key, exc)
            return None

    def load_model_with_alias(self, model_type: str, alias: str, force_reload: bool = False) -> bool:
        """Load a model with a specific alias (Production, B, shadow1, shadow2)"""
        if model_type not in AVAILABLE_MODELS:
            logger.error(f"Unknown model type: {model_type}")
            return False
            
        # Create a cache key that includes the alias
        cache_key = f"{model_type}@{alias}"
        if cache_key in self.loaded_models and not force_reload:
            latest_info = self._get_alias_info(model_type, alias)
            current_meta = self.model_metadata.get(cache_key, {})
            current_run = current_meta.get("run_id")
            current_version = str(current_meta.get("version")) if current_meta.get("version") is not None else None

            needs_reload = False
            if not current_meta:
                needs_reload = True
            elif latest_info:
                latest_run = latest_info.get("run_id")
                latest_version = latest_info.get("version")
                if latest_run and current_run and latest_run != current_run:
                    needs_reload = True
                elif latest_version and current_version and latest_version != current_version:
                    needs_reload = True
                elif latest_run and not current_run:
                    needs_reload = True
                elif latest_version and not current_version:
                    needs_reload = True

            if not needs_reload:
                return True

            logger.info(
                "Alias update detected for %s (run %s -> %s); forcing reload",
                cache_key,
                current_run,
                latest_info.get("run_id") if latest_info else None,
            )
            force_reload = True

        if force_reload:
            self.loaded_models.pop(cache_key, None)
            self.model_metadata.pop(cache_key, None)
            self._alias_info_cache.pop(cache_key, None)

        config = AVAILABLE_MODELS[model_type].copy()
        config["alias"] = alias  # Override the alias
        
        model = None
        model_version = None
        run_id: Optional[str] = None
        loading_method = "mlflow_registry"
        extra_metadata: Dict[str, Any] = {}
        
        # Try MinIO first for fastest loading
        if USE_MINIO and self.s3_client:
            logger.info(f"Attempting to load {model_type}@{alias} from MinIO")
            model, run_id, version_number, minio_meta = self._load_model_from_minio(model_type, alias)
            if model:
                loading_method = "minio"
                model_version = version_number
                extra_metadata.update(minio_meta)
                logger.info(f"✅ Loaded {model_type}@{alias} from MinIO")
                self._on_model_loaded(
                    cache_key,
                    model,
                    config,
                    f"minio://{run_id}" if run_id else "minio",
                    f"alias:{alias}",
                    alias,
                    model_version,
                    run_id=run_id,
                )
                metadata = self.model_metadata.get(cache_key, {})
                metadata.update(extra_metadata)
                return True
        
        # Try loading from local storage next (if enabled)
        if self.prefer_local_models and model is None:
            try:
                run_id = self._resolve_alias_to_run_id(config["name"], alias)
                if run_id:
                    logger.info(f"Resolved alias '{alias}' to run_id: {run_id}")
                    model = self._load_model_from_local(config["name"], run_id, model_type)
                    if model:
                        loading_method = "local_filesystem"
                        model_version = run_id[:7]  # Use short run_id as version
                        logger.info(f"✅ Loaded {model_type}@{alias} from local storage")
            except Exception as e:
                logger.warning(f"Local model load failed for {model_type}@{alias}, falling back to MLflow: {e}")
        
        # Fallback to MLflow if both MinIO and local load failed
        if model is None:
            try:
                set_mlflow_tracking_uri_if_needed()
                client = mlflow.tracking.MlflowClient()
                
                alias_uri = f"models:/{config['name']}@{alias}"
                logger.info(f"Loading {model_type} with alias '{alias}' from MLflow: {alias_uri}")
                
                model = self._load_model_from_uri(cache_key, alias_uri)
                loading_method = "mlflow_registry"
                
                # Get version info for the alias
                try:
                    aliased = client.get_model_version_by_alias(config["name"], alias)
                    model_version = aliased.version
                    run_id = aliased.run_id
                except Exception as e:
                    logger.warning(f"Unable to resolve alias '{alias}' to version for {config['name']}: {e}")
                    model_version = None
                    
            except Exception as e:
                logger.error(f"Failed to load {model_type} with alias '{alias}' from MLflow: {e}")
                self._failed_model_loads[cache_key] = time.time()
                self.record_prediction_source(
                    cache_key,
                    "load_failed",
                    str(e),
                    activity="load",
                )
                return False
        
        # Store the loaded model
        self._on_model_loaded(
            cache_key,
            model,
            config,
            f"models:/{config['name']}@{alias}",
            f"alias:{alias}",
            alias,
            model_version,
            run_id=run_id,
        )
        
        # Update metadata to reflect loading method
        if cache_key in self.model_metadata:
            self.model_metadata[cache_key]["loading_method"] = loading_method
            if run_id:
                self.model_metadata[cache_key]["run_id"] = run_id
            if extra_metadata:
                self.model_metadata[cache_key].update(extra_metadata)
        
        logger.info(f"Successfully loaded {model_type} with alias '{alias}' via {loading_method}")
        return True

    def is_model_loaded(self, model_key: str) -> bool:
        """Check if a model is currently loaded. 
        
        Args:
            model_key: Either model_type or model_type@alias format
        """
        return model_key in self.loaded_models
    
    def load_model(self, model_type: str, force_reload: bool = False) -> bool:
        """Load a model from MLflow registry"""
        if model_type not in AVAILABLE_MODELS:
            logger.error(f"Unknown model type: {model_type}")
            return False

        if not force_reload:
            failure_ts = self._failed_model_loads.get(model_type)
            if failure_ts is not None:
                elapsed = time.time() - failure_ts
                if elapsed < self._reload_backoff_seconds:
                    logger.warning(
                        "Skipping load for %s; last failure %.1fs ago (<%.1fs cooldown)",
                        model_type,
                        elapsed,
                        self._reload_backoff_seconds,
                    )
                    self.record_prediction_source(
                        model_type,
                        "load_skipped",
                        "recent_failure",
                        activity="load",
                    )
                    _MODEL_LOAD_SKIP_COUNTER.labels(model=model_type).inc()
                    return False

        if self.is_model_loaded(model_type) and not force_reload:
            logger.info(f"Model {model_type} already loaded")
            return True

        config = AVAILABLE_MODELS[model_type]
        client = mlflow.tracking.MlflowClient()

        model_uri = None
        loaded_stage = None
        alias_used: Optional[str] = None
        model_version: Optional[str] = None
        model_run_id: Optional[str] = None

        alias_name = config.get("alias")
        if alias_name:
            alias_uri = f"models:/{config['name']}@{alias_name}"
            try:
                logger.info(f"Attempting to load {model_type} via alias '{alias_name}'")
                model = self._load_model_from_uri(model_type, alias_uri)
                model_uri = alias_uri
                loaded_stage = f"alias:{alias_name}"
                alias_used = alias_name
                try:
                    aliased = client.get_model_version_by_alias(config["name"], alias_name)
                    model_version = aliased.version
                    model_run_id = aliased.run_id
                except Exception as e:  # pragma: no cover - informational warning only
                    logger.warning(
                        "Unable to resolve alias '%s' to a version for %s: %s",
                        alias_name,
                        config["name"],
                        e,
                    )
                self._on_model_loaded(
                    model_type,
                    model,
                    config,
                    model_uri,
                    loaded_stage,
                    alias_used,
                    model_version,
                    run_id=model_run_id,
                )
                return True
            except Exception as e:
                logger.warning(f"Alias '{alias_name}' unavailable for {config['name']}: {e}")

        for stage_to_try in [config['stage'], config.get('fallback_stage', 'None')]:
            try:
                test_uri = f"models:/{config['name']}/{stage_to_try}"
                logger.info(f"Attempting to load {model_type} from {config['name']}/{stage_to_try}")
                versions = client.get_latest_versions(config['name'], stages=[stage_to_try])
                if versions:
                    model_uri = test_uri
                    loaded_stage = stage_to_try
                    model_version = versions[0].version
                    model_run_id = versions[0].run_id
                    logger.info(f"Found model at stage: {stage_to_try}")
                    break
                else:
                    logger.warning(f"No model found at stage: {stage_to_try}")
            except Exception as e:
                logger.warning(f"Failed to check stage {stage_to_try}: {e}")
                continue

        if model_uri is None:
            try:
                logger.info(f"Falling back to latest version for {config['name']}")
                latest_versions = client.get_latest_versions(config['name'])
                if latest_versions:
                    latest_version = max(latest_versions, key=lambda v: int(v.version))
                    model_uri = f"models:/{config['name']}/{latest_version.version}"
                    loaded_stage = f"v{latest_version.version}"
                    model_version = latest_version.version
                    model_run_id = latest_version.run_id
                    logger.info(f"Using version {latest_version.version}")
            except Exception as e:
                logger.error(f"Failed to find any version of {config['name']}: {e}")
                return False

        if model_uri is None:
            logger.error(f"No loadable model found for {model_type}")
            return False

        try:
            model = self._load_model_from_uri(model_type, model_uri)
            self._on_model_loaded(
                model_type,
                model,
                config,
                model_uri,
                loaded_stage,
                alias_used,
                model_version,
                run_id=model_run_id,
            )
            logger.info(
                "✅ %s loaded successfully from %s (performance: %s%%)",
                model_type,
                loaded_stage,
                config["performance"],
            )
            self._failed_model_loads.pop(model_type, None)
            return True

        except Exception as e:
            self.record_prediction_source(model_type, "load_failed", str(e), activity="load")
            logger.error(f"Failed to load model {model_type}: {e}")
            self._failed_model_loads[model_type] = time.time()
            _MODEL_LOAD_FAILURE_COUNTER.labels(model=model_type).inc()
            return False

    def _load_model_from_uri(self, model_type: str, model_uri: str):
        """Helper to load a model using the best available loader."""
        # All models are now logged as pyfunc with custom wrappers
        return mlflow.pyfunc.load_model(model_uri)

    def _on_model_loaded(
        self,
        model_type: str,
        model: Any,
        config: Dict[str, Any],
        model_uri: str,
        loaded_stage: Optional[str],
        alias_name: Optional[str],
        model_version: Optional[str],
        *,
        run_id: Optional[str] = None,
    ) -> None:
        """Book-keep metadata when a model load succeeds."""
        adapter_label = config.get("adapter_label") or (model_type.split("@", 1)[0] if "@" in model_type else model_type)
        self.loaded_models[model_type] = _ModelAdapter(adapter_label, model)
        self.model_metadata[model_type] = {
            "name": config["name"],
            "stage": loaded_stage,
            "alias": alias_name,
            "version": model_version,
            "run_id": run_id,
            "description": config["description"],
            "loaded_at": pd.Timestamp.now().isoformat(),
            "model_uri": model_uri,
            "performance": config["performance"],
            "loading_method": "mlflow_registry",
        }
        self.record_prediction_source(model_type, "mlflow", None, activity="load")
        self._update_model_info_metric(
            model_type,
            model_version,
            loaded_stage,
            alias_name,
        )

    def _capture_inference_snapshot(
        self,
        *,
        model_type: str,
        alias: Optional[str],
        features: pd.DataFrame,
        prediction: Dict[str, Any],
        source: str,
    ) -> None:
        """Persist a lightweight inference snapshot when debugging is enabled."""

        capture_dir = os.getenv("RPS_CAPTURE_INFERENCE_DIR")
        if not capture_dir:
            return

        try:
            sample_rate = float(os.getenv("RPS_CAPTURE_INFERENCE_SAMPLE_RATE", "1.0"))
        except ValueError:
            sample_rate = 1.0

        if sample_rate <= 0.0 or random.random() > sample_rate:
            return

        try:
            base_path = Path(capture_dir)
            base_path.mkdir(parents=True, exist_ok=True)

            timestamp = pd.Timestamp.utcnow().isoformat()
            alias_id = alias or "noalias"
            file_name = f"{timestamp}_{model_type}_{alias_id}_{uuid.uuid4().hex[:8]}.json"

            snapshot = {
                "timestamp": timestamp,
                "model_type": model_type,
                "alias": alias,
                "source": source,
                "probabilities": prediction.get("probabilities"),
                "pick": prediction.get("pick"),
                "feature_columns": list(features.columns),
                "features": features.to_dict(orient="records"),
            }

            metadata_key = f"{model_type}@{alias}" if alias else model_type
            metadata = self.model_metadata.get(metadata_key) or self.model_metadata.get(model_type)
            if metadata:
                snapshot["model_metadata"] = {
                    "version": metadata.get("version"),
                    "run_id": metadata.get("run_id"),
                    "loading_method": metadata.get("loading_method"),
                    "model_uri": metadata.get("model_uri"),
                }

            with (base_path / file_name).open("w", encoding="utf-8") as fh:
                json.dump(snapshot, fh, default=float)

        except Exception as exc:  # pragma: no cover - diagnostics must not break inference
            logger.debug("Failed to capture inference snapshot: %s", exc)
    
    def predict_with_alias(self, model_type: str, alias: str, features: pd.DataFrame) -> Optional[Dict]:
        """Make prediction using specified model with specific alias"""
        cache_key = f"{model_type}@{alias}"
        
        if not self.is_model_loaded(cache_key):
            logger.info(f"Model {model_type} with alias '{alias}' not loaded, attempting to load...")
            if not self.load_model_with_alias(model_type, alias):
                logger.error(f"Failed to load model {model_type} with alias '{alias}'")
                return None
        
        try:
            adapter = self.loaded_models[cache_key]
            prediction = adapter.predict(features)
            source = prediction.get("source", "mlflow")
            probs = prediction.get("probabilities")
            if probs is not None:
                prediction["probabilities"] = probs
            
            # Add alias information to the prediction
            prediction["model_alias"] = alias
            prediction["model_type"] = model_type
            prediction["model_cache_key"] = cache_key
            
            self.record_prediction_source(cache_key, source, None)
            self._capture_inference_snapshot(
                model_type=model_type,
                alias=alias,
                features=features,
                prediction=prediction,
                source=source,
            )
            
            # NOTE: Alias-specific prediction metrics are recorded in app/routes/game.py
            # where all 4 aliases are queried together with correctness checking.
            # Do NOT record here to avoid double-counting the active alias.
            
            return prediction

        except Exception as e:
            logger.error(f"Prediction failed for {model_type} with alias '{alias}': {e}")
            self.record_prediction_source(cache_key, "mlflow_error", str(e))
            return None

    def predict(self, model_type: str, features: pd.DataFrame) -> Optional[Dict]:
        """Make prediction using specified model"""
        if not self.is_model_loaded(model_type):
            logger.warning(f"Model {model_type} not loaded, attempting to load...")
            if not self.load_model(model_type):
                logger.error(f"Failed to load model {model_type}")
                return None
        
        try:
            adapter = self.loaded_models[model_type]
            prediction = adapter.predict(features)
            source = prediction.get("source", "mlflow")
            probs = prediction.get("probabilities")
            if probs is not None:
                prediction["probabilities"] = probs
            self.record_prediction_source(model_type, source, None)
            self._capture_inference_snapshot(
                model_type=model_type,
                alias=None,
                features=features,
                prediction=prediction,
                source=source,
            )
            return prediction

        except Exception as e:
            logger.error(f"Prediction failed for {model_type}: {e}")
            self.record_prediction_source(model_type, "mlflow_error", str(e))
            return None
    
    def _normalise_probability_dict(
        self,
        probabilities: Optional[Union[Dict[str, Any], List[float], Tuple[float, ...]]],
    ) -> Dict[str, float]:
        """Convert assorted probability payloads into a move->probability map."""
        if probabilities is None:
            probs = np.array([1.0 / 3] * 3)
        elif isinstance(probabilities, dict):
            if {"rock", "paper", "scissors"}.issubset(probabilities.keys()):
                probs = np.array(
                    [
                        float(probabilities["rock"]),
                        float(probabilities["paper"]),
                        float(probabilities["scissors"]),
                    ]
                )
            elif "probabilities" in probabilities:
                seq = list(probabilities["probabilities"])
                probs = np.array((seq + [0, 0, 0])[:3], dtype=float)
            else:
                seq = list(probabilities.values())
                probs = np.array((seq + [0, 0, 0])[:3], dtype=float)
        else:
            seq = list(probabilities)
            probs = np.array((seq + [0, 0, 0])[:3], dtype=float)

        probs = np.clip(probs, 0.0, None)
        total = float(probs.sum())
        if total <= 0.0:
            probs = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
        else:
            probs = probs / total

        return {
            "rock": float(probs[0]),
            "paper": float(probs[1]),
            "scissors": float(probs[2]),
        }

    def record_ab_action(
        self,
        model_type: str,
        probabilities: Optional[Union[Dict[str, Any], List[float], Tuple[float, ...]]],
        actual_move: str,
    ) -> None:
        """Record action-level accuracy and calibration metrics for AB variants.
        
        DEPRECATED: Use record_action_accuracy() instead.
        This function was for cross-class A/B testing (xgboost vs feedforward_nn).
        New architecture uses alias-based testing (Production vs B within same model type).
        """
        # Deprecated - no-op
        pass

    def record_ab_game_result(self, model_type: str, bot_won: bool):
        """Record game-level results for A/B testing models.
        
        DEPRECATED: Use record_game_result() instead.
        This function was for cross-class A/B testing (xgboost vs feedforward_nn).
        New architecture uses alias-based testing (Production vs B within same model type).
        """
        # Deprecated - no-op
        pass

    def record_model_game_result(self, model_type: Optional[str], bot_won: Optional[bool]) -> None:
        """Track win rates for individual models (bot perspective)."""
        if not model_type or model_type not in AVAILABLE_MODELS:
            return

        stats = self.model_game_stats[model_type]
        stats["games_total"] += 1
        if bot_won:
            stats["games_won"] += 1

        try:
            win_rate = stats["games_won"] / stats["games_total"] if stats["games_total"] else 0.0
            _MODEL_WIN_RATE_GAUGE.labels(model=model_type).set(win_rate)
            # Update new comprehensive metrics
            _MODEL_GAMES_TOTAL_COUNTER.labels(model=model_type).inc()
            _MODEL_GAME_WIN_RATE_GAUGE.labels(model=model_type).set(win_rate)
            if bot_won:
                _MODEL_GAMES_WON_COUNTER.labels(model=model_type).inc()
        except Exception:
            logger.debug("Unable to update model win-rate gauge", exc_info=True)

    def record_model_prediction(self, model_type: Optional[str], was_correct: Optional[bool], source: Optional[str] = None) -> None:
        """Track prediction-level accuracy for individual models (excluding fallbacks)."""
        if not model_type or model_type not in AVAILABLE_MODELS:
            return
            
        # Skip fallback predictions - only count actual ML model predictions
        if source and source in FALLBACK_SOURCES:
            return
            
        # Track stats in internal counters for accuracy calculation
        if model_type not in self._model_accuracy_stats:
            self._model_accuracy_stats[model_type] = {"total": 0, "correct": 0}
            
        self._model_accuracy_stats[model_type]["total"] += 1
        if was_correct:
            self._model_accuracy_stats[model_type]["correct"] += 1
            
        try:
            _MODEL_PREDICTIONS_TOTAL_COUNTER.labels(model=model_type).inc()
            if was_correct:
                _MODEL_PREDICTIONS_CORRECT_COUNTER.labels(model=model_type).inc()
                
            # Calculate accuracy from internal stats
            stats = self._model_accuracy_stats[model_type]
            accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
            _MODEL_PREDICTION_ACCURACY_GAUGE.labels(model=model_type).set(accuracy)
            
        except Exception:
            logger.debug("Unable to update model prediction metrics", exc_info=True)
            
    def backfill_model_metrics_from_db(self) -> Dict[str, Dict[str, int]]:
        """Backfill model metrics from historical database data."""
        try:
            from app.db import DB
            cur = DB.cursor()
            
            # Query historical game data excluding fallbacks
            game_query = """
            SELECT 
                bot_model_version,
                COUNT(*) as total_games,
                SUM(CASE WHEN winner = 'bot' THEN 1 ELSE 0 END) as games_won
            FROM games 
            WHERE winner IS NOT NULL 
                AND bot_model_version IS NOT NULL
                AND bot_model_version NOT LIKE 'legacy:%'
                AND bot_model_version NOT LIKE '%fallback%'
                AND bot_model_version IN ('feedforward_nn', 'xgboost', 'multinomial_logistic')
            GROUP BY bot_model_version
            """
            
            game_results = cur.execute(game_query).fetchall()
            
            # Query historical prediction data from events table
            # This is trickier since we need to determine correctness from user/bot moves
            prediction_query = """
            SELECT 
                g.bot_model_version,
                COUNT(*) as total_predictions,
                SUM(CASE 
                    WHEN (e.user_move = 'rock' AND e.bot_move = 'paper') OR
                         (e.user_move = 'paper' AND e.bot_move = 'scissors') OR  
                         (e.user_move = 'scissors' AND e.bot_move = 'rock')
                    THEN 1 ELSE 0 END) as correct_predictions
            FROM events e
            JOIN games g ON e.game_id = g.id
            WHERE g.bot_model_version IS NOT NULL
                AND g.bot_model_version NOT LIKE 'legacy:%'
                AND g.bot_model_version NOT LIKE '%fallback%'
                AND g.bot_model_version IN ('feedforward_nn', 'xgboost', 'multinomial_logistic')
            GROUP BY g.bot_model_version
            """
            
            prediction_results = cur.execute(prediction_query).fetchall()
            
            # Update metrics with historical data
            backfilled_data = {}
            
            # Process game results
            for model, total_games, games_won in game_results:
                if model in AVAILABLE_MODELS:
                    # Update internal stats (these will be used for ongoing tracking)
                    self.model_game_stats[model] = {
                        "games_total": int(total_games),
                        "games_won": int(games_won)
                    }
                    
                    # For Prometheus counters, we can only increment, not set values
                    # So we'll set the gauges but not the counters to historical values
                    win_rate = games_won / total_games if total_games > 0 else 0.0
                    _MODEL_GAME_WIN_RATE_GAUGE.labels(model=model).set(win_rate)
                    _MODEL_WIN_RATE_GAUGE.labels(model=model).set(win_rate)
                    
                    backfilled_data[model] = {
                        "games_total": int(total_games),
                        "games_won": int(games_won),
                        "game_win_rate": win_rate
                    }
            
            # Process prediction results  
            for model, total_predictions, correct_predictions in prediction_results:
                if model in AVAILABLE_MODELS:
                    # Update internal accuracy stats
                    self._model_accuracy_stats[model] = {
                        "total": int(total_predictions),
                        "correct": int(correct_predictions)
                    }
                    
                    # Update accuracy gauge only (counters will start from 0 going forward)
                    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
                    _MODEL_PREDICTION_ACCURACY_GAUGE.labels(model=model).set(accuracy)
                    
                    if model in backfilled_data:
                        backfilled_data[model].update({
                            "predictions_total": int(total_predictions),
                            "predictions_correct": int(correct_predictions),
                            "prediction_accuracy": accuracy
                        })
                    else:
                        backfilled_data[model] = {
                            "predictions_total": int(total_predictions),
                            "predictions_correct": int(correct_predictions),
                            "prediction_accuracy": accuracy
                        }
            
            logger.info(f"Backfilled metrics for {len(backfilled_data)} models: {list(backfilled_data.keys())}")
            return backfilled_data
            
        except Exception as e:
            logger.error(f"Failed to backfill model metrics: {e}")
            return {}

    def _record_shadow_disagreement(
        self,
        active_model: str,
        shadow_model: str,
        active_pick: str,
        shadow_pick: Optional[str],
    ) -> None:
        stats = self.shadow_stats[(active_model, shadow_model)]
        stats["total"] += 1
        if shadow_pick != active_pick:
            stats["disagree"] += 1
        try:
            ratio = stats["disagree"] / stats["total"] if stats["total"] else 0.0
            _AB_DISAGREE_RATIO_GAUGE.labels(
                active_model=active_model,
                shadow_model=shadow_model,
            ).set(ratio)
        except Exception:
            logger.debug("Unable to update AB disagree ratio", exc_info=True)

    def evaluate_shadow_predictions(
        self,
        features: Optional[pd.DataFrame],
        active_model: Optional[str],
        active_probs: Optional[Dict[str, float]],
    ) -> None:
        if features is None or not isinstance(features, pd.DataFrame):
            return
        if not active_model or active_model not in AVAILABLE_MODELS:
            return
        if not active_probs:
            active_probs = {move: 1.0 / 3 for move in MOVES}

        active_pick = max(active_probs, key=active_probs.get)

        for shadow_model in AVAILABLE_MODELS.keys():
            if shadow_model == active_model:
                continue
            try:
                if not self.is_model_loaded(shadow_model):
                    if not self.load_model(shadow_model):
                        continue
                adapter = self.loaded_models.get(shadow_model)
                if not adapter:
                    continue
                prediction = adapter.predict(features)
            except Exception:
                logger.debug(
                    "Shadow model prediction failed",
                    exc_info=True,
                )
                prediction = None
            shadow_pick = prediction.get("pick") if prediction else None
            self._record_shadow_disagreement(active_model, shadow_model, active_pick, shadow_pick)
    
    def predict_with_ab_testing(self, features: pd.DataFrame, user_id: str = None) -> Optional[Dict]:
        """Make prediction using A/B testing model selection"""
        selected_model = self.select_model_for_ab_test(user_id)
        result = self.predict(selected_model, features)
        
        if result:
            result["ab_test_model"] = selected_model
            result["ab_testing_enabled"] = ENABLE_AB_TESTING
            
        return result
    
    def get_model_info(self, model_type: str) -> Optional[Dict]:
        """Get metadata for a loaded model"""
        if model_type not in self.model_metadata:
            return None
        return self.model_metadata[model_type].copy()
    
    def get_all_model_info(self) -> Dict[str, Dict]:
        """Get metadata for all loaded models"""
        return {
            model_type: {
                **info.copy(),
                **self.model_usage.get(model_type, {}),
            }
            for model_type, info in self.model_metadata.items()
        }
    
    def unload_model(self, model_type: str) -> bool:
        """Unload a model to free memory"""
        if model_type in self.loaded_models:
            del self.loaded_models[model_type]
            if model_type in self.model_metadata:
                del self.model_metadata[model_type]
            labels = self._model_info_labels.pop(model_type, None)
            if labels:
                try:
                    _MODEL_INFO_GAUGE.labels(**labels).set(0)
                except Exception:  # pragma: no cover - defensive guard
                    logger.debug("Unable to clear model info gauge", exc_info=True)
            logger.info(f"Unloaded model {model_type}")
            return True
        return False
    
    def load_default_model(self) -> bool:
        """Load the default model"""
        return self.load_model(DEFAULT_MODEL_TYPE)
    
    def load_all_models(self) -> Dict[str, bool]:
        """Load all available models"""
        results = {}
        for model_type in AVAILABLE_MODELS:
            results[model_type] = self.load_model(model_type)
        return results

    def record_prediction_source(
        self,
        model_type: str,
        source: str,
        error: Optional[str],
        *,
        activity: str = "prediction",
    ) -> None:
        """Track how the most recent prediction/load attempt was satisfied."""
        usage = self.model_usage.setdefault(model_type, {
            "last_source": None,
            "last_error": None,
            "last_updated": None,
            "details": None,
        })
        usage["last_source"] = source
        usage["last_error"] = error
        usage["last_updated"] = pd.Timestamp.now().isoformat()
        usage["details"] = error
        try:
            _MODEL_SOURCE_COUNTER.labels(model=model_type, source=source).inc()
        except Exception:  # pragma: no cover - defensive guard
            logger.debug("Unable to update model source counter", exc_info=True)

        if source in FALLBACK_SOURCES:
            try:
                _MODEL_FALLBACK_COUNTER.labels(model=model_type, reason=source).inc()
            except Exception:
                logger.debug("Unable to update fallback counter", exc_info=True)

        if activity == "prediction" and model_type in AVAILABLE_MODELS:
            stats = self._model_prediction_stats.setdefault(model_type, {"total": 0, "fallback": 0})
            stats["total"] += 1
            if source in FALLBACK_SOURCES:
                stats["fallback"] += 1
            try:
                ratio = stats["fallback"] / stats["total"] if stats["total"] else 0.0
                _MODEL_FALLBACK_RATIO_GAUGE.labels(model=model_type).set(ratio)
            except Exception:
                logger.debug("Unable to update fallback ratio gauge", exc_info=True)
    
    # === Stage Management Methods ===
    
    def promote_model_to_stage(self, model_name: str, version: str, target_stage: str) -> bool:
        """Promote a specific model version to a target stage"""
        try:
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=target_stage
            )
            logger.info(f"✅ Promoted {model_name} v{version} to {target_stage}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to promote {model_name} v{version} to {target_stage}: {e}")
            return False
    
    def demote_model_from_stage(self, model_name: str, current_stage: str) -> bool:
        """Demote models from a specific stage (move to Archived)"""
        try:
            client = mlflow.tracking.MlflowClient()
            versions = client.get_latest_versions(model_name, stages=[current_stage])
            
            for version in versions:
                client.transition_model_version_stage(
                    name=model_name,
                    version=version.version,
                    stage="Archived"
                )
                logger.info(f"Archived {model_name} v{version.version} from {current_stage}")
            return True
        except Exception as e:
            logger.error(f"Failed to demote {model_name} from {current_stage}: {e}")
            return False
    
    def list_model_stages(self, model_name: str) -> Dict[str, str]:
        """List all versions and their current stages for a model"""
        try:
            client = mlflow.tracking.MlflowClient()
            model_info = client.get_registered_model(model_name)
            stages = {}
            
            for version in model_info.latest_versions:
                stages[f"v{version.version}"] = version.current_stage or "None"
                
            return stages
        except Exception as e:
            logger.error(f"Failed to get stage info for {model_name}: {e}")
            return {}
    
    def get_production_models(self) -> Dict[str, Dict]:
        """Get all models currently in Production stage"""
        production_models = {}
        
        try:
            client = mlflow.tracking.MlflowClient()
            for model in client.search_registered_models():
                prod_versions = client.get_latest_versions(model.name, stages=["Production"])
                if prod_versions:
                    production_models[model.name] = {
                        "version": prod_versions[0].version,
                        "stage": "Production",
                        "status": prod_versions[0].status
                    }
        except Exception as e:
            logger.error(f"Failed to get production models: {e}")
            
        return production_models


# ============================================================================
# Metrics Recording Helpers
# ============================================================================

def record_action_accuracy(
    model_type: str,
    alias: str,
    predicted_move: str,
    actual_user_move: str,
    round_no: int
) -> None:
    """Record action-level prediction accuracy for a model@alias.
    
    This tracks whether the model correctly predicted the user's move.
    Shadow models are also tracked here.
    
    Args:
        model_type: Model type (xgboost, feedforward_nn, multinomial_logistic)
        alias: Model alias (Production, B, shadow1, shadow2)
        predicted_move: Move predicted by the model
        actual_user_move: Actual move played by user
        round_no: Current round number (skip first 3 gambit moves)
    """
    try:
        manager = get_model_manager()
        manager.record_alias_action(model_type, alias, predicted_move, actual_user_move, round_no)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to record action accuracy for %s@%s: %s", model_type, alias, exc)


def record_game_result(
    model_type: str,
    alias: str,
    bot_won: bool,
    policy: str,
    difficulty_mode: str
) -> None:
    """Record game-level result for a model@alias.
    
    This tracks wins/losses for the bot. Only active models (Production/B)
    should call this - shadow models don't play games.
    
    Args:
        model_type: Model type (xgboost, feedforward_nn, multinomial_logistic)
        alias: Model alias (should be Production or B)
        bot_won: True if bot won the game
        policy: Policy name (brian, forrest, logan)
        difficulty_mode: Difficulty mode (standard, easy)
    """
    try:
        manager = get_model_manager()
        manager.record_alias_game_result(model_type, alias, bot_won, policy, difficulty_mode)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to record game result for %s@%s: %s", model_type, alias, exc)



# Global model manager instance
model_manager = ModelManager()

def get_model_manager() -> ModelManager:
    """Get the global model manager instance"""
    return model_manager
