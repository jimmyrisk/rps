#!/usr/bin/env python3
"""
Alias-Preserving Continuation Training for RPS MLOps Platform

This script performs continuation training on all 12 aliased models:
- 3 model types (xgboost, feedforward_nn, multinomial_logistic)
- 4 aliases per type (Production, B, shadow1, shadow2)

Process per model:
1. Load existing model by alias from MLflow registry
2. Extract hyperparameters and trained weights
3. Continue training on current data (with overlap, that's OK)
4. Register as new version in MLflow
5. Re-point the same alias to the new version
6. Sync to MinIO for fast serving
7. Trigger app reload

Runs sequentially to manage memory on single 4GB server.
"""

import json
import os
import sys
import time
import argparse
import sqlite3
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import mlflow.pyfunc
import mlflow.xgboost
import mlflow.pytorch
import requests

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_mlflow_tracking_uri, get_training_data_since_date
from app.legacy_models import get_legacy_display_name, list_legacy_policies
from bot_vs_bot_sim import APIClient, play_single_game
from trainer.train_feedforward import FeedforwardRPSModel
from trainer.train_mnlogit_torch import MNLogitRPSModel
from trainer.train_xgboost import XGBoostRPSModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Model type to trainer class mapping
MODEL_TRAINERS = {
    "xgboost": XGBoostRPSModel,
    "feedforward_nn": FeedforwardRPSModel,
    "multinomial_logistic": MNLogitRPSModel,
}

# Model type to MLflow registry name mapping
MODEL_REGISTRY_NAMES = {
    "xgboost": "rps_bot_xgboost",
    "feedforward_nn": "rps_bot_feedforward",
    "multinomial_logistic": "rps_bot_mnlogit",
}

# All aliases to process
ALIASES = ["Production", "B", "shadow1", "shadow2"]

# Legacy simulation configuration for auto-fill
LEGACY_USER_POLICIES: List[str] = list_legacy_policies()
SERVER_POLICIES: List[str] = ["brian", "forrest", "logan"]
AUTOFILL_STATE_FILE = Path(
    os.getenv(
        "TRAINING_SIM_STATE_FILE",
        Path("local") / "training_autofill_state.json",
    )
)
ENABLE_AUTOFILL = os.getenv("ENABLE_TRAINING_AUTOFILL", "true").lower() != "false"
AUTOFILL_DIFFICULTY = os.getenv("TRAINING_SIM_DIFFICULTY", "normal")
HEALTH_ENDPOINTS = ("/healthz", "/health")


def _load_autofill_state() -> Dict[str, int]:
    defaults = {"legacy_index": 0, "server_index": 0}
    try:
        if AUTOFILL_STATE_FILE.exists():
            data = json.loads(AUTOFILL_STATE_FILE.read_text())
            defaults["legacy_index"] = int(data.get("legacy_index", 0))
            defaults["server_index"] = int(data.get("server_index", 0))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load auto-fill rotation state: %s", exc)
    return defaults


def _save_autofill_state(state: Dict[str, int]) -> None:
    try:
        AUTOFILL_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        AUTOFILL_STATE_FILE.write_text(json.dumps(state))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to persist auto-fill rotation state: %s", exc)


def _discover_api_base() -> Optional[str]:
    configured = os.getenv("TRAINING_SIM_API_BASE")
    candidates: List[str] = []
    if configured:
        candidates.append(configured.strip())
    candidates.extend(
        [
            "http://rps-app.mlops-poc.svc.cluster.local:8080",
            "http://localhost:8080",
            "https://mlops-rps.uk",
        ]
    )

    unique_candidates: List[str] = []
    seen = set()
    for base in candidates:
        if not base:
            continue
        normalized = base.rstrip("/")
        if normalized not in seen:
            seen.add(normalized)
            unique_candidates.append(normalized)

    session = requests.Session()
    for base in unique_candidates:
        for endpoint in HEALTH_ENDPOINTS:
            try:
                resp = session.get(f"{base}{endpoint}", timeout=5)
                if resp.status_code < 500:
                    session.close()
                    logger.info("Using %s for auto-fill gameplay", base)
                    return base
            except requests.RequestException:
                continue
    session.close()
    return None


def _autofill_games_via_api(api_base: str) -> Dict[str, Any]:
    if not LEGACY_USER_POLICIES:
        logger.warning("No legacy policies available for auto-fill gameplay")
        return {"played": 0, "successes": 0, "failures": 0, "details": []}

    state = _load_autofill_state()
    total_users = len(LEGACY_USER_POLICIES)
    total_servers = len(SERVER_POLICIES) or 1

    start_legacy = state.get("legacy_index", 0) % total_users
    start_server = state.get("server_index", 0) % total_servers

    ordered_users = LEGACY_USER_POLICIES[start_legacy:] + LEGACY_USER_POLICIES[:start_legacy]

    api_client = APIClient(api_base)
    results: List[Dict[str, Any]] = []
    successes = 0
    failures = 0

    for offset, user_policy in enumerate(ordered_users):
        server_policy = SERVER_POLICIES[(start_server + offset) % total_servers]
        try:
            result = play_single_game(
                api_client,
                user_policy,
                server_policy,
                verbose=False,
                easy_mode=False,
                difficulty_mode=AUTOFILL_DIFFICULTY,
            )
            winner = result.get("winner")
            if winner in {"error", "timeout", None}:
                failures += 1
            else:
                successes += 1
            results.append(
                {
                    "user_policy": get_legacy_display_name(user_policy),
                    "server_policy": server_policy,
                    "game_id": result.get("game_id"),
                    "winner": winner,
                }
            )
        except Exception as exc:  # noqa: BLE001
            failures += 1
            results.append(
                {
                    "user_policy": get_legacy_display_name(user_policy),
                    "server_policy": server_policy,
                    "error": str(exc),
                }
            )
            logger.warning(
                "Auto-fill gameplay failed for %s vs %s: %s",
                user_policy,
                server_policy,
                exc,
            )

    new_state = {
        "legacy_index": (start_legacy + 1) % total_users,
        "server_index": (start_server + 1) % total_servers,
    }
    _save_autofill_state(new_state)

    return {
        "played": len(results),
        "successes": successes,
        "failures": failures,
        "details": results,
    }


def check_training_needed(
    db_path: str,
    min_new_rows: int = 50,
    min_total_rows: int = 300,
) -> Tuple[bool, dict]:
    """
    Check if training is needed based on new data availability.
    Excludes test games from count.
    
    NOTE: These counts represent raw database rows BEFORE downsampling.
    Legacy bot game downsampling (configured via TRAINING_LEGACY_GAME_STRIDE)
    happens during actual training in BaseRPSModel.load_data().
    The thresholds here ensure sufficient overall activity before triggering
    expensive training jobs.
    
    Returns (should_train, stats_dict)
    """
    training_since = os.getenv("TRAINING_DATA_SINCE_DATE", get_training_data_since_date())
    try:
        lookback_minutes = int(os.getenv("TRAINING_NEW_ROWS_LOOKBACK_MINUTES", "60"))
    except ValueError:
        lookback_minutes = 60
    
    try:
        with sqlite3.connect(db_path) as conn:
            # Get total non-test rows within time window (exclude "now" timestamps)
            total_rows = conn.execute(
                """SELECT COUNT(*) FROM events e
                   INNER JOIN games g ON e.game_id = g.id
                   WHERE g.is_test = 0
                   AND g.created_ts_utc IS NOT NULL
                   AND g.created_ts_utc != 'now'
                   AND g.created_ts_utc >= ?""",
                (training_since,)
            ).fetchone()[0]
            
            # Determine lookback window for new event threshold
            cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=max(lookback_minutes, 1))
            cutoff_iso = cutoff_time.replace(microsecond=0).isoformat()
            
            # Count new non-test rows since cutoff within time window (exclude "now")
            new_rows = conn.execute(
                """SELECT COUNT(*) FROM events e
                   INNER JOIN games g ON e.game_id = g.id
                   WHERE g.is_test = 0 
                   AND g.created_ts_utc IS NOT NULL
                   AND g.created_ts_utc != 'now'
                   AND g.created_ts_utc > ?
                   AND g.created_ts_utc >= ?""",
                (cutoff_iso, training_since)
            ).fetchone()[0]
            
            # Count test games for reporting
            test_games = conn.execute(
                "SELECT COUNT(*) FROM games WHERE is_test = 1"
            ).fetchone()[0]
            
            stats = {
                "total_rows": total_rows,
                "new_rows_since_cutoff": new_rows,
                "min_new_rows_threshold": min_new_rows,
                "min_total_rows_threshold": min_total_rows,
                "cutoff_time": cutoff_iso,
                "lookback_minutes": lookback_minutes,
                "training_data_since": training_since,
                "test_games_excluded": test_games
            }
            
            should_train = (total_rows >= min_total_rows and new_rows >= min_new_rows)
            
            return should_train, stats
            
    except Exception as e:
        logger.error(f"⚠️  Error checking training requirements: {e}")
        # Default to NOT training if we can't check (fail safe)
        return False, {"error": str(e)}


def extract_hyperparameters_from_run(run_id: str, client: mlflow.MlflowClient) -> Dict[str, Any]:
    """Extract hyperparameters from an MLflow run"""
    run = client.get_run(run_id)
    params = run.data.params
    
    # Convert string params back to their original types
    hyperparams = {}
    for key, value in params.items():
        # Skip non-hyperparameter params
        if key in ['lookback', 'model_name', 'model_type']:
            continue
            
        # Try to convert to appropriate type
        try:
            # Try int first
            if '.' not in value:
                hyperparams[key] = int(value)
            else:
                hyperparams[key] = float(value)
        except (ValueError, AttributeError):
            # Keep as string or boolean
            if value.lower() in ['true', 'false']:
                hyperparams[key] = value.lower() == 'true'
            else:
                hyperparams[key] = value
    
    return hyperparams


def load_model_and_extract_hyperparams(model_type: str, alias: str) -> Tuple[Optional[Any], Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    """
    Load a model by alias and extract its hyperparameters.
    
    For feedforward_nn, extracts architecture from the saved pyfunc model.
    For other models, extracts hyperparameters from MLflow run params.
    
    Returns:
        (loaded_model, hyperparams_dict, version, run_id)
    """
    import mlflow
    from mlflow.tracking import MlflowClient
    from app.config import set_mlflow_tracking_uri_if_needed
    
    set_mlflow_tracking_uri_if_needed()
    client = MlflowClient()
    
    # Map model type to MLflow model name
    model_registry_name = {
        'xgboost': 'rps_bot_xgboost',
        'feedforward_nn': 'rps_bot_feedforward',
        'multinomial_logistic': 'rps_bot_mnlogit'
    }[model_type]
    
    logger.info(f"Step 1/5: Loading existing {model_type}@{alias}...")
    
    # Get model version by alias
    mv = client.get_model_version_by_alias(model_registry_name, alias)
    version = mv.version
    run_id = mv.run_id
    
    logger.info(f"📦 Found {model_type}@{alias}: version {version}, run {run_id[:8]}")
    
    # Load the model to extract configuration
    model_uri = f"models:/{model_registry_name}@{alias}"
    hyperparams = {}
    
    if model_type == "feedforward_nn":
        # Load pyfunc model and extract architecture from it
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        
        # Access the underlying Python model to get architecture
        pyfunc_model = loaded_model._model_impl.python_model
        
        # Extract architecture parameters
        hyperparams = {
            "HIDDEN_SIZES": ",".join(map(str, pyfunc_model.hidden_sizes)),
            "DROPOUT_RATE": "0.3",  # Default - not stored in model
            "EPOCHS": "200",  # We'll train with same epochs
            "BATCH_SIZE": "64",  # Default
            "LR": "0.001",  # Default
            "WEIGHT_DECAY": "0.0001",  # Default  
            "PATIENCE": "15",  # Default
            "REDUCE_LR_PATIENCE": "7"  # Default
        }
        
        # Try to get actual training hyperparams from run
        run = client.get_run(run_id)
        run_params = run.data.params
        for key in ["dropout_rate", "epochs", "batch_size", "lr", "weight_decay", "patience", "reduce_lr_patience"]:
            if key in run_params:
                hyperparams[key.upper()] = run_params[key]
        
        logger.info(f"📋 Extracted architecture: hidden_sizes={pyfunc_model.hidden_sizes}")
        logger.info(f"📋 Using hyperparameters: {hyperparams}")
        
    else:
        # For other models, extract from run params
        run = client.get_run(run_id)
        hyperparams = {k.upper(): v for k, v in run.data.params.items()}
        logger.info(f"📋 Extracted hyperparameters: {hyperparams}")
        
        # Load model for validation - ALL models use pyfunc now
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"✅ Successfully loaded model from {model_uri}")
    
    return loaded_model, hyperparams, version, run_id


def train_with_hyperparameters(
    model_type: str,
    alias: str,
    hyperparams: Optional[Dict[str, Any]],
    trainer_class: Any
) -> bool:
    """
    Train a model using the specified hyperparameters.
    
    Returns:
        True if training succeeded, False otherwise
    """
    # Set hyperparameters as environment variables
    # BUT: Don't overwrite env vars that are already explicitly set (e.g., from CronJob config)
    if hyperparams:
        logger.info(f"🔧 Applying hyperparameters for {model_type}@{alias}")
        for key, value in hyperparams.items():
            env_key = key.upper()
            if env_key not in os.environ:
                # Only set if not already defined in environment
                os.environ[env_key] = str(value)
                logger.debug(f"   Set {env_key}={value}")
            else:
                # Use the value from environment (e.g., fast training overrides)
                logger.debug(f"   Keeping existing {env_key}={os.environ[env_key]} (not overwriting with {value})")
    
    # Set alias for promotion (this will assign the alias after training)
    os.environ['PROMOTE_ALIAS'] = alias
    
    # Create trainer instance
    logger.info(f"🏋️  Initializing {model_type} trainer...")
    trainer = trainer_class()
    
    # Run full training workflow (uses BaseRPSModel.train())
    logger.info(f"🚀 Starting training for {model_type}@{alias}...")
    success = trainer.train()
    
    if success:
        logger.info(f"✅ Training completed successfully for {model_type}@{alias}")
        return True
    else:
        logger.error(f"❌ Training failed for {model_type}@{alias}")
        return False


def sync_to_minio(model_type: str, alias: str, run_id: str, version: str) -> bool:
    """
    Sync the newly trained model to MinIO for fast serving.
    
    Args:
        model_type: Model type (xgboost, feedforward_nn, multinomial_logistic)
        alias: Model alias (Production, B, shadow1, shadow2)
        run_id: MLflow run ID
        version: Model version number
    
    Returns:
        True if sync succeeded, False otherwise
    """
    from app.minio_sync import sync_single_model_to_minio
    
    logger.info(f"🔄 Syncing {model_type}@{alias} v{version} (run {run_id[:8]}) to MinIO...")
    
    success = sync_single_model_to_minio(model_type, alias, run_id, version)
    
    if success:
        logger.info(f"✅ MinIO sync completed for {model_type}@{alias}")
    else:
        logger.warning(f"⚠️  MinIO sync failed for {model_type}@{alias}")
        
    return success


def verify_minio_presence(run_ids: list) -> Tuple[int, int]:
    """
    Verify that specific run_ids are present in MinIO.
    
    Args:
        run_ids: List of MLflow run_ids to check
    
    Returns:
        (found_count, total_count) tuple
    """
    try:
        import subprocess
        
        # Check if we can access MinIO via kubectl
        try:
            result = subprocess.run(
                ["kubectl", "-n", "mlops-poc", "exec", "deploy/minio", "--", 
                 "mc", "ls", "local/mlflow-artifacts/"],
                capture_output=True,
                text=True,
                timeout=15
            )
            
            if result.returncode != 0:
                logger.warning(f"⚠️  Could not list MinIO contents: {result.stderr}")
                return (0, len(run_ids))
            
            minio_contents = result.stdout
            found = 0
            
            for run_id in run_ids:
                if run_id in minio_contents:
                    found += 1
                    logger.debug(f"✅ Found {run_id[:8]} in MinIO")
                else:
                    logger.warning(f"❌ Missing {run_id[:8]} from MinIO")
            
            return (found, len(run_ids))
            
        except FileNotFoundError:
            logger.debug("kubectl not available, skipping MinIO verification")
            return (0, len(run_ids))
        except subprocess.TimeoutExpired:
            logger.warning("⚠️  MinIO verification timed out")
            return (0, len(run_ids))
            
    except Exception as e:
        logger.error(f"❌ Error verifying MinIO presence: {e}", exc_info=True)
        return (0, len(run_ids))


def trigger_app_reload() -> Tuple[bool, Dict[str, Any]]:
    """
    Trigger the app to reload models from MinIO/MLflow.
    
    Returns:
        (success, response_data) tuple where response_data contains reload details
    """
    try:
        import requests
        
        # Try production endpoint first
        endpoints = [
            "https://mlops-rps.uk/models/reload",
            "http://localhost:8080/models/reload",
            "http://rps-app.mlops-poc.svc.cluster.local:8080/models/reload"
        ]
        
        for endpoint in endpoints:
            try:
                logger.info(f"🔄 Triggering reload at {endpoint}...")
                response = requests.post(endpoint, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"✅ App reload successful at {endpoint}")
                    logger.info(f"   MinIO: {data.get('minio_synced', 0)} synced, "
                              f"{data.get('minio_skipped', 0)} skipped, "
                              f"{data.get('minio_errors', 0)} errors")
                    logger.info(f"   Models: {len(data.get('reloaded', []))} reloaded, "
                              f"{len(data.get('failed', []))} failed")
                    return True, data
                else:
                    logger.warning(f"⚠️  Reload returned {response.status_code}: {response.text}")
                    
            except requests.RequestException as e:
                logger.debug(f"Could not reach {endpoint}: {e}")
                continue
        
        logger.warning("⚠️  Could not trigger app reload at any endpoint")
        return False, {}
        
    except Exception as e:
        logger.error(f"❌ Error triggering app reload: {e}", exc_info=True)
        return False, {}


def report_training_completion_event(
    model_type: str,
    status: str,
    alias: Optional[str] = None,
    duration_seconds: Optional[float] = None,
) -> bool:
    """Send training completion event to the app for Prometheus metrics."""
    try:
        import requests
    except ImportError:
        logger.debug("requests library not available; skipping training completion callback")
        return False

    payload = {
        "model_type": model_type,
        "status": status,
        "alias": alias,
        "duration_seconds": duration_seconds,
        "source": "train_all_aliases",
    }

    custom_endpoints = os.getenv("TRAINING_COMPLETION_ENDPOINTS")
    if custom_endpoints:
        endpoints = [endpoint.strip() for endpoint in custom_endpoints.split(",") if endpoint.strip()]
    else:
        endpoints = [
            "https://mlops-rps.uk/internal/training/complete",
            "http://localhost:8080/internal/training/complete",
            "http://rps-app.mlops-poc.svc.cluster.local:8080/internal/training/complete",
        ]

    for endpoint in endpoints:
        try:
            response = requests.post(endpoint, json=payload, timeout=10)
            if response.status_code == 200:
                logger.info(
                    "📈 Recorded training completion for %s@%s (%s) via %s",
                    model_type,
                    alias or "all",
                    status,
                    endpoint,
                )
                return True
            logger.debug(
                "Training completion endpoint %s responded %s: %s",
                endpoint,
                response.status_code,
                response.text,
            )
        except requests.RequestException as exc:
            logger.debug("Could not reach %s: %s", endpoint, exc)

    logger.warning(
        "⚠️  Unable to record training completion metric for %s@%s (%s)",
        model_type,
        alias or "all",
        status,
    )
    return False


def train_all_aliases(
    model_types: Optional[list] = None,
    aliases: Optional[list] = None,
    skip_minio_sync: bool = False,
    skip_app_reload: bool = False
) -> Dict[str, Any]:
    """
    Train all 12 models (or a subset) with alias preservation.
    
    Args:
        model_types: List of model types to train (default: all 3)
        aliases: List of aliases to train (default: all 4)
        skip_minio_sync: Skip MinIO sync after training
        skip_app_reload: Skip app reload at the end
    
    Returns:
        Dictionary with training results and statistics
    """
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(get_mlflow_tracking_uri())
    client = mlflow.MlflowClient()
    
    # Default to all model types and aliases
    if model_types is None:
        model_types = list(MODEL_TRAINERS.keys())
    if aliases is None:
        aliases = ALIASES
    
    # Results tracking
    results = {
        "total": 0,
        "succeeded": 0,
        "failed": 0,
        "skipped": 0,
        "details": []
    }
    
    total_models = len(model_types) * len(aliases)
    logger.info(f"🎯 Starting alias-preserving continuation training")
    logger.info(f"   Model types: {model_types}")
    logger.info(f"   Aliases: {aliases}")
    logger.info(f"   Total models to train: {total_models}")
    logger.info(f"=" * 80)
    
    # Process each model type
    for model_type in model_types:
        trainer_class = MODEL_TRAINERS[model_type]
        
        # Process each alias
        for alias in aliases:
            results["total"] += 1
            model_key = f"{model_type}@{alias}"
            
            logger.info("")
            logger.info(f"{'=' * 80}")
            logger.info(f"🔨 Processing {model_key} ({results['total']}/{total_models})")
            logger.info(f"{'=' * 80}")
            
            start_time = time.time()
            
            # Step 1: Load existing model and hyperparameters
            logger.info(f"Step 1/5: Loading existing {model_key}...")
            loaded_model, hyperparams, old_version, run_id = load_model_and_extract_hyperparams(
                model_type, alias
            )
            
            # If model doesn't exist yet, use default hyperparameters
            if hyperparams is None:
                logger.warning(f"⚠️  No existing model found for {model_key}, using default hyperparameters")
                hyperparams = {}
            
            # Step 2: Train with existing hyperparameters
            logger.info(f"Step 2/5: Training {model_key} with hyperparameters...")
            training_success = train_with_hyperparameters(
                model_type, alias, hyperparams, trainer_class
            )
            
            if not training_success:
                logger.error(f"❌ Training failed for {model_key}")
                duration = time.time() - start_time
                report_training_completion_event(
                    model_type,
                    "failure",
                    alias=alias,
                    duration_seconds=duration,
                )
                results["failed"] += 1
                results["details"].append({
                    "model": model_key,
                    "status": "failed",
                    "reason": "training_error",
                    "old_version": old_version,
                    "duration_seconds": duration
                })
                continue
            
            # Step 3: Verify new version was created and alias was assigned
            logger.info(f"Step 3/5: Verifying alias assignment for {model_key}...")
            try:
                new_model_version = client.get_model_version_by_alias(
                    MODEL_REGISTRY_NAMES[model_type], alias
                )
                new_version = new_model_version.version
                new_run_id = new_model_version.run_id
                
                logger.info(f"✅ Alias updated: {old_version} -> {new_version} (run {new_run_id[:8]})")
                
            except Exception as e:
                logger.error(f"❌ Could not verify alias assignment: {e}")
                duration = time.time() - start_time
                report_training_completion_event(
                    model_type,
                    "failure",
                    alias=alias,
                    duration_seconds=duration,
                )
                results["failed"] += 1
                results["details"].append({
                    "model": model_key,
                    "status": "failed",
                    "reason": "alias_assignment_failed",
                    "old_version": old_version,
                    "duration_seconds": duration
                })
                continue
            
            # Step 4: Sync to MinIO
            if not skip_minio_sync:
                logger.info(f"Step 4/5: Syncing {model_key} to MinIO...")
                sync_success = sync_to_minio(model_type, alias, new_run_id, new_version)
                if not sync_success:
                    logger.warning(f"⚠️  MinIO sync failed for {model_key}, but continuing...")
            else:
                logger.info(f"Step 4/5: Skipping MinIO sync (--skip-minio-sync)")
            
            # Step 5: Cleanup
            logger.info(f"Step 5/5: Cleanup for {model_key}...")
            import gc
            gc.collect()
            
            duration = time.time() - start_time
            logger.info(f"✅ Completed {model_key} in {duration:.1f}s")
            report_training_completion_event(
                model_type,
                "success",
                alias=alias,
                duration_seconds=duration,
            )
            
            results["succeeded"] += 1
            results["details"].append({
                "model": model_key,
                "status": "success",
                "old_version": old_version,
                "new_version": new_version,
                "old_run_id": run_id[:8] if run_id else None,
                "new_run_id": new_run_id[:8],
                "duration_seconds": duration
            })
    
    # Training Summary
    logger.info("")
    logger.info(f"{'=' * 80}")
    logger.info(f"📊 Training Summary")
    logger.info(f"{'=' * 80}")
    logger.info(f"Total models: {results['total']}")
    logger.info(f"✅ Succeeded: {results['succeeded']}")
    logger.info(f"❌ Failed: {results['failed']}")
    logger.info(f"⏭️  Skipped: {results['skipped']}")
    
    # Final app reload with verification
    logger.info("")
    logger.info(f"{'=' * 80}")
    logger.info(f"� Final Verification & Reload")
    logger.info(f"{'=' * 80}")
    
    # Collect all new run_ids for verification
    new_run_ids = [
        detail['new_run_id'] 
        for detail in results['details'] 
        if detail['status'] == 'success' and 'new_run_id' in detail
    ]
    
    # Expand short run_ids to full run_ids for MinIO check
    full_run_ids = []
    if new_run_ids:
        logger.info(f"📋 Verifying {len(new_run_ids)} trained models...")
        for detail in results['details']:
            if detail['status'] == 'success':
                # Get full run_id from MLflow
                try:
                    model_type = detail['model'].split('@')[0]
                    alias = detail['model'].split('@')[1]
                    mv = client.get_model_version_by_alias(
                        MODEL_REGISTRY_NAMES[model_type], alias
                    )
                    full_run_ids.append(mv.run_id)
                except Exception as e:
                    logger.warning(f"Could not get full run_id for {detail['model']}: {e}")
    
    # Verify MinIO presence
    if full_run_ids and not skip_minio_sync:
        logger.info(f"🔍 Checking MinIO for {len(full_run_ids)} models...")
        found, total = verify_minio_presence(full_run_ids)
        logger.info(f"   MinIO status: {found}/{total} models present")
        
        if found < total:
            logger.warning(f"⚠️  Only {found}/{total} models found in MinIO")
        else:
            logger.info(f"✅ All {total} models verified in MinIO")
    
    if not skip_app_reload:
        logger.info("")
        logger.info(f"🔄 Triggering app reload...")
        reload_success, reload_data = trigger_app_reload()
        
        if reload_success:
            logger.info("✅ App successfully reloaded new models")
            
            # Verify the reload actually picked up our models
            reloaded = reload_data.get('reloaded', [])
            failed = reload_data.get('failed', [])
            
            if failed:
                logger.warning(f"⚠️  Some models failed to reload: {failed}")
            
            # Check if our model types are in the reloaded list
            expected_types = set(detail['model'].split('@')[0] for detail in results['details'] if detail['status'] == 'success')
            reloaded_types = set()
            for model_name in reloaded:
                if 'feedforward' in model_name:
                    reloaded_types.add('feedforward_nn')
                elif 'xgboost' in model_name:
                    reloaded_types.add('xgboost')
                elif 'mnlogit' in model_name or 'logistic' in model_name:
                    reloaded_types.add('multinomial_logistic')
            
            missing_types = expected_types - reloaded_types
            if missing_types:
                logger.warning(f"⚠️  Expected model types not reloaded: {missing_types}")
            else:
                logger.info(f"✅ All expected model types reloaded: {expected_types}")
        else:
            logger.warning("⚠️  App reload failed - manual reload may be needed")
    else:
        logger.info("⏭️  Skipping app reload (--skip-app-reload)")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Alias-preserving continuation training for RPS models"
    )
    parser.add_argument(
        "--model-types",
        nargs="+",
        choices=list(MODEL_TRAINERS.keys()),
        help="Model types to train (default: all)"
    )
    parser.add_argument(
        "--aliases",
        nargs="+",
        choices=ALIASES,
        help="Aliases to train (default: all)"
    )
    parser.add_argument(
        "--skip-minio-sync",
        action="store_true",
        help="Skip syncing to MinIO after training"
    )
    parser.add_argument(
        "--skip-app-reload",
        action="store_true",
        help="Skip triggering app reload at the end"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force training even without data-driven triggers (default: always trains)"
    )
    
    args = parser.parse_args()
    
    # Check environment variable for force flag (for CronJob usage)
    force_training = args.force or os.getenv("FORCE_TRAINING", "false").lower() == "true"
    
    # Data-driven training check (unless forced)
    if not force_training:
        # Get database path
        data_path_env = os.getenv("DATA_PATH")
        if data_path_env:
            db_path = Path(data_path_env) / "rps.db"
        else:
            db_path = Path("local/rps.db")
        
        min_new_rows = int(os.getenv("MIN_NEW_ROWS_FOR_TRAINING", "10"))
        min_total_rows = int(os.getenv("MIN_TOTAL_ROWS", "100"))
        
        logger.info("🔍 Checking if training is needed (data-driven mode)...")
        should_train, stats = check_training_needed(str(db_path), min_new_rows, min_total_rows)
        
        logger.info(f"📊 Training data check:")
        lookback = stats.get('lookback_minutes', '?')
        logger.info(f"   Total events: {stats.get('total_rows', 0)}")
        logger.info(f"   New events (last {lookback} min): {stats.get('new_rows_since_cutoff', 0)}")
        logger.info(f"   Thresholds: {min_new_rows} new, {min_total_rows} total")
        logger.info(f"   Training needed: {'✅ Yes' if should_train else '❌ No'}")
        
        if not should_train:
            if ENABLE_AUTOFILL:
                logger.info("🕹️  Insufficient data detected — auto-playing legacy games to top up events")
                api_base = _discover_api_base()
                if not api_base:
                    logger.warning("⚠️  Could not reach game API for auto-fill. Skipping training run.")
                    sys.exit(0)

                autofill_result = _autofill_games_via_api(api_base)
                logger.info(
                    "🎮 Auto-fill summary: played=%s, successes=%s, failures=%s",
                    autofill_result.get("played", 0),
                    autofill_result.get("successes", 0),
                    autofill_result.get("failures", 0),
                )

                if autofill_result.get("successes", 0) == 0:
                    logger.warning("⚠️  Auto-fill did not complete any games successfully. Skipping training.")
                    sys.exit(0)

                recheck_delay = int(os.getenv("TRAINING_SIM_RECHECK_DELAY", "5"))
                if recheck_delay > 0:
                    time.sleep(recheck_delay)

                should_train, stats = check_training_needed(str(db_path), min_new_rows, min_total_rows)
                lookback = stats.get('lookback_minutes', '?')
                logger.info(
                    "📊 Post auto-fill check — Total events: %s, New events (last %s min): %s",
                    stats.get("total_rows", 0),
                    lookback,
                    stats.get("new_rows_since_cutoff", 0),
                )

                if not should_train:
                    logger.info("⏭️  Still below thresholds after auto-fill. Skipping training.")
                    sys.exit(0)

                logger.info("✅ Threshold met after auto-fill — proceeding with training")
            else:
                logger.info("⏭️  Skipping training - insufficient new data (auto-fill disabled)")
                logger.info("   (Enable auto-fill or use --force to train anyway)")
                sys.exit(0)
        else:
            logger.info("✅ Proceeding with training - sufficient new data detected")
    else:
        logger.info("Training mode: FORCED (will train regardless of new data)")
    
    # Run training
    results = train_all_aliases(
        model_types=args.model_types,
        aliases=args.aliases,
        skip_minio_sync=args.skip_minio_sync,
        skip_app_reload=args.skip_app_reload
    )
    
    # Exit with appropriate code
    if results["failed"] > 0:
        logger.error(f"❌ Training completed with {results['failed']} failures")
        sys.exit(1)
    else:
        logger.info(f"✅ All training completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
