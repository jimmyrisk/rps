# app/main.py
"""
RPS Quest MLOps Platform - Refactored FastAPI Application
Kubernetes-native MLOps platform with modular route organization.
"""
import os
from pathlib import Path
from datetime import datetime, timezone
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator  # type: ignore[import-not-found]

# Import modular route handlers
from app.routes.health import router as health_router
from app.routes.ui import router as ui_router
from app.routes.models import router as models_router
from app.routes.predictions import router as predictions_router
from app.routes.game import router as game_router
from app.routes.analytics import router as analytics_router
from app.routes.monitoring import router as monitoring_router
from app.routes.misc import router as misc_router
from app.routes.training import router as training_router
from app.routes.promotion import router as promotion_router

from app.model_serving import get_model_manager

# ---- Configuration
_DEFAULT_DATA_PATH = os.getenv("DATA_PATH", "/data")
DATA_DIR = Path(_DEFAULT_DATA_PATH)
STATIC_DIR = Path(__file__).parent / "static"
ASSET_VERSION = os.getenv("ASSET_VERSION")
if not ASSET_VERSION:
    ASSET_VERSION = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")

# Ensure directories exist, falling back to a local path when /data is not writable
try:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
except PermissionError:
    fallback_dir = Path.cwd() / "data"
    fallback_dir.mkdir(parents=True, exist_ok=True)
    DATA_DIR = fallback_dir
    os.environ["DATA_PATH"] = str(DATA_DIR)
    print(f"Warning: DATA_PATH '{_DEFAULT_DATA_PATH}' not writable; using '{DATA_DIR}' instead")

STATIC_DIR.mkdir(parents=True, exist_ok=True)

# ---- FastAPI App Initialization
app = FastAPI(
    title="RPS Quest MLOps Platform",
    description="Kubernetes-native MLOps platform demonstrating complete ML lifecycle through rock-paper-scissors gameplay",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ---- Middleware Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten later if you want
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Static File Serving
class ImmutableStatic(StaticFiles):
    async def get_response(self, path, scope):
        r = await super().get_response(path, scope)
        # long cache for versioned assets
        r.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return r

# replace the existing mount:
app.mount("/static", ImmutableStatic(directory=str(STATIC_DIR)), name="static")

# ---- Prometheus Instrumentation
Instrumentator(
    should_respect_env_var=True,
    excluded_handlers=["/metrics"],
).instrument(app).expose(app, include_in_schema=False, should_gzip=True)

# ---- Router Registration
# Health and monitoring (no prefix for /healthz)
app.include_router(health_router, tags=["Health"])

# UI routes (no prefix for / and /ui-lite)
app.include_router(ui_router, tags=["UI"])

# Core API routes
app.include_router(models_router, tags=["Models"])
app.include_router(predictions_router, tags=["Predictions"])
app.include_router(game_router, tags=["Game Flow"])
app.include_router(analytics_router, tags=["Analytics"])
app.include_router(training_router, tags=["Training"])
app.include_router(promotion_router, tags=["Promotion"])

# Monitoring and debugging
app.include_router(monitoring_router, tags=["Monitoring"])

# Legacy and miscellaneous
app.include_router(misc_router, tags=["Miscellaneous"])

# ---- Startup Event
@app.on_event("startup")
async def startup_event():
    """Load default model on startup and initialize metrics"""
    # Initialize all Prometheus metrics label combinations
    initialize_all_metrics()
    
    manager = get_model_manager()
    success = manager.load_default_model()
    if success:
        print("Default model loaded successfully")
    else:
        print("Warning: Failed to load default model")

    preload_results = manager.load_all_models()
    failed_models = [m for m, ok in preload_results.items() if not ok]
    if failed_models:
        print(f"Models failed to preload at startup: {failed_models}")
    
    # Backfill model metrics from historical data
    try:
        backfilled_data = manager.backfill_model_metrics_from_db()
        if backfilled_data:
            print(f"Backfilled metrics for {len(backfilled_data)} models on startup")
    except Exception as e:
        print(f"Failed to backfill model metrics on startup: {e}")
    
    # Start periodic metrics refresh to keep time series alive
    start_metrics_keepalive()


def initialize_all_metrics():
    """Initialize all Prometheus metrics label combinations to ensure they export.
    
    Prometheus only exports metrics that have been touched at least once.
    This function initializes all expected label combinations with .inc(0)
    so that metrics exist even with 0 values, allowing dashboards to draw
    flat lines instead of showing "No data".
    """
    try:
        from app.metrics import (
            _MODEL_ALIAS_PREDICTIONS_COUNTER,
            _MODEL_ALIAS_CORRECT_PREDICTIONS_COUNTER,
            _MODEL_ALIAS_GAMES_TOTAL_COUNTER,
            _MODEL_ALIAS_GAME_WINS_COUNTER,
            _MODEL_ALIAS_GAME_LOSSES_COUNTER,
            _PROMOTION_TEST_STAT_GAUGE,
            _PROMOTION_P_VALUE_GAUGE,
            _PROMOTION_DECISIONS_COUNTER,
            _PROMOTION_ALIAS_REORDER_COUNTER,
            _PROMOTION_ALIAS_RANK_GAUGE,
            _PROMOTION_DECISION_STATE_GAUGE,
            _ACTION_PREDICTIONS_COUNTER,
            _ACTION_WINS_COUNTER,
            _ACTION_LOSSES_COUNTER,
            _ACTION_TIES_COUNTER,
            _GAME_WINS_COUNTER,
            _GAME_LOSSES_COUNTER,
            _GAMES_TOTAL_COUNTER,
            _TRAINING_COMPLETED_COUNTER,
        )
        
        model_types = ["xgboost", "feedforward_nn", "multinomial_logistic"]
        aliases = ["Production", "B", "shadow1", "shadow2"]
        difficulties = ["standard"]
        policies = ["brian", "forrest", "logan"]
        moves = ["rock", "paper", "scissors"]
        training_statuses = ["success", "failure"]
        
        # Initialize model-alias level metrics (all 12 combinations)
        for model in model_types:
            for alias in aliases:
                _MODEL_ALIAS_PREDICTIONS_COUNTER.labels(model=model, alias=alias).inc(0)
                _MODEL_ALIAS_CORRECT_PREDICTIONS_COUNTER.labels(model=model, alias=alias).inc(0)
                if alias in ["B", "shadow1", "shadow2"]:
                    _PROMOTION_ALIAS_RANK_GAUGE.labels(model=model, alias=alias).set(0)

                # Only Production and B play full games and expose difficulty labels
                if alias in ["Production", "B"]:
                    for difficulty in difficulties:
                        _MODEL_ALIAS_GAMES_TOTAL_COUNTER.labels(
                            model=model,
                            alias=alias,
                            difficulty=difficulty,
                        ).inc(0)
                        _MODEL_ALIAS_GAME_WINS_COUNTER.labels(
                            model=model,
                            alias=alias,
                            difficulty=difficulty,
                        ).inc(0)
                        _MODEL_ALIAS_GAME_LOSSES_COUNTER.labels(
                            model=model,
                            alias=alias,
                            difficulty=difficulty,
                        ).inc(0)
        
        # Initialize promotion-level gauges/counters
        for model in model_types:
            _PROMOTION_TEST_STAT_GAUGE.labels(model=model).set(0)
            _PROMOTION_P_VALUE_GAUGE.labels(model=model).set(1)
            for decision in [
                "swap_production_b",
                "retain_production",
                "insufficient_data",
                "alias_reorder",
            ]:
                _PROMOTION_DECISIONS_COUNTER.labels(model=model, decision=decision).inc(0)
                _PROMOTION_DECISION_STATE_GAUGE.labels(model=model, decision=decision).set(0)
            _PROMOTION_ALIAS_REORDER_COUNTER.labels(model=model).inc(0)

        # Initialize policy-level metrics
        for policy in policies:
            for model_type in model_types:
                for difficulty in difficulties:
                    # Action-level metrics
                    _ACTION_WINS_COUNTER.labels(
                        policy=policy, model_type=model_type, difficulty=difficulty
                    ).inc(0)
                    _ACTION_LOSSES_COUNTER.labels(
                        policy=policy, model_type=model_type, difficulty=difficulty
                    ).inc(0)
                    _ACTION_TIES_COUNTER.labels(
                        policy=policy, model_type=model_type, difficulty=difficulty
                    ).inc(0)
                    
                    # Game-level metrics
                    _GAME_WINS_COUNTER.labels(
                        policy=policy, model_type=model_type, difficulty=difficulty
                    ).inc(0)
                    _GAME_LOSSES_COUNTER.labels(
                        policy=policy, model_type=model_type, difficulty=difficulty
                    ).inc(0)
                    _GAMES_TOTAL_COUNTER.labels(
                        policy=policy, model_type=model_type, difficulty=difficulty
                    ).inc(0)
                    
                    # Action predictions by move
                    for move in moves:
                        _ACTION_PREDICTIONS_COUNTER.labels(
                            policy=policy, model_type=model_type, difficulty=difficulty, move=move
                        ).inc(0)

        # Initialize training metrics for dashboard visibility
        for model in model_types:
            for status in training_statuses:
                _TRAINING_COMPLETED_COUNTER.labels(model_type=model, status=status).inc(0)
        
        print(f"✅ Initialized {len(model_types) * len(aliases)} model@alias combinations")
        print(f"✅ Initialized {len(policies) * len(model_types) * len(difficulties)} policy combinations")
        print("✅ All Prometheus metrics initialized for export")
        
    except Exception as e:
        print(f"⚠️ Failed to initialize metrics: {e}")
        # Don't fail startup if metrics init fails


import threading

_keepalive_stop = threading.Event()
_keepalive_thread = None

def start_metrics_keepalive():
    """Start thread to ping metrics every 5min (keeps Grafana time series alive)."""
    global _keepalive_thread
    def worker():
        interval = int(os.getenv("METRICS_KEEPALIVE_SECONDS", "300"))
        while not _keepalive_stop.wait(interval):
            try:
                initialize_all_metrics()  # Reuse init function
                print(f"🔄 Metrics keepalive ping")
            except Exception as e:
                print(f"⚠️ Keepalive error: {e}")
    _keepalive_thread = threading.Thread(target=worker, daemon=True)
    _keepalive_thread.start()
    print(f"✅ Metrics keepalive started (every {os.getenv('METRICS_KEEPALIVE_SECONDS', '300')}s)")

def stop_metrics_keepalive():
    """Stop keepalive thread."""
    _keepalive_stop.set()
    if _keepalive_thread and _keepalive_thread.is_alive():
        _keepalive_thread.join(timeout=5)



@app.on_event("shutdown")
async def shutdown_event():
    manager = get_model_manager()
    try:
        manager.stop_metrics_rollup()
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to stop metrics rollup cleanly: {exc}")
    
    try:
        stop_metrics_keepalive()
    except Exception as exc:
        print(f"Failed to stop metrics keepalive cleanly: {exc}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)