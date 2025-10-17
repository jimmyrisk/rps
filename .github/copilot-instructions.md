# GitHub Copilot Instructions — RPS MLOps Platform

**What you're working on:** Production Kubernetes-native MLOps platform for multi-round Rock-Paper-Scissors games

> **⚡ ALWAYS START HERE:** Read `README.md`, `docs/architecture.md`, and `docs/operations.md` before making changes!

**Core principles:**
- Repository is **self-contained** - everything needed lives here
- **Single k3s node** (4GB RAM + 8GB swap) - no horizontal scaling
- Legacy **easy mode** exists only in historical data
- **Execute scripts directly** - don't tell users to run them
- **Verify changes** - always test that changes took effect

---

## 🤖 AI Agent Quick Start

**Before writing code:**
1. Read `README.md` - System overview, quick start, core workflows  
2. Read `docs/architecture.md` - Component design, data flows, storage strategy
3. Read `docs/operations.md` - Deployment procedures, debugging, maintenance
4. Test locally: `conda activate rps && python tests/test_essential.py`

**Critical files:**
- `app/routes/*.py` - FastAPI endpoints (game, predictions, analytics, models, health, monitoring, ui, misc)
- `app/features.py` - 50-feature contract (SACRED - changing count requires 3-4hr retrain)
- `app/model_serving.py` - ModelManager singleton for predictions
- `trainer/base_model.py` - Base class all trainers inherit from
- `trainer/train_*.py` - Model-specific training implementations

**Deployment workflow:**
```bash
# PRIMARY: Docker images (only supported path)
./scripts/build_push_deploy.sh --tag latest --push --deploy

# CLEANUP: Ensure no ConfigMap overlays linger
./scripts/clear_configmap_overlays.sh --dry-run

# VERIFY: After deployment
kubectl -n mlops-poc rollout status deployment/rps-app
kubectl -n mlops-poc logs -f deploy/rps-app --tail=50
```

**Cache busting:** Changes to `app/static/js/` require `ASSET_VERSION` bump:
```bash
# Update manifest or set env var directly
kubectl -n mlops-poc set env deployment/rps-app ASSET_VERSION=$(date +%Y%m%d%H%M%S)
```

---

<a id="platform-overview-60-seconds" name="platform-overview-60-seconds"></a>
## 🧭 Platform Overview (60 seconds)

**System architecture:**
- **12 active models**: 3 types (xgboost, feedforward_nn, multinomial_logistic) × 4 aliases (Production, B, shadow1, shadow2)
- **Dual storage**: MinIO (local, ~400ms first load) + DagsHub MLflow (backup/history)
- **Feature contract**: Exactly 50 features - changing count requires 3-4hr retrain
- **Game mechanics**: Multi-round matches to 10 points (~20-30 rounds), NOT single predictions
- **Two metric types**: Action accuracy (per-round) vs Game wins (full match)

**Common mistakes:**
1. **Data leakage:** Filter `WHERE step_no < current_round` for inference queries
2. **Feature extraction:** `extract_features_stateless()` for `/predict`, `extract_inference_features()` for games
3. **Analytics queries:** Include `WHERE is_test = 0 AND created_ts_utc >= ?`
4. **Metrics imports:** Use `from app.metrics import get_counter` - never direct `prometheus_client`
5. **Round points:** Call `deterministic_round_points(game_id, step_no)` BEFORE move selection
6. **Context managers:** Never return inside `with` blocks - use flag variable
7. **Model serving:** Use `get_model_manager()` singleton - never instantiate directly
8. **Static assets:** `app/static/js/` changes require `ASSET_VERSION` bump

---

## 🚀 Critical Workflows

### Code Change Workflow
```bash
# 1. Setup environment (first time)
conda env create -f environment.yml && conda activate rps

# 2. Test locally
python tests/test_essential.py  # ~5s offline tests
uvicorn app.main:app --reload   # Optional: local API

# 3. Deploy to cluster
./scripts/build_push_deploy.sh --tag latest --push --deploy  # Recommended
# OR manual:
# docker build -f Dockerfile.app -t ghcr.io/jimmyrisk/rps-app:latest . && docker push ghcr.io/jimmyrisk/rps-app:latest
# kubectl set image deployment/rps-app app=ghcr.io/jimmyrisk/rps-app:latest -n mlops-poc

# 4. Verify
kubectl -n mlops-poc rollout status deployment/rps-app
kubectl -n mlops-poc logs -f deploy/rps-app --tail=40
```

### Testing Hierarchy (Fast → Comprehensive)
```bash
python tests/test_essential.py                     # 5s - Core logic
python tests/test_critical_endpoints.py            # 10s - API contracts
python tests/test_comprehensive_games.py           # 90s - All combinations
python tests/test_e2e_metrics.py                   # 120s - Metrics pipeline
python tests/test_phased_validation_v2.py --all    # 8min - Gated validation
python scripts/validate.py --games 500 --batch 50   # 45min - Dashboard data
```

### Model Training & Experimentation
```bash
# Hyperparameter experiments (JSON configs, no auto-promotion)
python trainer/run_configured_training.py --config configs/model_hyperparameters.json --dry-run
python trainer/run_configured_training.py --config configs/model_hyperparameters.json --model-types xgboost feedforward mnlogit

# Manual training with promotion
PROMOTE_ALIAS=Production python trainer/train_xgboost.py

# Continuation training (preserves hyperparameters)
python trainer/train_all_aliases.py  # All 12 models

# Trigger CronJob manually
kubectl create job train-manual-$(date +%s) --from=cronjob/rps-trainer -n mlops-poc
```

### Debugging Workflow
```bash
# System health
kubectl get pods -n mlops-poc
curl https://mlops-rps.uk/healthz
curl https://mlops-rps.uk/metrics | grep rps_

# Logs & verification
kubectl -n mlops-poc logs -f deploy/rps-app --tail=100
kubectl -n mlops-poc describe pod -l app=rps-app | grep Image  # App image
kubectl -n mlops-poc describe cronjob rps-trainer | grep Image # Trainer image (separate!)

# Test in pod
kubectl -n mlops-poc exec -it deploy/rps-app -- python
>>> from app.features import FEATURE_COUNT
>>> print(FEATURE_COUNT)  # Should be 50

# Database check
kubectl -n mlops-poc exec deploy/rps-app -- python -c "
from app.db import connect
conn = connect()
cursor = conn.execute('SELECT COUNT(*) FROM games WHERE is_test = 0')
print(f'Non-test games: {cursor.fetchone()[0]}')
"

# A/B split verification
kubectl -n mlops-poc exec deploy/rps-app -- \
  python -c "import app.model_serving as ms; print(ms.MODEL_AB_SPLIT_RATIOS)"
```

---

## 🎯 Critical Non-Obvious Facts

### Game Mechanics (Most Common Misunderstanding)
- **NOT single predictions** - Games run to 10 points (~20-30 rounds), like a fighting game
- **Point values vary per round** (1.0-2.0 deterministically random) - ALWAYS use `deterministic_round_points()`
- **Two different metrics**: Action accuracy (predicted user's move?) vs Game wins (won to 10 points?)
- **Gambits (rounds 1-3)**: Predetermined openings, NO ML. **ML starts round 4+** with lag-3 lookback

### Training ↔ Production Accuracy Parity (CRITICAL 2025-10-08)
- Deterministic opponents (e.g., `cal`) **must** yield comparable accuracy during training and in production; a persistent ~33% production accuracy with 100% training accuracy indicates a deployment or feature-parity defect, not stochastic variance.
- Treat these divergences as P0: verify MinIO artifacts, alias assignment, and ensure inference feature frames (`extract_inference_features`) match the training dataset ordering and column names.
- Enable server-side capture with `RPS_CAPTURE_INFERENCE_DIR=/tmp/rps_inference` (optional `RPS_CAPTURE_INFERENCE_SAMPLE_RATE=0.05`) to persist JSON snapshots of features + probabilities for side-by-side comparison with training pipelines.
- Cross-check Grafana's `rps_model_correct_predictions_by_alias_total` against MLflow evaluation metrics immediately after each promotion; do not assume sweep test accuracy reflects production reality without verification.

### Deployment Model (Docker Images + Emergency Patches)
```bash
# PRIMARY: Docker images contain all code
# Build images (from workspace root)
docker build -f Dockerfile.app -t ghcr.io/jimmyrisk/rps-app:latest .
docker build -f Dockerfile.trainer -t ghcr.io/jimmyrisk/rps-trainer:latest .
docker build -f Dockerfile.ui -t ghcr.io/jimmyrisk/rps-ui:latest .

# Push to registry
docker push ghcr.io/jimmyrisk/rps-app:latest
docker push ghcr.io/jimmyrisk/rps-trainer:latest
docker push ghcr.io/jimmyrisk/rps-ui:latest

# Update deployments
kubectl set image deployment/rps-app app=ghcr.io/jimmyrisk/rps-app:latest -n mlops-poc
kubectl set image deployment/rps-ui ui=ghcr.io/jimmyrisk/rps-ui:latest -n mlops-poc
# Trainer updates automatically (CronJob pulls latest on each run)
# IMPORTANT: After any trainer Dockerfile, dependency, or `scripts/` change, rebuild and push `ghcr.io/jimmyrisk/rps-trainer:latest`; CronJob jobs run whatever image tag is in the registry.

# Cache busting checklist
- Change any file under `app/static/js/`? Update `ASSET_VERSION` everywhere (manifests + live deployment) before validating the UI.
- Preferred format is a UTC timestamp, e.g. `20251011095530`.
- Fastest rollout: `kubectl -n mlops-poc set env deployment/rps-app ASSET_VERSION=<new value>` then watch the rollout status.
- Verify with `curl -s https://mlops-rps.uk/static/js/rps-lite.js?v=<new value> | head` and a private `/ui-lite-debug` session.
# LEGACY: ConfigMap overlays have been retired. Use `./scripts/clear_configmap_overlays.sh` to remove any remaining `rps-app-*-patch-*` ConfigMaps before deploying new images.
```

### MinIO Cache Scope (READ THIS)
- MinIO is a **cluster-local cache** that only stores the 12 active model artifacts (3 model types × aliases Production/B/shadow1/shadow2). All historical versions live in DagsHub MLflow. See the MinIO operations section in `docs/operations.md` and `docs/MODEL_STORAGE_ARCHITECTURE.md` for the complete flow.
- Expect exactly 12 run IDs when listing `local/mlflow-artifacts/`. Extra artifacts mean alias cleanup failed and needs attention.
- Local machines cannot reach MinIO without a `kubectl port-forward`; off-cluster workflows should pull from DagsHub or call the `/models/reload` endpoint instead of hitting MinIO directly.
- **MinIO Batch Delete Limitation** (Fixed 2025-10-06): MinIO requires Content-MD5 header for batch deletes. Our sync code deletes objects one-by-one to avoid this requirement (see `app/minio_sync.py:delete_run_from_minio()`)
- **Alias Markers** (Added 2025-10-06): Each model has a marker file `{run_id}/.alias_{alias}` containing format "rps_bot_xgboost@Production v94" for O(1) lookups and version tracking
- **Model Name Validation** (Added 2025-10-06): Sync code validates model names in marker content to prevent cross-contamination between model types

### Conda Environment Quirks
- The managed server lacks a system-wide `python` shim—activate the `rps` environment (`conda activate rps`) or call `python3` directly.
- `conda run -n rps python - <<'PY' ... PY` will drop stdout because the heredoc lacks a TTY. Prefer `conda run -n rps python -c "..."`, wrap commands in `conda run -n rps /bin/bash -lc '...'`, or activate the environment before executing scripts. Details live in the "Conda environment quirks" section of `docs/operations.md`.

**Environment setup files:**
- `environment.yml` - Conda environment specification (Python 3.12, all dependencies)
- `requirements.app.txt` - App-specific dependencies (FastAPI, uvicorn, MLflow)
- `requirements.trainer.txt` - Training dependencies (PyTorch, XGBoost, scikit-learn)
- `requirements.ui.txt` - UI dependencies (lightweight FastAPI debug interface)

### 50-Feature Contract (SACRED INVARIANT)
```python
# app/features.py defines FEATURE_COUNT = 50
# Shared between app (serving) and trainer (training)
# NEVER change without retraining ALL 3 models (3-4 hours)
# Column order MATTERS - the fiftieth feature (index 49) MUST be the legacy easy_mode flag
```

### Model Architecture Pattern
All trainers inherit from `trainer/base_model.py:BaseRPSModel` to eliminate duplication:
- `get_hyperparameters()` - From env vars (EPOCHS, LR, L2_LAMBDA, LAMBDA_DANGER, etc.)
- `needs_feature_scaling()` - True for logistic/NN, False for trees
- `run_cross_validation()` - Game-stratified CV (NOT time-series!)
- Base class handles: data loading (excludes `is_test=1`), feature extraction, MLflow tracking, alias assignment

**3-Step Training Pipeline (Updated 2025-10-06):**
1. **STEP 1: Hold-out evaluation** - Train on 60%, validate on 20%, test on 20% (chronological)
2. **STEP 2: Production model** - Train on 80%, validate on 20% (game-stratified, every 5th game)
3. **STEP 3: Cross-validation** - 3-fold game-stratified CV on training subset

**Critical:** Production model (STEP 2) now uses validation monitoring for better generalization assessment

### JSON-Driven Hyperparameter Configuration (NEW - 2025-10-05)
**Central config:** `configs/model_hyperparameters.json` - Single source of truth for all hyperparameter experiments

**Key features:**
- **Reproducible experiments:** All hyperparameters + alias assignments in version-controlled JSON
- **Orchestrated training:** `train_all_models.py` runs all configs sequentially with MLflow tracking
- **Explicit alias promotion:** Each config declares `promote_alias` (Production/B/shadow1/shadow2)
- **MLflow tags:** Automatic tagging for experiment families, config IDs, target aliases

**Example workflow:**
```bash
# 1. Edit configs/model_hyperparameters.json to define experiments
# 2. Run orchestrated training
python trainer/train_all_models.py --model-type feedforward

# 3. Compare results in MLflow UI
# https://dagshub.com/jimmyrisk/rps.mlflow

# 4. Best configs automatically promoted to their target aliases
# 5. Sync to MinIO for serving
curl -X POST https://mlops-rps.uk/models/reload
```

**Critical distinction:**
- `train_all_models.py` = NEW models from JSON configs (hyperparameter exploration)
- `train_all_aliases.py` = UPDATE existing aliased models (continuation training)
- CronJob uses `run_staggered_alias_training.py` which calls `train_all_aliases.py` per alias


### Training Strategy (3-Step Evaluation Pipeline)
**Current implementation (Updated 2025-10-07):**

**STEP 1: Hold-out evaluation (60/20/20 chronological split)**
- 60% training, 20% validation, 20% test (oldest → newest)
- Tests temporal generalization (can model predict future behavior?)
- MLflow metrics: `val_acc_holdout`, `test_acc_holdout`, `test_accuracy`
- Most pessimistic estimate (least data, hardest test)

**STEP 2: Production model (80/20 game-stratified split)**
- 80% training (4 of every 5 games), 20% validation (every 5th game)
- Game-stratified (entire games held out together, NOT chronological)
- Uses validation for early stopping (better generalization)
- MLflow metrics: `train_acc_production`, `val_acc_production`
- **This is the model that gets registered/promoted to aliases**

**STEP 3: Cross-validation (3-fold on 60% training subset)**
- Uses only the STEP 1 training data (60% of total)
- Game-stratified folds (entire games stay together)
- MLflow metrics: `cv_mean_acc`, `cv_std_acc`, `cv_fold_1_acc`, etc.
- Provides confidence intervals and stability assessment

**MLflow Metrics Reference:**
```python
# For model selection (hyperparameter tuning):
cv_mean_acc           # PRIMARY: Sort by this (higher = better)
cv_std_acc            # SECONDARY: Lower = more stable
val_acc_production    # Production readiness check
test_acc_holdout      # Temporal generalization sanity check

# Model-specific (multinomial_logistic only):
train_acc_final       # Final training accuracy after convergence
val_acc_final         # Final validation accuracy after early stopping
train_loss, val_loss  # Logged every 10 epochs during training

# Alias for test accuracy (same value):
test_accuracy == test_acc_holdout  # Both refer to STEP 1 test set
```

**Key Differences Between Metrics:**
- `val_acc_holdout` (STEP 1) uses **chronological** validation → harder, tests future prediction
- `val_acc_production` (STEP 2) uses **game-stratified** validation → easier, production estimate
- `cv_mean_acc` (STEP 3) provides **robust estimate** with confidence intervals

**Implementation:**
- `trainer/base_model.py:train()` - 3-step orchestrator
- `trainer/validation_utils.py:game_stratified_cv_split()` - Greedy bin-packing for balanced folds
- All trainers inherit from `BaseRPSModel` and implement:
  - `run_cross_validation()` - Game-stratified CV with weighted metrics
  - `train_final_model()` - Handles optional validation (None for production model)
- Production model (STEP 2) uses validation monitoring for early stopping

**Model-Specific Early Stopping:**
- **XGBoost**: `early_stopping_rounds` parameter, passes `eval_set` when validation provided
- **Feedforward NN**: `patience` parameter, creates training plots when validation available
- **Multinomial Logistic**: Logs train/val every 10 epochs, reduces LR on plateau

### Dual Storage Strategy (MinIO Cache + MLflow Archive)
```python
# CRITICAL: MinIO contains ONLY the 12 currently promoted models (3 types × 4 aliases)
# Old models are automatically removed when aliases change

# Storage tiers:
# 1. MLflow (DagsHub) - Source of truth, all versions forever
# 2. MinIO S3 (local) - Fast cache, only aliased models (Production/B/shadow1/shadow2)
#    - Model artifacts: {run_id}/model/
#    - Alias markers: {run_id}/.alias_{alias} (format: "rps_bot_xgboost@Production v94")
# 3. Memory cache - In-process, cleared on pod restart

# Model promotion flow:
# 1. Train → Register in MLflow → Assign alias (Production/B/shadow1/shadow2)
# 2. Auto-sync to MinIO (via trainer or /models/reload endpoint)
# 3. Old models cleaned from MinIO (only current 12 aliased models kept)

# Serving priority: MinIO (~4s first load) → Memory (<0.001s) → MLflow DagsHub (~8s fallback)

# MinIO Gotchas (CRITICAL):
# 1. Batch delete requires Content-MD5 header → Use one-by-one delete_object()
# 2. Alias markers MUST be validated for model name to prevent cross-contamination
# 3. Version extraction requires reading marker file content, not just finding it

# Model metadata tracking (includes version!)
cache_key = f"{model_type}@{alias}"  # e.g., "feedforward_nn@Production"
model_metadata = manager.model_metadata.get(cache_key, {})
version = model_metadata.get("version")  # e.g., "240" (extracted from marker)
run_id = model_metadata.get("run_id")    # e.g., "a091693d..."
```

### Model Info in API Responses
```python
# ALWAYS include version in model_info dict for UI display
"model_info": {
    "model_type": model_type,           # "feedforward_nn"
    "model_alias": selected_alias,      # "Production" | "B" | "shadow1" | "shadow2"
    "model_version": model_version,     # "63" ← REQUIRED for UI
    "model_run_id": run_id[:8],         # "a091693d" ← Truncated for readability
    "difficulty_mode": difficulty_mode,
    "policy": effective_policy,
    "probability_source": source,       # "mlflow" | "minio" | "fallback"
}
```

### Critical Python Patterns (Project-Specific)

**1. Context Manager Returns (Common Bug)**
```python
# ❌ WRONG - Return inside with block
def train(self):
    with mlflow.start_run():
        # training code...
        return True  # BUG: Returns None!

# ✅ CORRECT - Flag variable, return outside
def train(self):
    success = False
    with mlflow.start_run():
        # training code...
        success = True
    return success  # Works correctly
```

**2. Database Test Filtering (ALWAYS Required)**
```python
# ❌ WRONG - Includes test games
query = "SELECT * FROM games"

# ✅ CORRECT - Exclude test data and respect date cutoff
from app.config import get_metrics_since_date
query = """
    SELECT * FROM games 
    WHERE is_test = 0 
    AND created_ts_utc >= ?
"""
cursor.execute(query, (get_metrics_since_date(),))
```

**3. Model Serving Pattern (Singleton)**
```python
# ❌ WRONG - Creating new instance
manager = ModelManager()  # Don't do this

# ✅ CORRECT - Use singleton accessor
from app.model_serving import get_model_manager
manager = get_model_manager()  # Reuses existing instance
```

**4. Metrics Registration (Import Pattern)**
```python
# ❌ WRONG - Direct prometheus_client import
from prometheus_client import Counter
my_counter = Counter(...)  # Registration conflicts!

# ✅ CORRECT - Use app.metrics helpers
from app.metrics import get_counter
my_counter = get_counter("rps_my_metric", "Description", labelnames=["label"])
my_counter.labels(label="value").inc()
```

**5. Round Points (Game Logic)**
```python
# ❌ WRONG - Forgot to get round points
bot_move = choose_bot_move(...)  # Uses default 1.0 for all moves

# ✅ CORRECT - Get round points FIRST
from app.game_utils import deterministic_round_points
round_pts = deterministic_round_points(game_id, step_no)
bot_move = choose_bot_move(..., round_values=round_pts)
user_delta = round_pts.get(user_move, 1.0) if result == "win" else 0.0
```



### Model Promotion & Auto-Promotion

**Post-training auto-promotion (2025-10-12 update):**
- `trainer/run_staggered_alias_training.py` now launches `scripts/auto_promote_models.py` immediately after all aliases finish training (unless `DISABLE_AUTO_PROMOTION=true`).
- The script still computes the z/p statistics for observability, but the proof-of-concept decision rule simply compares Production and B win rates once both aliases have logged at least three games. Whichever alias leads in win rate becomes Production; ties favour the incumbent. Keep the z-score handy for dashboards and regression alerts.
- Challenger slots (B, shadow1, shadow2) are re-ranked by observed action accuracy so `B` always holds the next-best challenger. Ties preserve the existing order.
- Metrics + decisions are published via `POST /internal/promotion/report`, updating new Prometheus series: `rps_model_promotion_z_statistic`, `rps_model_promotion_p_value`, `rps_model_promotion_events_total`, `rps_model_alias_reorders_total`, and `rps_model_alias_rank`.
- Promotions trigger `/models/reload` automatically so serving pods pick up the new alias mapping. Set `AUTO_PROMOTION_VERBOSE=true` to mirror the script's detailed logs in CronJob output.

**Promotion ledger exposure & backfill (READ BEFORE DASHBOARD WORK):**
- The Grafana JSON datasource `rps-promotion-ledger` reads from `/internal/promotion/history`; set the data source base URL to `https://mlops-rps.uk` and keep panel queries relative so you don't hit the `https://https://…` double-scheme bug. Confirm the endpoint exists in OpenAPI before debugging dashboards.
- If `/internal/promotion/history` is missing, ship a new image that includes matching updates to `app/routes/promotion.py` and `app/db.py` so the route and persistence layer stay in sync.
- Backfill the ledger from an NDJSON export inside the pod to avoid sqlite tooling drift:
  ```bash
  kubectl -n mlops-poc cp ./exports/promotion_ledger.ndjson deploy/rps-app:/tmp/promotion_ledger.ndjson
  kubectl -n mlops-poc exec deploy/rps-app -- bash -lc "python /app/scripts/load_promotion_ledger_ndjson.py /tmp/promotion_ledger.ndjson"
  ```
- Spot-check the database with the in-pod Python snippet from `docs/operations.md` instead of the sqlite CLI; it enforces the right pragmas and prints the most recent events cleanly.
- After seeding, hit `/internal/promotion/history?limit=5` to verify the API response, then reload Grafana—the panels should resume once the datasource sees valid JSON.

### Alias-Preserving Continuation Training (NEW!)
```bash
# Train all 12 models (3 types × 4 aliases) while preserving existing hyperparameters
python trainer/train_all_aliases.py
# → Loads each aliased model → Extracts hyperparameters → Retrains on current data
# → Registers new version → Re-assigns same alias → Syncs to MinIO → Reloads app

# Train specific model types only
python trainer/train_all_aliases.py --model-types xgboost feedforward_nn

# Train specific aliases only (e.g., only Production and B)
python trainer/train_all_aliases.py --aliases Production B

# Train single model@alias combination
python trainer/train_all_aliases.py --model-types xgboost --aliases Production

# Skip MinIO sync (for testing)
python trainer/train_all_aliases.py --skip-minio-sync

# Skip app reload (for batch operations)
python trainer/train_all_aliases.py --skip-app-reload
```

**How it works:**
1. **Load existing model**: Gets model version by alias from MLflow registry
2. **Extract hyperparameters**: Reads original training hyperparameters from run metadata
3. **Continue training**: Trains on current data using existing hyperparameters (overlap OK)
4. **Register new version**: Saves as new MLflow version with incremented version number
5. **Re-assign alias**: Atomically points the same alias to the new version
6. **Sync to MinIO**: Mirrors new model to MinIO for fast serving (~4s first load)
7. **Reload app**: Triggers `/models/reload` endpoint to pick up new versions

**Use cases:**
- **Automated training**: CronJob calls this to update all models with new data
- **Manual refresh**: After adding new training data, refresh all models
- **Staged rollout**: Train only specific aliases (e.g., shadow1/shadow2 first, then Production)

## 🔍 Key Files

```
app/
  main.py              # FastAPI app, router registration, startup hooks
  features.py          # 50-feature contract (NEVER change without retraining)
  model_serving.py     # ModelManager singleton - ALWAYS use for predictions
  policies.py          # EV calculation, POLICY_MODEL_MAP
  legacy_models.py     # User emulation (ace/bob/cal/dan/edd/fox/gus/hal)
  routes/
    game.py            # Stateful multi-round games (/start_game, /play)
    predictions.py     # Stateless predictions (/predict, /predict-with-policy)
    analytics.py       # Stats & leaderboard (/stats, /leaderboard)
    models.py          # Model management (/models/reload, /models/status)
    health.py          # Health checks (/healthz, /ready)
    monitoring.py      # Metrics endpoint (/metrics)
    ui.py              # Web interfaces (/, /ui-lite, /ui-lite-debug)
    misc.py            # Utilities (/rand, /info, /debug)
  db.py                # SQLite schema, idempotent migrations
  config.py            # Environment configuration, MLflow URIs

trainer/
  base_model.py        # INHERIT from BaseRPSModel for all trainers
  train_*.py           # Individual model trainers (xgboost, feedforward, mnlogit)
  train_all_models.py  # JSON-driven hyperparameter orchestration (NEW models)
  train_all_aliases.py # Continuation training (UPDATE existing 12 aliased models)
  run_staggered_alias_training.py  # Sequential training for CronJob
  validation_utils.py  # Time-series splits, feature prep, label encoding
  model_defs/
    pyfunc_wrap.py     # MLflow pyfunc wrappers for all model types

tests/
  test_essential.py              # 5s offline tests (FASTEST)
  test_critical_endpoints.py     # 10s API contracts
  test_comprehensive_games.py    # 90s all bot×difficulty combos
  test_e2e_metrics.py            # 120s full metrics pipeline
  test_phased_validation_v2.py   # 8min gated testing (RECOMMENDED)

scripts/
  sync_promoted_models_to_minio.py     # Mirror aliased models to MinIO
  register_models_and_aliases.py       # Batch alias registration
  populate_dashboard_metrics.py        # Generate 500 games for dashboard
  preflight_check.py                   # Pre-deployment validation
  (See scripts/README.md for complete listing)

configs/
  model_hyperparameters.json     # Central config for all hyperparameter experiments
```

<a id="code-examples-from-codebase" name="code-examples-from-codebase"></a>
## 📝 Code Examples from Codebase

### How to Add a New Model Trainer
```python
# trainer/train_my_model.py - Follow this pattern
from trainer.base_model.py import BaseRPSModel

class MyRPSModel(BaseRPSModel):
    def __init__(self):
        super().__init__(model_name="rps_bot_mymodel", model_type="mymodel")
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Extract hyperparameters from environment variables"""
        return {
            "epochs": int(os.getenv("EPOCHS", "100")),
            "lr": float(os.getenv("LR", "0.001")),
            # ... other hyperparameters
        }
    
    def needs_feature_scaling(self) -> bool:
        """Return True for neural nets/logistic, False for trees"""
        return True  # If your model needs normalized features
    
    def train_final_model(self, X_train, y_train, X_val, y_val):
        """Model-specific training logic"""
        params = self.get_hyperparameters()
        # Your training code here
        self.trained_model = YourModel(**params)
        self.trained_model.fit(X_train, y_train)
    
    def predict(self, X) -> np.ndarray:
        """Return probabilities for [rock, paper, scissors]"""
        return self.trained_model.predict_proba(X)
    
    def create_pyfunc_model(self, scaler, label_map):
        """Wrap for MLflow serving - see model_defs/pyfunc_wrap.py"""
        from trainer.model_defs.pyfunc_wrap import MyModelWrapper
        return MyModelWrapper(self.trained_model, scaler, label_map)

# Base class handles: data loading, time-series split, MLflow tracking,
# model registration, alias assignment, MinIO sync
```

### How to Get Model Predictions (In Routes)
```python
# app/routes/predictions.py or app/routes/game.py
from app.model_serving import get_model_manager
from app.features import extract_inference_features
from app.policies import POLICY_MODEL_MAP

# 1. Get model type from policy name
model_type, difficulty = POLICY_MODEL_MAP.get(policy_name)  # e.g., "feedforward_nn", "standard"

# 2. Extract 50 features from game history
features = extract_inference_features(
    user_moves=user_history,  # ["rock", "paper", ...]
    bot_moves=bot_history,
    results=result_history,   # ["win", "lose", "draw", ...]
    round_values={"rock": 1.5, "paper": 1.2, "scissors": 1.0},  # Current round points
    difficulty_mode=difficulty
)

# 3. Get model predictions with alias support
manager = get_model_manager()
alias = manager.select_model_alias_for_policy(model_type, game_id, session_id)  # A/B testing
probs = manager.predict_with_alias(model_type, alias, features)  # {"rock": 0.3, "paper": 0.5, "scissors": 0.2}
```

### How Features Are Engineered (50-Feature Contract)
```python
# app/features.py - NEVER change FEATURE_COUNT without retraining ALL models

FEATURE_COUNT = 50  # SACRED INVARIANT

# Two extraction functions - use the right one for your context:

def extract_inference_features(game_id, round_no, round_values, cursor, difficulty_mode):
    """
    STATEFUL: For multi-round games (/start_game, /play endpoints)
    Reads history from database via cursor
    CRITICAL: Filters WHERE step_no < round_no to prevent data leakage
    Returns: pd.DataFrame with 50 features
    """
    # Query database for historical events (ONLY past rounds!)
    events = cursor.execute("""
        SELECT user_move, bot_move, result, ...
        FROM events 
        WHERE game_id = ? AND step_no < ?
        ORDER BY step_no ASC
    """, (game_id, round_no)).fetchall()
    # ... build features from history

def extract_features_stateless(user_moves, bot_moves, results, round_values, difficulty_mode):
    """
    STATELESS: For /predict endpoint (no database)
    Takes history arrays directly as parameters
    Handles lag-3 lookback with padding for short histories
    Returns: pd.DataFrame with 50 features
    """
    # No database - use provided arrays
    # Pad if history < 3 rounds
    # ... build same 50 features
    
# 50-feature breakdown (order MATTERS!):
# Lag-3 lookback (27 features): user/bot moves + results for last 3 rounds
# Current round point values (3 features): rock_pts, paper_pts, scissors_pts
# Lagged round points (9 features): lag-1, lag-2, lag-3 point values
# User tendencies (3 features): rock/paper/scissors percentages over all history
# Favored move analysis (3 features): tend to pick favored, beater of favored, etc.
# Score context (4 features): score_diff, user_pts_to_win, bot_pts_to_win, step_no
# Legacy easy mode flag (1 feature) — MUST be the fiftieth feature (index 49)
```

## 📚 Full Reference

For deep dives, see:
- `docs/METRICS_ARCHITECTURE.md` - Prometheus metrics, recording flow
- `docs/HYPERPARAMETER_EXPERIMENTS.md` - Experiment framework, danger penalty
- `docs/MODEL_PROMOTION_WORKFLOW.md` - Alias promotion, reload mechanics
- `READY_FOR_EXPERIMENTS.md` - Quick start for model optimization
- `QUICK_REFERENCE.md` - Fast lookups

**Production**: https://mlops-rps.uk | **SSH**: `ssh root@65.21.151.52` | **Grafana**: https://jimmyrisk41.grafana.net/d/18dc23df-eea3-406c-a7e7-99066a987538/rps-mlops-poc-since-pod-restart | **MLflow**: https://dagshub.com/jimmyrisk/rps.mlflow

---

**Note:** This is a single-server MLOps POC. Resources are limited, but technical precision still matters. When in doubt, check logs and metrics to validate system health.
