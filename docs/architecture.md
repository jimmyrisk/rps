# Architecture Guide

This guide expands the overview in `README.md` with a focus on topology, data flow, and component responsibilities. Deployment workflows live in `docs/operations.md`; observability specifics live in `docs/metrics.md`.

## Deployment Topology

- **Cluster** – single-node k3s host (`ubuntu-4gb`, 4 GiB RAM + 8 GiB swap) that runs every workload.
- **Namespace** – `mlops-poc` isolates gameplay, trainers, and observability.
- **Core workloads**
  - `rps-app` Deployment: FastAPI gameplay service, `/metrics`, Lite UI.
  - `rps-ui` Deployment: optional Streamlit-style debug surface behind `/ui-lite` and `/ui-lite-debug`.
  - `rps-trainer` CronJob: sequentially launches per-alias Jobs using the trainer image and MLflow registry.
  - MinIO StatefulSet: S3-compatible cache for promoted model artifacts.
  - Grafana Alloy sidecar: scrapes `/metrics` and sends data to Grafana Cloud.
- **Persistent volumes**
  - `data-pvc` (1 Gi) – SQLite gameplay database (`/data/rps.db`).
  - `minio-storage-pvc` (20 Gi) – MinIO object storage (`mlflow-artifacts/`).

## Major Components

| Area | Location | Responsibility |
| --- | --- | --- |
| API bootstrap | `app/main.py` | Starts FastAPI, registers routers, enables Prometheus instrumentation. |
| Gameplay routes | `app/routes/*.py` | `/start_game`, `/play`, `/stats`, `/models/*`, `/internal/promotion/*`, Lite UI. |
| Feature pipeline | `app/features.py`, `trainer/validation_utils.py` | Shared 50-feature contract for training and inference. |
| Model serving | `app/model_serving.py` | `ModelManager` singleton, MinIO/MLflow loading, in-process cache, alias routing. |
| Policies | `app/policies.py` | Converts predicted user move probabilities into deterministic bot actions. |
| Persistence | `app/db.py` | SQLite connection, migrations, gameplay queries, promotion ledger storage. |
| Training | `trainer/base_model.py`, `trainer/train_*.py`, `trainer/run_staggered_alias_training.py` | Common pipeline, model-specific implementations, CronJob entrypoint. |
| Promotion automation | `app/auto_promotion.py`, `scripts/auto_promote_models.py` | Evaluates Production vs B win rates, reorders challenger aliases, records results. |

## Model Families & Policies

- **Feedforward neural network** (`rps_bot_feedforward`) → policy `brian`.
- **XGBoost** (`rps_bot_xgboost`) → policy `forrest`.
- **Multinomial logistic regression** (`rps_bot_mnlogit`) → policy `logan`.

Each family maintains four aliases (`Production`, `B`, `shadow1`, `shadow2`). `ModelManager.select_model_alias_for_policy()` assigns a deterministic alias per game so the same model responds throughout the match. Shadow aliases log accuracy for observability but do not influence gameplay decisions.

## Gameplay Loop & Feature Contract

- Games run to 10 points and average 20–30 rounds. Rounds 1–3 follow scripted gambits; machine learning takes over starting round 4 with a lag-3 lookback.
- `deterministic_round_points(game_id, step_no)` returns round multipliers (1.0–2.0). Call it before requesting predictions so models see accurate value weights.
- `FEATURE_COUNT = 50`, with index 49 reserved for the legacy easy-mode flag. Features include lagged moves/results, current and lagged round values, score context, and historical move tendencies.
- `extract_inference_features()` (stateful) reads prior rounds from SQLite while `extract_features_stateless()` accepts history arrays; shared helpers keep training and inference tensors aligned.

## Training Pipeline & Data Flow

1. **Ingestion** – Gameplay routes persist rounds and metadata to SQLite (production analytics ignore `is_test = 1`).
2. **Dataset build** – Trainers query SQLite, exclude gambit rounds, and honour the metrics window from `app/config.py`.
3. **Training** – `BaseRPSModel.train()` follows a three-step pipeline:
   - Step 1: chronological 60/20/20 split for hold-out evaluation (temporal generalisation check).
   - Step 2: 80/20 game-stratified training for the promotion candidate with validation monitoring/early stopping.
   - Step 3: three-fold game-stratified cross-validation over the Step 1 training subset to measure stability.
4. **Artifact logging** – Runs log metrics/parameters to DagsHub MLflow and sync active artifacts to MinIO for serving.
5. **Alias promotion** – Trainers register MLflow aliases, write `.alias_<name>` markers in MinIO, and call `/models/reload`.
6. **Serving** – `ModelManager` loads from MinIO (~400 ms cold start) with MLflow fallback, caches models in memory, and honours per-game alias assignments.
7. **Observability** – `app/metrics.py` exposes Prometheus counters; auto-promotion posts summaries to `/internal/promotion/report`, which persist in SQLite and surface through `/internal/promotion/history` for Grafana’s JSON datasource.

## Storage & Alias Strategy

| Tier | Technology | Contents | Notes |
| --- | --- | --- | --- |
| Serving cache | MinIO (`mlflow-artifacts/`) | Exactly 12 promoted models plus `.alias_*` markers. | Keep lean; `scripts/sync_promoted_models_to_minio.py --clean` enforces it. |
| Source of truth | DagsHub MLflow registry | Complete experiment history (runs, metrics, artifacts). | Use for audits and disaster recovery. |
| Gameplay state | SQLite PVC (`/data/rps.db`) | Games, events, promotion ledger. | Shared by API and trainers. |

## Operational Guardrails

- Single-node resource limits require serialized heavy jobs (`concurrencyPolicy: Forbid` for trainers).
- Prefer Docker image rollouts. ConfigMap overlays are deprecated; run `./scripts/clear_configmap_overlays.sh --dry-run` to confirm no residual mounts before deploying.
- `ENABLE_AB_TESTING` toggles alias routing; defaults keep deterministic per-game assignments.
- When MinIO and MLflow are both unavailable, the API serves uniform probabilities and increments fallback metrics to surface the degradation.

## Related References

- `README.md` – prerequisites, installation steps, and quick-start workflows.
- `docs/operations.md` – deployment runbooks, cache busting, CronJob management, promotion workflows.
- `docs/metrics.md` – Prometheus catalogue, PromQL snippets, Grafana deployment flow.
- `.github/copilot-instructions.md` – sacred invariants (50-feature contract, single-node constraints, promotion expectations).