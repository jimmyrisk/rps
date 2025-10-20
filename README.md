# RPS Quest — Kubernetes-Native MLOps Proof of Concept

RPS Quest is a full-stack Rock-Paper-Scissors platform that demonstrates how to run an end-to-end MLOps system on a single Kubernetes node. The service plays full games to 10 points (usually 20-30 rounds), manages 12 production model aliases, and surfaces live metrics through Grafana Cloud.  You can see it in action [here](https://mlops-rps.uk/ui-lite).

This repository contains the all that is necessary to get this running (assuming you have the ability to host some required services):
- FastAPI gameplay service, 
- model definitions and hyperparameter configurations,
- model training jobs and automation setups, 
- data handling and validation harnesses, 
- Kubernetes manifests, and
- operational tooling required to reproduce the production environment.

---

## File Overview

- **Gameplay API & engine** (`app/`) - multi-round matches, policies, metrics, debug UI endpoints.
- **Feature pipeline** (`app/features.py`, `trainer/validation_utils.py`) - 50-feature contract shared by training and inference.
- **Model training** (`trainer/`, `scripts/run_config_sweep.py`) - three model families (feedforward NN, XGBoost, multinomial logistic) with Production/B/shadow aliases and auto-promotion.
- **Storage** - SQLite gameplay DB, MinIO cache for the 12 live models, and DagsHub MLflow for experiment tracking and artifact retention.
- **Operations toolkit** (`infra/k8s/`, `scripts/`, `ops/`) - manifests, deployment scripts, CronJobs, and Grafana dashboard automation.

For fuller details, read

- `docs/architecture.md` for a deep dive into the topology and data flow, and 
- `docs/operations.md` for day-two runbooks.

---

## Big Picture Architecture

| Layer | Purpose | Notes |
| --- | --- | --- |
| API & Game Engine | FastAPI app (`app/main.py`, `app/routes/*`) serving `/start_game`, `/play`, `/metrics`, `/ui-lite`, etc. | Game loop enforces gambits (rounds 1-3) before ML predictions; use `get_model_manager()` for all inference calls. |
| Feature Contract | `app/features.py`, shared helpers in the trainer | Exactly 50 ordered features; index 49 remains the legacy easy-mode flag for backward compatibility. |
| Model Lifecycle | `trainer/base_model.py`, `trainer/train_*.py`, CronJob orchestrators | Training writes artifacts to MinIO, logs runs to MLflow, assigns aliases (Production/B/shadow1/shadow2), and calls `scripts/auto_promote_models.py` to evaluate swaps. |
| Storage | SQLite on `data-pvc`, MinIO (`mlflow-artifacts/` bucket), DagsHub MLflow | MinIO caches only the 12 aliased models with `.alias_<name>` markers; DagsHub preserves history. |
| Observability | `/metrics`, Grafana dashboard (`ops/grafana-dashboard.json`), promotion ledger API | Prometheus counters drive Grafana Cloud; `app/routes/promotion.py` exposes history for dashboards and audits. |

---

## Prerequisites

- Linux/macOS workstation with Git, Docker ≥24, kubectl, and Conda (or Mambaforge).
- Access to a single-node k3s cluster (4 GB RAM + 8 GB swap recommended). The production reference host is `65.21.151.52`.
- Credentials:
  - GHCR push rights for `ghcr.io/jimmyrisk/rps-*` images.
  - DagsHub MLflow token (set via `MLFLOW_TRACKING_TOKEN` or config file).
  - Grafana Cloud API key stored as `GRAFANA_API_KEY`, `ops/.grafana_api_key`, or `~/.config/rps/grafana_api_key` for dashboard deployments.
  - MinIO access/secret keys for the in-cluster object store (`infra/k8s/minio/02-secret.yaml`).
- DNS/TLS provisioning for your ingress host (update `infra/k8s/12-app-ingress.yaml`). Cloudflare is optional but recommended for HTTPS termination.

---

## Installation & Cluster Bootstrap

1. **Clone and set up Python tooling**
	```bash
	git clone https://github.com/jimmyrisk/rps.git
	cd rps
	conda env create -f environment.yml
	conda activate rps
	```

2. **Provision k3s and configure kubectl**
	- Install k3s on the target host (`curl -sfL https://get.k3s.io | sh -`).
	- Copy `/etc/rancher/k3s/k3s.yaml` to your workstation as `~/.kube/config` and update the server address to the cluster’s public IP or domain.

3. **Prepare Kubernetes secrets**
	- Copy `infra/k8s/01-secrets.example.yaml` to `infra/k8s/01-secrets.yaml` and fill in values for:
	  - `MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD`
	  - `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY`
	  - `PROMETHEUS_BEARER_TOKEN` (Grafana Alloy scrape)
	  - Any other environment variables referenced by `app/config.py`.
	- Copy `infra/k8s/minio/02-secret.example.yaml` to `infra/k8s/minio/02-secret.yaml` and populate matching MinIO credentials.
	- Store TLS certificates or Cloudflare tokens as required by your ingress controller.

4. **Create namespace and persistent volumes**
	```bash
	kubectl apply -f infra/k8s/00-namespace.yaml
	kubectl apply -f infra/k8s/02-pvc.yaml
	kubectl apply -f infra/k8s/05-model-storage-pvc.yaml
	kubectl apply -f infra/k8s/minio/01-pvc.yaml
	```

5. **Deploy MinIO object storage**
	```bash
	kubectl apply -f infra/k8s/minio/02-secret.yaml
	kubectl apply -f infra/k8s/minio/03-deployment.yaml
	kubectl apply -f infra/k8s/minio/04-service.yaml
	kubectl apply -f infra/k8s/minio/05-setup-job.yaml
	kubectl -n mlops-poc logs job/minio-setup --tail=40
	```
	The setup job provisions the `mlflow-artifacts` bucket and applies lifecycle rules for alias markers.

6. **Build and push container images**
	```bash
	./scripts/build_push_deploy.sh --tag latest --push --no-deploy
	```
	This builds `Dockerfile.app`, `Dockerfile.ui`, and `Dockerfile.trainer` and publishes them to GHCR (`latest` tag by default). Supply `--registry` or `--tag` overrides as needed.

7. **Deploy the gameplay API and UI**
	```bash
	kubectl apply -f infra/k8s/10-rps-app.yaml
	kubectl apply -f infra/k8s/11-app-service.yaml
	kubectl apply -f infra/k8s/12-app-ingress.yaml
	kubectl apply -f infra/k8s/30-rps-ui.yaml
	```
	Update `ASSET_VERSION` in the deployment manifest whenever you change files under `app/static/js/`.

8. **Enable scheduled training and telemetry**
	```bash
	kubectl apply -f infra/k8s/20-trainer-cronjob.yaml
	kubectl apply -f infra/k8s/21-individual-trainers.yaml
	kubectl apply -f infra/k8s/22-legacy-gameplay-cronjob.yaml
	```
	The CronJobs retrain aliases sequentially and run legacy-vs-ML matches every 30 minutes for telemetry.

9. **Validate the deployment**
	```bash
	kubectl get pods -n mlops-poc
	curl https://<your-host>/healthz
	curl https://<your-host>/metrics | grep rps_model_predictions_by_alias_total
	./scripts/verify_current_state.sh
	```
	When the app pod starts for the first time, it may take a few seconds to lazily load all 12 models from MinIO.

10. **Wire observability**
	 - Run `./ops/deploy_clean_dashboard.sh` to publish `ops/grafana-dashboard.json` to Grafana Cloud. The script reads the API key from the locations listed in “Prerequisites”.
	 - Confirm the JSON data source `rps-promotion-ledger` points to `https://<your-host>` without path suffixes so relative queries resolve correctly.

At this point the platform mirrors production: gameplay endpoints, model training pipelines, MinIO cache, MLflow tracking, and dashboards are operational.

---

## Using the Platform After Installation

### Local development & smoke tests
- Activate the Conda environment (`conda activate rps`).
- Run the fast safety net: `python tests/test_essential.py` (~5 s).
- Start the API locally if you need interactive debugging: `uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload`.
- Populate metrics or seed dashboards with the validator: `python scripts/validate.py --games 500 --batch 50`.
- Extended validation options:
  - `python tests/test_phased_validation_v2.py --all --verbose`
  - `python scripts/capture_cal_forrest_dataset.py --games 10 --strict-parity --sleep 0.1`
  - `python scripts/verify_legacy_policies.py`

### Deploying code changes
1. Update code and commit changes.
2. Build/push new images: `./scripts/build_push_deploy.sh --tag <tag> --push --deploy` (defaults to `latest`).
3. If you touched `app/static/js/`, increment `ASSET_VERSION` in `infra/k8s/10-rps-app.yaml` (or run `kubectl -n mlops-poc set env deployment/rps-app ASSET_VERSION=<ts>`).
4. Verify rollout: `kubectl -n mlops-poc rollout status deployment/rps-app` and run `./scripts/verify_current_state.sh`.
5. Verify that no ConfigMap overlays remain from previous workflows (`./scripts/clear_configmap_overlays.sh --dry-run`); always ship fixes in a rebuilt image.

### Training, promotion, and model sync
- Trigger the trainer CronJob manually:
  ```bash
  kubectl create job train-manual-$(date +%s) --from=cronjob/rps-trainer -n mlops-poc
  ```
- Run the JSON-driven sweep orchestrator locally: `python trainer/train_all_models.py --model-type feedforward`.
- Refresh existing aliases with continuation training: `python trainer/train_all_aliases.py --aliases Production B`.
- Resync artifacts to MinIO (cleans stale runs): `python scripts/sync_promoted_models_to_minio.py --clean`.
- Reload models in the app pod after external changes: `curl -X POST https://<your-host>/models/reload`.
- Training dataset uses a single cutoff date exposed as `TRAINING_DATA_SINCE_DATE` (default `2025-10-01T00:00:00Z`); there is no rolling seven-day limit.
- `scripts/auto_promote_models.py` evaluates Production vs B win rates (minimum three games per alias) and reorders challengers by action accuracy. Promotions are persisted through `/internal/promotion/report` and MinIO alias markers.

### Observability & debugging
- Health: `curl https://<your-host>/healthz`
- Metrics: `curl https://<your-host>/metrics | grep rps_`
- Pod logs: `kubectl -n mlops-poc logs -f deploy/rps-app --tail=100`
- Training counters: `curl -s https://<your-host>/metrics | grep rps_training_completed_total`
- Promotion ledger: `curl -s https://<your-host>/internal/promotion/history?limit=5`
- Enable inference capture for parity checks:
  ```bash
  kubectl set env deployment/rps-app -n mlops-poc \
	 RPS_CAPTURE_INFERENCE_DIR=/tmp/rps_inference \
	 RPS_CAPTURE_INFERENCE_SAMPLE_RATE=0.05
  kubectl rollout status deployment/rps-app -n mlops-poc
  ```
  Disable by removing the environment variables when finished.

### Data & storage expectations
- Gameplay data lives in SQLite on `data-pvc` (`/data/rps.db`).
- Only 12 models (3 families × 4 aliases) should be present in MinIO; alias markers `.alias_<name>` encode version metadata.
- DagsHub MLflow retains the full experiment history and artifacts.
- The feature contract is immutable without retraining **all** models; ensure any schema changes flow through both `app/features.py` and the trainer modules.

---

## Repository Map

| Path | Description |
| --- | --- |
| `app/` | FastAPI service, game engine, feature pipeline, metrics helpers, promotion ledger routes |
| `trainer/` | Base model class, model-specific trainers, orchestrators, validation utilities |
| `scripts/` | Deployment helpers, model sync, validation harnesses, promotion tooling |
| `infra/k8s/` | Kubernetes manifests for namespace, PVCs, MinIO, app, UI, CronJobs |
| `docs/` | Architecture, operations, API reference, metrics catalogue, audit logs |
| `ui/` | Debug UI (served behind `/ui-lite` and `/ui-lite-debug`) |
| `tests/` | Smoke tests, endpoint contracts, end-to-end validation suites |
| `ops/` | Grafana dashboard JSON and deployment helpers |

---

## Further Reading & Reference

- `docs/architecture.md` - component topology, data flow, storage strategy.
- `docs/operations.md` - deployment, training, troubleshooting, and Grafana workflows.
- `docs/metrics.md` - Prometheus series and dashboard mapping (includes disable instructions if you don't need telemetry).
- `docs/API_REFERENCE.md` - endpoint catalogue.
- `.github/copilot-instructions.md` - agent onboarding, sacred invariants, and coding conventions.

Legacy documentation resides in `docs/archive/`. Session-specific notes live under `session_logs/`.
