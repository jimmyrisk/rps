# Operations Guide

This runbook covers ongoing operations for the production cluster. Installation workflows are in `README.md`; architectural context lives in `docs/architecture.md`; metric definitions and dashboards are documented in `docs/metrics.md`.

---

## Deployment & Rollouts

- **Primary workflow**
	```bash
	./scripts/build_push_deploy.sh --tag <tag> --push --deploy
	```
	Builds all three images, pushes them to GHCR, and restarts `rps-app` and `rps-ui`. Use `--no-deploy` for registry-only pushes or override `--namespace` / `--registry` for alternate clusters.

- **Static asset cache busting**
	- Any change under `app/static/js/` requires a fresh `ASSET_VERSION`.
	- Patch the deployment live: `kubectl -n mlops-poc set env deployment/rps-app ASSET_VERSION=$(date +%Y%m%d%H%M%S)`.
	- Verify rollout: `kubectl -n mlops-poc rollout status deployment/rps-app`.

- **Image rollbacks**
	```bash
	kubectl -n mlops-poc rollout undo deployment/rps-app
	kubectl -n mlops-poc rollout undo deployment/rps-ui
	```

---

## Training & Promotion Operations

- **Training data composition**
	- Legacy bot games (Ace, Bob, Cal, Dan, Edd, Fox, Gus, Hal) are downsampled to 20% (1 in 5 games kept).
	- Human games are always preserved at 100% rate.
	- Historical window is controlled solely by `TRAINING_DATA_SINCE_DATE` (default: `2025-10-01T00:00:00Z`); there is **no** rolling 7-day limit.
	- Configure via `TRAINING_LEGACY_GAME_STRIDE` (default: 5, set to 1 to disable downsampling).
	- Legacy gameplay CronJob runs every 150 minutes (2.5 hours) to generate training data.
	- Downsampling happens in `BaseRPSModel.load_data()` during training, not during data collection.

- **Trigger CronJob manually**
	```bash
	kubectl create job train-manual-$(date +%s) --from=cronjob/rps-trainer -n mlops-poc
	```
- **Local trainers**
	```bash
	python trainer/train_all_aliases.py                   # refresh existing aliases
	python trainer/train_all_models.py --model-type xgboost  # run JSON-config experiments
	```
- **Auto-promotion summary**
	- `scripts/auto_promote_models.py` runs after CronJob completion, comparing Production vs B win rates once each alias logs ≥3 games **since the most recent Production/B swap** (ticker window).
	- Tickers (`rps_model_games_since_production_swap{alias}` and `rps_model_promotion_cycles_since_swap`) advance every cycle, reset only on a Production/B swap, and ignore challenger-only reseeds.
	- Decisions post to `/internal/promotion/report`, persist in SQLite (`promotion_events`), and emit `rps_model_promotion_*` counters (see `docs/metrics.md`).
	- Challenger aliases (`B`, `shadow1`, `shadow2`) are re-ranked by observed action accuracy.
- **Reload or resync**
	```bash
	curl -X POST https://mlops-rps.uk/models/reload                 # refresh ModelManager caches
	python scripts/sync_promoted_models_to_minio.py --clean         # enforce 12-model MinIO cache
	```

---

## Data & Storage Access

| Task | Command |
| --- | --- |
| Inspect gameplay DB | `kubectl exec -n mlops-poc deploy/rps-app -- sqlite3 /data/rps.db "SELECT COUNT(*) FROM games WHERE is_test = 0;"` |
| Snapshot MinIO cache | `kubectl -n mlops-poc exec deploy/minio -- mc ls local/mlflow-artifacts/` |
| Check alias marker | `kubectl -n mlops-poc exec deploy/minio -- mc cat local/mlflow-artifacts/<run>/.alias_Production` |
| Enable inference capture | `kubectl -n mlops-poc set env deployment/rps-app RPS_CAPTURE_INFERENCE_DIR=/tmp/rps_inference RPS_CAPTURE_INFERENCE_SAMPLE_RATE=0.05` |
| Disable capture | `kubectl -n mlops-poc set env deployment/rps-app RPS_CAPTURE_INFERENCE_DIR- RPS_CAPTURE_INFERENCE_SAMPLE_RATE-` |
| Sample promotion ledger | `curl -s https://mlops-rps.uk/internal/promotion/history?limit=5 | jq '.'` |

MinIO should contain exactly 12 run directories (three model families × four aliases) plus `.alias_<name>` marker files. MLflow (DagsHub) remains the source of truth for historical artifacts.

---

## Health Checks & Monitoring

- **Service health**
	```bash
	kubectl get pods -n mlops-poc
	curl -s https://mlops-rps.uk/healthz
	curl -s https://mlops-rps.uk/metrics | grep rps_
	```
- **Promotion dashboard** – Grafana UID `18dc23df-eea3-406c-a7e7-99066a987538`. See `docs/metrics.md` for detailed panel references and alert suggestions.
- **Analytics window** – `METRICS_SINCE_DATE` defaults to `2025-10-10T08:00:00Z`. Leaderboards and `/bot_win_rates` apply the same cutoff and reuse `app/player_filters.py` to exclude legacy/simulation usernames so Grafana panels stay aligned with the UI.
- **End-to-end smoke test** – `./scripts/verify_current_state.sh` checks API readiness, MinIO access, and Prometheus counters in one pass.

---

## Troubleshooting

| Symptom | Checks | Recommended action |
| --- | --- | --- |
| Code changes missing after rollout | `kubectl describe pod`, compare image digests | Re-run `./scripts/build_push_deploy.sh`; run `./scripts/clear_configmap_overlays.sh --dry-run` to verify no ConfigMap overlays remain, then redeploy with a fresh image. |
| Spikes in `rps_model_fallback_total` | `curl -s https://mlops-rps.uk/metrics | grep rps_model_fallback_total` | Inspect app logs for load errors, confirm MinIO availability, run `/models/reload`, then resync artifacts if needed. |
| Accuracy drop vs deterministic bots | `python scripts/capture_cal_forrest_dataset.py --games 10 --strict-parity --sleep 0.1` | Validate feature parity, confirm alias markers in MinIO, check MLflow alias assignments. |
| Trainer jobs failing or queued | `kubectl get jobs -n mlops-poc`, `kubectl logs -n mlops-poc -l app=rps-trainer --tail=100` | Ensure PVC mounted, verify environment variables, rebuild trainer image if dependencies changed. |
| Grafana panels flatline | `kubectl -n mlops-poc logs -l app=alloy --tail=100` | Restart Alloy, confirm scrape annotations on `rps-app`, ensure `ENABLE_METRICS` remains `true`. |

---

## Quick Commands

```bash
# Deployments
kubectl -n mlops-poc set image deployment/rps-app app=ghcr.io/jimmyrisk/rps-app:<tag>
kubectl -n mlops-poc rollout status deployment/rps-app

# Overlay cleanup (legacy)
./scripts/clear_configmap_overlays.sh --dry-run

# Validation suites
python tests/test_essential.py
python tests/test_phased_validation_v2.py --all --verbose
python scripts/validate.py --games 500 --batch 50

# Emergency levers
kubectl -n mlops-poc delete jobs --all
curl -X POST https://mlops-rps.uk/models/reload
ssh root@65.21.151.52 'free -h'
```

Always allow games to reach 10 points so scores, metrics, and promotion logic remain consistent. Refer to `docs/metrics.md` for alert thresholds and dashboard maintenance.





