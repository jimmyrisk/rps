# Metrics & Observability Guide

The gameplay service exposes Prometheus metrics from `/metrics`, which Grafana Alloy scrapes and forwards to Grafana Cloud. Use this guide to find the key series, representative PromQL queries, and dashboard workflows. For runbook procedures see `docs/operations.md`; for architecture context see `docs/architecture.md`.

---

## Access Points & Defaults

- Prometheus endpoint: `https://mlops-rps.uk/metrics`
- Grafana dashboard: https://jimmyrisk41.grafana.net/d/18dc23df-eea3-406c-a7e7-99066a987538/rps-mlops-poc-since-pod-restart
- Grafana Alloy logs: `kubectl -n mlops-poc logs -l app=alloy --tail=200`
- Metrics window: `_DEFAULT_METRICS_SINCE = 2025-10-10T08:00:00Z` (override via `METRICS_SINCE_DATE` on the `rps-app` deployment)

Counters reset when the `rps-app` pod restarts; Grafana Cloud preserves historical series.

---

## Metric Families

### Model Alias Metrics

| Metric | Description |
| --- | --- |
| `rps_model_predictions_by_alias_total` | Total predictions made by a model family × alias. |
| `rps_model_correct_predictions_by_alias_total` | Correct predictions per alias. |
| `rps_model_games_by_alias_total` | Completed games for Production/B aliases. |
| `rps_model_game_wins_by_alias_total` | Game wins for Production/B aliases. |

12-hour Production/B action accuracy:

```promql
sum by (model, alias) (
  increase(rps_model_correct_predictions_by_alias_total{alias=~"Production|B"}[12h])
) /
clamp_min(
  sum by (model, alias) (
    increase(rps_model_predictions_by_alias_total{alias=~"Production|B"}[12h])
  ),
  1
)
```

Deterministic opponents (for example `cal`) should mirror training accuracy; sustained divergence is a defect (feature parity, alias routing, or artifact drift).

### Promotion Metrics

| Metric | Description |
| --- | --- |
| `rps_model_promotion_z_statistic` | One-sided z-statistic comparing Production vs B. |
| `rps_model_promotion_p_value` | Associated p-value; alert when challengers convincingly outperform Production. |
| `rps_model_promotion_events_total{decision}` | Counts decision outcomes (`swap_production_b`, `retain_production`, `insufficient_data`, `alias_reorder`). |
| `rps_model_alias_rank{alias}` | Challenger ordering (1 = next in line for Production). |
| `rps_model_alias_reorders_total` | Number of challenger reseeds executed. |
| `rps_model_games_since_production_swap{alias}` | Ticker showing Production/B games since the most recent Production/B swap. |
| `rps_model_promotion_cycles_since_swap` | Auto-promotion cycles evaluated since the last swap (continues through challenger-only reorders). |

Auto-promotion posts JSON payloads to `/internal/promotion/report`; the handler persists them to SQLite and exposes history through `/internal/promotion/history`. Metrics reset on pod restarts—seed new games or run `python scripts/capture_cal_forrest_dataset.py --games 10 --strict-parity --sleep 0.1` before rerunning the automation.

Ticker behaviour:
- Production/B tickers (`games_since_production_swap` and `promotion_cycles_since_swap`) advance on every promotion cycle, whether or not a swap occurs.
- The counters reset to zero only when a Production/B swap is executed; challenger reseeds leave them untouched.
- Prod vs B comparisons require ≥3 games per alias **since the last swap**; if either alias has fewer games in the ticker window the automation exits with `decision="insufficient_data"`.

Alert snippet:

```promql
rps_model_promotion_p_value < 0.05
```

### Policy & Game Metrics

| Metric | Labels | Description |
| --- | --- | --- |
| `rps_action_predictions_total` | `policy`, `difficulty`, `model_type` | Per-round predictions routed by a policy. |
| `rps_action_wins_total`, `rps_action_losses_total`, `rps_action_ties_total` | same | Round outcomes from the bot perspective. |
| `rps_games_total` | `policy`, `difficulty` | Games completed. |
| `rps_game_wins_total`, `rps_game_losses_total` | `policy`, `difficulty` | Game-level outcomes. |

Competitive (standard difficulty) win rate:

```promql
sum by (policy, difficulty) (
  increase(rps_game_wins_total{policy="$policy", difficulty="standard"}[12h])
) /
clamp_min(
  sum by (policy, difficulty) (
    increase(rps_games_total{policy="$policy", difficulty="standard"}[12h])
  ),
  1
)
```

### Training & Infrastructure Metrics

| Metric | Description |
| --- | --- |
| `rps_training_completed_total{model_type,status}` | Successful vs failed trainer runs. |
| `rps_training_duration_seconds_bucket` | Histogram for training runtimes. |
| `rps_model_fallback_total{model_type,alias,reason}` | Counts inference fallbacks (MinIO miss, MLflow miss, etc.). |
| `rps_active_games` | Gauge of concurrent games. |

---

## Dashboards & Automation

- **Panel map** (Grafana UID `18dc23df-eea3-406c-a7e7-99066a987538`)
  - Panels 201–209: alias action accuracy and Production/B game win rates for `brian`, `logan`, `forrest`.
  - Panels 210–215: throughput (predictions, games) and aggregate counters.
  - Panels 216–217: training completions and model fallbacks.
  - Panels 218–222: promotion decisions, alias ranks, and the JSON-powered ledger snapshot.
- **JSON data source** – Configure UID `rps-promotion-ledger` with base URL `https://mlops-rps.uk` so panel requests like `/internal/promotion/history?limit=20` avoid double-scheme errors.

### Update workflow

1. Clone the production dashboard in Grafana (*Save As → staging*).
2. Edit panels and confirm label filters via *Inspect → Query*.
3. Replace `ops/grafana-dashboard.pretty.json` with the exported JSON.
4. Normalise formatting:
   ```bash
   jq '.' ops/grafana-dashboard.pretty.json > ops/grafana-dashboard.pretty.json.tmp && mv ops/grafana-dashboard.pretty.json.tmp ops/grafana-dashboard.pretty.json
   jq -c '.' ops/grafana-dashboard.pretty.json > ops/grafana-dashboard.json
   ```
5. Deploy: `./ops/deploy_clean_dashboard.sh` (reads `GRAFANA_API_KEY` or configured key file).

### Manual hotfix

1. Copy `ops/grafana-dashboard.json`.
2. Grafana → *Dashboard → New → Import* → paste JSON → select `grafanacloud-ml-metrics`.
3. Remove superseded dashboards to avoid confusion.

---

## Alerts & Troubleshooting

- **Production accuracy:** Watch the 12-hour accuracy query; investigate when production deviates from training by >5 percentage points.
- **Fallback spikes:** Alert on `increase(rps_model_fallback_total[5m]) > 0` and check MinIO/MLflow reachability.
- **Promotion readiness:** Alert when `increase(rps_model_promotion_events_total{decision="insufficient_data"}[1h]) > 0` after trainer runs (insufficient live games).
- **Service health:** Track `rps_active_games` for sustained plateaus and pair with `/healthz` checks from `docs/operations.md`.

| Symptom | Checks | Action |
| --- | --- | --- |
| Panels flatline | `kubectl -n mlops-poc logs -l app=alloy --tail=100` | Restart Alloy or verify scrape annotations on `rps-app`. |
| Alias metrics missing | `curl -s https://mlops-rps.uk/metrics | grep rps_model_predictions_by_alias_total` | Ensure `/models/reload` succeeded and MinIO contains `.alias_*` markers. |
| Promotion panels empty | `curl -s https://mlops-rps.uk/internal/promotion/history?limit=5` | Confirm auto-promotion ran and the ledger is populated. |
| Capture parity issues | `python scripts/capture_cal_forrest_dataset.py --games 10 --strict-parity --sleep 0.1` | Compare output to MLflow evaluation metrics; verify feature contract deployment. |

---

## Quick Commands

```bash
# Grep promotion metrics
curl -s https://mlops-rps.uk/metrics | grep rps_model_promotion

# Pull latest promotion ledger rows
curl -s https://mlops-rps.uk/internal/promotion/history?limit=5 | jq '.'

# Render Grafana panel (requires API key)
curl -s -H "Authorization: Bearer $(<ops/.grafana_api_key)" \
  "https://jimmyrisk41.grafana.net/render/d-solo/18dc23df-eea3-406c-a7e7-99066a987538/rps-mlops-poc-since-pod-restart?orgId=1&panelId=221&width=1000&height=500&tz=UTC" \
  -o logs/panel221.png
```

See `docs/operations.md` for deployment levers and `README.md` for installation prerequisites.
  `ops/grafana-dashboard.json`, overwrites the existing board, and prints the new UID.

**Manual hotfix:**
1. Copy the contents of `ops/grafana-dashboard.json`.
2. Grafana → *Dashboard → New → Import* → paste JSON → choose `grafanacloud-ml-metrics` data source.
3. Clean up superseded dashboards afterwards to avoid confusion.

### Alerting and tuning tips

- **Production accuracy:** Plot `rps_model_correct_predictions_by_alias_total / rps_model_predictions_by_alias_total` for Production vs B to validate A/B experiments.
- **Game strength:** Build tables using `rps_game_wins_total` grouped by `policy` × `difficulty` to compare bots.
- **Training SLA:** Graph `rate(rps_training_duration_seconds_bucket[5m])` to catch unusually slow training jobs.
- **Fallback tracker:** Alert whenever `increase(rps_model_fallback_total[5m]) > 0`.
- **Promotion alerts:** Leverage `rps_model_promotion_p_value < 0.05` for paging; `ops/snippets/auto-promotion-alert.json` provides a ready-to-import template.

### Troubleshooting dashboards

| Symptom | Checks |
| --- | --- |
| Panels show `N/A` or zeros | `kubectl -n mlops-poc logs -l app=alloy --tail=100` to confirm Alloy is scraping; validate PromQL via Grafana Explore. |
| Accuracy looks off | Verify alias filters, compare against MLflow evaluation metrics, and run the strict Cal vs Forrest harness. |
| Scripted deploy fails with 401 | Generate a fresh API key in Grafana Cloud and retry. |
| Dashboard link missing in apps | Ensure `GRAFANA_DASHBOARD_URL` env var is set in both deployments. |

---

## Troubleshooting Checklist

1. **Metrics endpoint blank?** Ensure `prometheus_fastapi_instrumentator` is started in `app/main.py` and pod logs show no exceptions.
2. **Grafana missing data?** Confirm Grafana Alloy pod is running and scraping the `rps-app` service (`prometheus.io/scrape: "true"`).
3. **Unexpected drops after pod restart?** Remember local counters reset—use Grafana Cloud’s long-term storage for history.
4. **Alias mismatch in dashboards?** Check alias markers in MinIO (`mc cat local/mlflow-artifacts/<run>/.alias_Production`).
5. **Strict parity failures?** Ensure `app/features.py` helpers `_build_stateless_history` / `_backfill_score_history` are deployed everywhere, rerun the capture script, and compare action accuracy to the latest MLflow metrics.

For end-to-end operations and promotion workflows refer back to `docs/operations.md`.

### Promotion panel redesign (2025-10-11)

The production dashboard now blends live Prometheus metrics with the persisted promotion ledger:

1. **Promotion Ledger (Panel 218)** – Pulls the last 20 rows from `/internal/promotion/history` via the Grafana JSON API data source (UID `rps-promotion-ledger`). Columns include the normalised win rates, z-score, p-value, and free-form reason captured in the ledger. Configure the JSON data source with base URL `https://mlops-rps.uk` and let the panel’s relative path (`/internal/promotion/history?limit=20`) drive the request—this prevents the plugin from generating `https://https://…` URLs.
2. **Promotion Trendlines (Panel 219)** – 24-hour dual-axis view of `rps_model_promotion_z_statistic` and `rps_model_promotion_p_value` with threshold lines at |z| = 1.28 and p = 0.20 so challengers that clear the α=0.20 barrier are immediately visible.
3. **Promotion Event Totals (Panel 220)** – Seven-day bar gauge grouped by decision, with colour overrides (`swap_production_b` green, `retain_production` blue, `insufficient_data` gray, `alias_reorder` orange) to quickly surface skewed automation behaviour.
4. **Alias Rank Timeline (Panel 221)** – Step-after timeline for challenger ranks with a threshold line at rank 1 to highlight which alias is next in line for Production.

After tweaking the layout, export the JSON and refresh both `ops/grafana-dashboard.pretty.json` and `ops/grafana-dashboard.json` (`jq -c '.' ops/grafana-dashboard.pretty.json > ops/grafana-dashboard.json`) so scripted deploys stay in sync.

### Cross-family accuracy tracking (feasibility notes)

- **Current state:** Only the policy’s native model family logs alias-level accuracy during live games. For example, Brian (feedforward) records `rps_model_*` counters for `feedforward_nn@{Production,B,...}`, while Forrest and Logan do the same for xgboost and multinomial logistic respectively.
- **Expanding to all 12 models:** We could loop through every `(model_type, alias)` combination inside `app/routes/game.py` after the gambit rounds and call `ModelManager.predict_with_alias()` for each, recording correctness against the user’s move. This would triple the number of model evaluations per turn (12 predictions instead of 4) and slightly increase latency while priming on-disk caches for any model family that has not been used recently.
- **Constraints to consider:**
  - **CPU & latency:** Each extra prediction performs an MLflow pyfunc call. Cached models respond quickly, but cold-loading all three model families for every match may spike CPU during bursts.
  - **Memory pressure:** Loading all 12 pyfunc wrappers into memory simultaneously increases the resident set size of the serving pod. Verify headroom on the 4 GB node before flipping the switch.
  - **Model parity:** Ensure the shared 50-feature frame matches expectations for all models; the current feature builder is shared, so this is low risk.
- **Suggested rollout:** Prototype a feature flag that gradually widens the logging loop (e.g., enable cross-family predictions for deterministic QA sessions or scheduled load tests). Measure per-request latency and RSS in Grafana before enabling it for all players.
