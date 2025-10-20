# Overall Goal

Ensure training pipelines rely far more on human gameplay data by downsampling legacy bot games while aligning cron-driven gameplay generation and follow-up validation so production accuracy and retraining cadence remain stable.

# To-Do

- [x] Implement legacy game downsampling in training data loaders. Purpose: only keep every Nth legacy bot game when assembling datasets so human matches dominate retraining. (may want to check downstream consumers of `BaseRPSModel.events_df` and `train_all_aliases` auto-fill logic for unexpected assumptions about full legacy coverage.)  Default to N=5 (see next point).
- [x] Add configuration surface for legacy stride. Purpose: expose a shared `TRAINING_LEGACY_GAME_STRIDE` option (default 5) via `app.config` so trainers, scripts, and tests can override the sampling factor consistently.
- [ ] Revisit dataset gating thresholds. Purpose: adjust `MIN_NEW_ROWS_FOR_TRAINING`/`MIN_TOTAL_ROWS` (or their computation) so the Cron job doesn’t keep triggering based on pre-filter counts when usable rows shrink. (may want to check how `ENABLE_AUTOFILL` and `check_training_needed` interplay once legacy rows are removed.)
- [x] Update training Cron documentation and dashboards. Purpose: communicate that effective training volume now excludes most bot-vs-bot data and refresh Grafana panels/alerts relying on legacy volumes.
- [x] Modify legacy gameplay Cron schedule. Purpose: change `infra/k8s/22-legacy-gameplay-cronjob.yaml` (and matching comments) so legacy bot simulations run 5× less often.  (This choice is coincidental to be matching N=5 above, just do this change once statically)
- [x] Rebuild and redeploy manifests. Purpose: apply the updated Cron job schedule to the cluster and confirm the next run time reflects the slower cadence.
- [x] Run targeted trainer regression tests. Purpose: execute `python tests/test_essential.py` and a forced single-trainer run to confirm dataset construction, MLflow logging, and alias promotion still succeed with the trimmed legacy data.  If possible, do some check or debug print about what data it ingests followed by what data it produces (to see if it truly is doing the skipping).  *(Single-trainer dry run still pending; essential suite passed.)*
- [ ] Monitor production metrics post-change. Purpose: watch `rps_model_correct_predictions_by_alias_total`, promotion ledger stats, and training job logs for shifts caused by the reduced legacy sampling.
- [ ] Update operational runbooks. Purpose: capture the new training mix, stride parameter, and Cron cadence in `docs/operations.md` (or related guides) so future responders know how to tune or disable the sampling.  Be concise and precise, do not add bloat.  (Scan these .md's and replace and old/redundant content.)
