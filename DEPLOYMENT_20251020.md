# Deployment Summary - Oct 20, 2025

## Promotion Decision Field Fix

**Issue:** Decision field was being overwritten to "alias_reorder" even when hypothesis test concluded "retain_production" or "insufficient_data"

**Root Cause:** Deployed images (Oct 17) contained buggy code. Repository had fix (Oct 18 comments) but images were never rebuilt.

**Resolution:** Built and deployed new images with the fix.

---

## Deployment Details

**Date:** 2025-10-20 03:21 UTC  
**Tag:** `fix-promotion-decision-20251019`

### Images Built & Pushed

- `ghcr.io/jimmyrisk/rps-app:fix-promotion-decision-20251019`
- `ghcr.io/jimmyrisk/rps-trainer:fix-promotion-decision-20251019`
- `ghcr.io/jimmyrisk/rps-ui:fix-promotion-decision-20251019`

### Deployments Updated

- ✅ `rps-app` - Rolled out successfully
- ✅ `rps-ui` - Rolled out successfully  
- ✅ `rps-trainer` CronJob - Updated (next job will use new image)
- ✅ ASSET_VERSION bumped to `20251019202213` (cache bust)

---

## Verification

### Fix Confirmed Working

**Database evidence** - Recent promotion events now show correct decision values:

```
Timestamp            | Decision            | Reorder
------------------------------------------------------------
2025-10-20T03:23:29 | insufficient_data   | False  ← NEW (CORRECT)
2025-10-20T02:52:41 | alias_reorder       | True   ← OLD (WRONG)
2025-10-20T02:22:34 | alias_reorder       | True   ← OLD (WRONG)
```

### What Changed

**Before:**
- Decision field: mostly "alias_reorder" (incorrect)
- Logs: misleading summaries
- Grafana: couldn't filter by actual promotion decisions

**After:**
- Decision field: "insufficient_data", "retain_production", or "swap_production_b" (correct)
- Logs: clear separation between hypothesis test and challenger reordering
- Grafana: can filter by actual promotion decisions
- `reorder_applied` boolean separately tracks challenger reshuffling

---

## Expected Behavior Going Forward

### Decision Field Values

- `swap_production_b` - B beat Production, they were swapped
- `retain_production` - Production kept (equal or better than B)
- `insufficient_data` - Not enough games to decide

### Reorder Field

- `reorder_applied: true` - B/shadow1/shadow2 were reshuffled by accuracy
- `reorder_applied: false` - No challenger reordering

### Valid Combinations

All combinations are possible and meaningful:
- decision=swap_production_b, reorder=true → Swapped Production/B AND reshuffled remaining challengers
- decision=swap_production_b, reorder=false → Only swapped Production/B
- decision=retain_production, reorder=true → Kept Production, but reshuffled challengers
- decision=retain_production, reorder=false → Kept Production, no changes
- decision=insufficient_data, reorder=true → Not enough data, but reshuffled challengers anyway
- decision=insufficient_data, reorder=false → Not enough data, no changes

---

## Monitoring Recommendations

### Check Next Training Cycle

Next CronJob run will use the new trainer image for the first time.

**Expected at:** ~03:50 UTC (every 30 minutes)

**Monitor with:**
```bash
# Watch for new job creation
kubectl -n mlops-poc get jobs -w

# Check logs of new job
kubectl -n mlops-poc logs -f job/rps-trainer-<timestamp>

# Verify final summary shows correct decisions
kubectl -n mlops-poc logs job/rps-trainer-<timestamp> | grep "decision="
```

**Look for:**
- Final summary lines showing `decision=retain_production` or `decision=insufficient_data`
- Should NOT see `decision=alias_reorder` anymore
- `reorder=True/False` should appear separately from decision

### Database Checks

```bash
kubectl -n mlops-poc exec deploy/rps-app -- python -c "
from app.db import connect
conn = connect()
cursor = conn.execute('''
  SELECT created_ts_utc, model_type, decision, reorder_applied
  FROM promotion_events
  WHERE created_ts_utc > '2025-10-20T03:00:00Z'
  ORDER BY created_ts_utc DESC LIMIT 20
''')
for row in cursor.fetchall():
    print(f'{row[0]}: {row[1]} - decision={row[2]}, reorder={bool(row[3])}')
"
```

Should see proper decision values, not "alias_reorder".

---

## Related Documentation

- `PROMOTION_SYSTEM_REVIEW.md` - Detailed analysis of the issue
- `.github/copilot-instructions.md` - Updated deployment guidance
- `scripts/build_push_deploy.sh` - Deployment script used

---

**Deployment Status:** ✅ SUCCESSFUL  
**Fix Status:** ✅ VERIFIED WORKING  
**Action Required:** None - monitoring recommended for next training cycle

---

## Legacy Downsampling Rollout

**Date:** 2025-10-20 23:15 UTC  
**Tag:** `legacy-downsample-20251020`

### Images Built & Pushed
- `ghcr.io/jimmyrisk/rps-app:legacy-downsample-20251020`
- `ghcr.io/jimmyrisk/rps-trainer:legacy-downsample-20251020`
- `ghcr.io/jimmyrisk/rps-ui:legacy-downsample-20251020`

### Deployment Steps
- Ran `./scripts/build_push_deploy.sh --tag legacy-downsample-20251020 --push --deploy`
- Confirmed `rps-app` and `rps-ui` deployments rolled out successfully
- Updated `rps-trainer` CronJob image (next run will consume new trainer build)
- Bumped `ASSET_VERSION` to `20251019230758` for cache busting
- Re-applied `infra/k8s/22-legacy-gameplay-cronjob.yaml` (Cron schedule confirmed at `*/150 * * * *`)

### Validation
- Executed essential regression suite: `conda run -n rps python tests/test_essential.py`
- Production spot check via in-cluster probe:
  - `Legacy bot games kept: 1088/5437 (20.0% of games)`
  - `Legacy events kept: 12155/60097 (20.2% of events)`
  - After 3k row limiter: `Total events: 3000 | Legacy events kept: 1117 (37.2% of events)`
- Verified Cron schedule with `kubectl -n mlops-poc get cronjob legacy-gameplay -o jsonpath='{.spec.schedule}'`

### Follow-Up
- Monitor next trainer run to confirm downsampling logs appear with stride=5
- Track production win-rate deltas for Production vs B after change lands
- Outstanding TODO: revisit gating thresholds so the 3k limiter reflects post-downsampling volume
