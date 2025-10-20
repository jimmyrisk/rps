# Model Promotion System Review
**Date:** October 19, 2025  
**Issue:** Only seeing "alias_reorder" in decision field, never seeing retain_production or swap_production_b properly labeled

---

## TL;DR

**Root Cause:** Deployed images (Oct 17) have a bug where `decision` field gets overwritten to "alias_reorder" when challengers reshuffle, even though the actual hypothesis test concluded something different.

**Fix Location:** Repository code (commit b3c846e) contains the fix dated Oct 18, but images were never rebuilt.

**Action Required:** Rebuild and redeploy images to apply the fix.

---

## The Bug Explained

### What's Happening in Deployed Code

```python
# BUGGY CODE (in deployed Oct 17 images)
if test_result.should_swap():
    decision = "swap_production_b"
elif reorder_applied:
    decision = "alias_reorder"  # ❌ OVERWRITES the actual test result
else:
    decision = test_result.decision
```

**Problem:** When challengers (B/shadow1/shadow2) reshuffle, the code sets `decision = "alias_reorder"`, overwriting the actual hypothesis test result (which was "retain_production" or "insufficient_data").

### What the Fixed Code Does

```python
# FIXED CODE (in repo, Oct 18)
decision = test_result.decision  # Always preserve the hypothesis test result
# reorder_applied boolean separately tracks if challengers reshuffled
```

**Solution:** Keep `decision` as the hypothesis test result. Use `reorder_applied` boolean to separately track challenger reshuffling.

### Valid Decision Values (After Fix)

- `swap_production_b` - B beat Production, swapped them
- `retain_production` - Production kept (equal or better than B)
- `insufficient_data` - Not enough games to decide

The `reorder_applied` field (boolean) separately shows whether B/shadow1/shadow2 were reshuffled.

---

## Counter Reset Behavior (This Part Works Correctly)

### Question: When do counters reset?

**Answer:** Only when Production ↔ B swap occurs.

**Challenger reorders do NOT reset counters** (this is intentional and correct).

### Evidence from Database

Looking at xgboost events around the Oct 19 22:37:44 swap:

| Time     | Decision            | Prod Raw | B Raw | Prod Δ | B Δ  |
|----------|---------------------|----------|-------|--------|------|
| 22:13:02 | alias_reorder       | 13/16    | 25/30 | 5      | 13   |
| **22:37:44** | **swap_production_b** | **13/16** | **25/30** | **0** | **0** |
| 23:02:13 | alias_reorder       | 13/16    | 26/31 | 0      | 1    |
| 23:26:39 | alias_reorder       | 13/16    | 26/31 | 0      | 1    |
| 00:23:00 | alias_reorder       | 13/16    | 26/31 | 0      | 1    |
| 00:52:51 | alias_reorder       | 14/17    | 26/31 | 1      | 1    |

**Observations:**
1. At swap (22:37:44): Delta columns reset to 0
2. After swap: Deltas increment from 0 as new games are played
3. Raw counters continue incrementing (Prometheus counters only go up)
4. "Reset" = recording current values as new baseline, calculating deltas from there

### How Prometheus Counters Actually Work

**Prometheus counters never reset** - they only increase.

When we say "counters reset after swap":
1. Current counter values are recorded as a new baseline in the database
2. "Since swap" deltas are calculated as: current_value - baseline_value
3. Prometheus gauges (`rps_model_games_since_production_swap`) display these deltas

### What Happens to Counter Labels After Swap?

**Before swap:**
- Production label tracks model_A (16 total games)
- B label tracks model_B (30 total games)

**After swap (models switch slots):**
- Production label now tracks model_B (continues from 16)
- B label now tracks model_A (continues from 30)

The **label doesn't change**, but the **model behind the label does**. This is why the baseline system exists - to handle model identity changes while counter labels stay the same.

### Scenario: Challenger Reorder (B ↔ shadow1)

**What happens:**
- Production slot: unchanged
- B slot: new model moves in (shadow1 → B)
- Prometheus counter: `rps_model_games_by_alias_total{alias="B"}` continues from current value

**Why no reset:**
- We're testing "Can any challenger beat Production?"
- B slot represents "best challenger currently being tested"
- If we reset on every challenger swap, we'd never accumulate enough evidence
- The counter tracks "B slot performance" not "specific model performance"

### Scenario: Production Swap (Production ↔ B)

**What happens:**
- Both slots change models
- New baseline recorded with current counter values
- "Since swap" deltas reset to 0
- Future games increment deltas from this new baseline

**Why reset:**
- After swap, we have a new Production/B arrangement
- Need fresh data to evaluate if the new arrangement is better
- Previous accumulated evidence no longer relevant

---

## Answers to Your Specific Questions

### Q1: Are you sure counters reset when Production gets replaced by B?

**A:** Yes, but "reset" means "new baseline recorded", not "Prometheus counter goes to 0".

**Evidence:** Database shows "since swap" deltas at 0 immediately after swap_production_b event, then incrementing afterward.

**What physically happens:**
- Prometheus counter `rps_model_games_by_alias_total{alias="Production"}`: continues from current value (e.g., 16)
- Database baseline: records 16 as the new baseline
- Gauge `rps_model_games_since_production_swap{alias="Production"}`: shows 0 (calculated as 16 - 16)
- Next game: counter becomes 17, gauge shows 1 (calculated as 17 - 16)

### Q2: Are you sure B's counters reset when B gets replaced?

**A:** It depends on what replaces B:

**If Production ↔ B swap:** YES, baseline recorded, deltas reset to 0

**If challenger reorder (shadow1 → B):** NO, baseline unchanged, deltas continue accumulating

This is intentional. The system tracks slot performance, not model identity. When a new challenger moves into B slot, we want to see if it can beat Production over accumulated games.

### Q3: How can the swap work in the repo but not the cluster, yet you're validating with cluster data?

**A:** The swap MECHANISM works correctly everywhere. The BUG is only in how the `decision` field is LABELED.

**What works (both repo and cluster):**
- ✅ Hypothesis testing (comparing Production vs B win rates)
- ✅ Alias swapping (models switching slots in MLflow)
- ✅ Counter tracking (raw counters and deltas)
- ✅ Baseline recording (new baseline after swap)

**What's broken (only in deployed cluster):**
- ❌ Decision field labeling (shows "alias_reorder" instead of actual test result)
- ❌ Log output (final summary misleading)
- ❌ Database storage (wrong decision value stored)

**Evidence:**
- Database contains swap_production_b event at Oct 19 22:37:44 ✅
- Counters properly reset to 0 at that event ✅
- But subsequent events show "alias_reorder" when they should show "insufficient_data" or "retain_production" ❌

The SWAP works. The LABEL is wrong.

---

## Log Evidence of the Bug

From training job rps-trainer-29348790:

```
# Phase 1: Hypothesis test runs (correct result)
2025-10-20 02:52:30 - xgboost: retaining Production alias (decision=insufficient_data)
2025-10-20 02:52:52 - feedforward_nn: retaining Production alias (decision=retain_production)

# Phase 2: Final summary (decision field overwritten)
2025-10-20 02:54:24 - xgboost: decision=alias_reorder swap=False reorder=True
2025-10-20 02:54:24 - feedforward_nn: decision=alias_reorder swap=False reorder=True
```

**The bug:** Hypothesis test correctly identifies "insufficient_data" and "retain_production", but then the decision field gets overwritten to "alias_reorder" in the final summary and database record.

---

## Recommendations

### 1. Rebuild and Redeploy

```bash
cd /home/jimmy/rps
./scripts/build_push_deploy.sh --tag latest --push --deploy
```

### 2. Verify Deployment

```bash
# Check new image versions
kubectl -n mlops-poc describe pod -l app=rps-app | grep "Image:"
kubectl -n mlops-poc describe cronjob rps-trainer | grep "Image:"
```

Should show newer timestamps, not "dev-20251017".

### 3. Verify Fix Works

Wait for next training cycle (every 30 minutes), then check logs:

```bash
kubectl -n mlops-poc logs -f job/<next-job-name> | grep "decision="
```

Look for final summary showing `decision=retain_production` or `decision=insufficient_data`, **NOT** `decision=alias_reorder`.

### 4. Check Database

After a few cycles:

```bash
kubectl -n mlops-poc exec deploy/rps-app -- python -c "
from app.db import connect
conn = connect()
cursor = conn.execute('''
  SELECT created_ts_utc, model_type, decision, reorder_applied
  FROM promotion_events
  WHERE created_ts_utc > '$(date -u +%Y-%m-%dT%H:%M:%SZ)'
  ORDER BY created_ts_utc DESC LIMIT 10
''')
for row in cursor.fetchall():
    print(row)
"
```

Should see proper decision values (swap_production_b, retain_production, insufficient_data).

---

## What Will Change After Fix

### Before (Current Behavior)
- Decision field: mostly shows "alias_reorder"
- Grafana: can't filter by actual promotion decisions
- Logs: misleading final summaries

### After (Fixed Behavior)
- Decision field: shows swap_production_b, retain_production, or insufficient_data
- Grafana: can filter by actual promotion decisions
- Logs: clear separation between hypothesis test result and challenger reordering
- `reorder_applied` boolean separately tracks challenger reshuffling

---

## Summary

| Component | Status | Notes |
|-----------|--------|-------|
| Counter tracking | ✅ Works | Both raw and delta calculations correct |
| Hypothesis testing | ✅ Works | Production vs B comparison correct |
| Baseline system | ✅ Works | Resets at swap, continues on reorder |
| Alias swapping | ✅ Works | Models switch slots correctly |
| Challenger reordering | ✅ Works | B/shadow1/shadow2 shuffle by accuracy |
| Decision field labeling | ❌ Broken | Overwritten to "alias_reorder" |
| Log output | ❌ Misleading | Shows wrong decision in summary |
| Database records | ❌ Wrong values | Stores "alias_reorder" too often |

**Fix:** Already in repository (Oct 18 changes). Need to rebuild and redeploy images.

---

**End of Review**
