# Model Promotion System Review
**Date:** October 19, 2025  
**Reviewer:** AI Assistant  
**Context:** Investigation into why only "alias_reorder" decisions appear in logs

---

## Executive Summary

### Root Cause: Decision Field Bug in Deployed Code

The deployed cluster (images from Oct 17) has a bug where the `decision` field gets overwritten to "alias_reorder" when challengers reshuffle, even when the actual hypothesis test concluded something different (like "retain_production" or "insufficient_data").

**Key Finding:** The repository contains a fix (dated Oct 18 in comments), but the deployed images predate this fix.

### What Actually Works

✅ **Counter reset logic** - Counters properly reset when Production↔B swap occurs  
✅ **Hypothesis testing** - Statistical comparison between Production and B works correctly  
✅ **Challenger reordering** - B/shadow1/shadow2 reshuffle based on accuracy works correctly  
✅ **Prometheus metrics** - All metrics are being recorded properly  

### What's Broken

❌ **Decision field in database** - Shows "alias_reorder" instead of actual test result  
❌ **Log output** - Final summary shows "decision=alias_reorder" instead of retain_production/insufficient_data  
❌ **Grafana filtering** - Can't filter by actual promotion decisions because they're mislabeled  

---

---

## How Counter Resets Actually Work

### Evidence from Database (xgboost example)

Looking at the actual counter values around the Oct 19 22:37:44 swap:

| Time     | Decision            | Prod Raw | B Raw | Prod Since Swap | B Since Swap |
|----------|---------------------|----------|-------|-----------------|--------------|
| 22:13:02 | alias_reorder       | 13/16    | 25/30 | 5               | 13           |
| **22:37:44** | **swap_production_b** | **13/16** | **25/30** | **0** | **0** |
| 23:02:13 | alias_reorder       | 13/16    | 26/31 | 0               | 1            |
| 23:26:39 | alias_reorder       | 13/16    | 26/31 | 0               | 1            |
| 00:23:00 | alias_reorder       | 13/16    | 26/31 | 0               | 1            |
| 00:52:51 | alias_reorder       | 14/17    | 26/31 | 1               | 1            |

### What This Shows

1. **Before swap (22:13:02):**
   - Raw Prometheus counters: Production=13/16, B=25/30
   - Since-swap deltas: Production=5, B=13
   - These deltas are calculated from a previous swap baseline

2. **At swap (22:37:44):**
   - Raw counters unchanged: Production=13/16, B=25/30
   - Since-swap deltas **reset to 0**
   - This row becomes the new baseline

3. **After swap (23:02:13+):**
   - Raw counters continue incrementing: 13/16 → 14/17, 25/30 → 26/31
   - Since-swap deltas calculated as: current - baseline
   - Example: 14 total games - 13 baseline = 1 game since swap

### The Key Insight

**Prometheus counters NEVER reset** - they only increase. The "reset" is actually:
1. Recording current counter values as a new baseline in the database
2. Calculating deltas from that baseline for display
3. The gauges (`rps_model_games_since_production_swap`) show these deltas

### What Happens to Model Identity After Swap?

This is the tricky part. When Production and B swap:
- **The models switch slots** (version that was in B → now in Production)
- **The counter labels don't change** (still "Production" and "B")
- **The counters keep incrementing** (B counter now tracks the model that was previously in Production)

So if:
- Before swap: Production=model_v1 (13/16), B=model_v2 (25/30)
- After swap: Production=model_v2 (25/30), B=model_v1 (13/16)

But Prometheus sees:
- Production label counter stays at 13/16, continues from there
- B label counter stays at 25/30, continues from there

**This is why the baseline system exists** - to handle the fact that the model identity behind each label changes, but the Prometheus label itself doesn't.

---

## The Decision Field Bug

### What the Buggy Code Does (Deployed Oct 17)

```python
# Simplified version of the bug
if test_result.should_swap():
    decision = "swap_production_b"
elif reorder_applied:
    decision = "alias_reorder"  # ❌ OVERWRITES the actual test result
else:
    decision = test_result.decision
```

This means:
- If swap happens: decision = "swap_production_b" ✅
- If swap AND reorder happen: decision = "swap_production_b" ✅ (doesn't reach elif)
- If NO swap but reorder happens: decision = "alias_reorder" ❌ (should be retain_production or insufficient_data)
- If nothing happens: decision = test_result.decision ✅

### What Actually Happens in Practice

From the logs, we see things like:
```
feedforward_nn: retaining Production alias (decision=retain_production)
...
feedforward_nn: decision=alias_reorder swap=False reorder=True
```

The first line is correct (hypothesis test says "retain_production"), but then the second line (final summary) says "alias_reorder" because the buggy code overwrites it.

### What the Fixed Code Does (Repository Oct 18)

```python
# Simplified fixed version
decision = test_result.decision  # Always use the hypothesis test result
# reorder_applied boolean separately tracks if challengers reshuffled
```

Now:
- `decision` field shows: swap_production_b, retain_production, or insufficient_data
- `reorder_applied` boolean separately shows: True or False

You can have any combination:
- decision=retain_production, reorder_applied=True → "Production is better, but we reshuffled challengers"
- decision=retain_production, reorder_applied=False → "Production is better, no changes"
- decision=swap_production_b, reorder_applied=True → "B is better, swapped them, also reshuffled remaining challengers"

---

## Database Evidence Summary

### Total Events by Decision (All Models)

```bash
$ kubectl exec rps-app -- python -c "
from app.db import connect
from collections import Counter
conn = connect()
cursor = conn.execute('SELECT decision FROM promotion_events')
print(Counter(row[0] for row in cursor.fetchall()))
"
```

Expected to show many "alias_reorder" with only a few "swap_production_b" events scattered throughout.

### The One Confirmed Swap

**xgboost on Oct 19 22:37:44 UTC:**
- Decision: swap_production_b (correctly stored)
- B had 83.3% win rate (25/30 games)
- Production had 81.2% win rate (13/16 games)
- B was better → swapped
- Counters reset to 0 after swap

**All subsequent xgboost events:** Show "alias_reorder" as decision, which is wrong. They should show "insufficient_data" (not enough games) but the buggy code overwrites it.

---

## Deployed vs Repository

### Deployed Images (Outdated)
- App: `ghcr.io/jimmyrisk/rps-app:dev-20251017e`
- Trainer: `ghcr.io/jimmyrisk/rps-trainer:dev-20251017d`
- Date: October 17, 2025
- Contains: Buggy decision field logic

### Repository Code (Current)
- Last commit: `b3c846e`
- Date: October 17 (but contains Oct 18 changes in comments)
- Contains: Fixed decision field logic

The Oct 18 fix was committed but images were never rebuilt and redeployed.

---

## Questions Answered

### Q1: Should "alias_reorder" only appear when there was NO prod/B swap?

**A:** "alias_reorder" should NEVER appear as a decision value in the fixed code.

The `decision` field should only ever be:
- `swap_production_b` - B beat Production, swapped them
- `retain_production` - Production kept (better or equal to B)
- `insufficient_data` - Not enough games to compare

The `reorder_applied` boolean (separate field) shows whether B/shadow1/shadow2 reshuffled.

### Q2: Are counters reset when there is a B/shadow1/shadow2 reshuffle?

**A:** NO. Counters continue accumulating. ✅ This is correct behavior.

Only Production↔B swaps trigger "resets" (really just recording a new baseline).

### Q3: Are counters reset when Production and B swap?

**A:** YES (via baseline system). ✅ This is correct behavior.

Evidence from database shows "since swap" values go to 0 immediately after swap, then increment from there.

### Q4: When Production gets replaced by B, do Production's counters reset?

**A:** This requires clarification on what "reset" means:

**Prometheus counter labels:** The `rps_model_games_by_alias_total{alias="Production"}` counter **does NOT reset to 0**. It continues incrementing. Prometheus counters can only increase.

**Displayed "since swap" values:** The gauge `rps_model_games_since_production_swap{alias="Production"}` **shows 0** after a swap, then increments as new games are played.

**What happens mechanically:**
1. Before swap: Production=model_A, counter shows 16 total games
2. Swap occurs: model_B → Production slot
3. After swap: Production=model_B, counter shows 16 total games (inherited from old value)
4. New games: Production counter goes 16→17→18... (model_B's performance in Production slot)
5. "Since swap" calculation: current (18) - baseline (16) = 2 games

So the raw counter doesn't reset, but we treat the swap point as a new baseline for comparisons.

### Q5: When B gets replaced (by shadow1 for example), do B's counters reset?

**A:** NO, and this is intentional. ✅

The B slot counter continues accumulating when a new model moves into it. This is correct because:
- We're testing "Is anything in the challenger pool better than Production?"
- B slot represents "best challenger currently being tested"
- When shadow1 → B, we want to see if the new B (which was shadow1) can beat Production
- Resetting would mean we'd never accumulate enough evidence
- The counter tracks "B slot performance" not "specific model performance"

---

## Recommendations

### Immediate Action Required

**Rebuild and redeploy images:**
```bash
cd /home/jimmy/rps
./scripts/build_push_deploy.sh --tag latest --push --deploy
```

This will incorporate the Oct 18 fix into the deployed cluster.

### Verification Steps

After deployment:

1. **Check image versions:**
```bash
kubectl -n mlops-poc describe pod -l app=rps-app | grep "Image:"
kubectl -n mlops-poc describe cronjob rps-trainer | grep "Image:"
```

2. **Wait for next training cycle** (every 30 minutes) and check logs:
```bash
kubectl -n mlops-poc logs -f job/<next-job-name> | grep "decision="
```

Look for final summary showing `decision=retain_production` or `decision=insufficient_data`, NOT `decision=alias_reorder`.

3. **Check database after a few cycles:**
```bash
kubectl -n mlops-poc exec deploy/rps-app -- python -c "
from app.db import connect
conn = connect()
cursor = conn.execute('''
  SELECT created_ts_utc, model_type, decision, reorder_applied
  FROM promotion_events
  WHERE created_ts_utc > '2025-10-20T04:00:00Z'
  ORDER BY created_ts_utc DESC LIMIT 10
''')
for row in cursor.fetchall():
    print(row)
"
```

Should see proper decision values, not "alias_reorder".

---

## Summary

### What's Working
- ✅ Counter tracking (both raw and since-swap deltas)
- ✅ Hypothesis testing (Production vs B comparison)
- ✅ Challenger reordering (B/shadow1/shadow2 by accuracy)
- ✅ Alias assignments in MLflow
- ✅ Prometheus metrics recording

### What's Broken (in deployed code)
- ❌ Decision field gets overwritten to "alias_reorder"
- ❌ Logs show wrong decision in final summary
- ❌ Database stores wrong decision value
- ❌ Grafana can't properly filter promotion events

### The Fix
Repository contains the fix. Just need to rebuild/redeploy images to apply it.

### Expected Behavior After Fix
- Database will show retain_production, swap_production_b, or insufficient_data
- Logs will clearly show when Production swaps vs when only challengers reshuffle
- Grafana panels will correctly categorize promotion events
- The `reorder_applied` boolean will separately show challenger reshuffling

---

## Technical Details: Counter Accumulation Logic

### Scenario: Challenger Reorder (B ↔ shadow1)

**Before:**
- Production: model_A (20 total games, 15 wins)
- B: model_B (30 total games, 18 wins)  
- shadow1: model_C (tracked by accuracy: 65%)

**Shadow1 has better accuracy → swap B ↔ shadow1:**
- Production: model_A (unchanged)
- B: model_C (was shadow1)
- shadow1: model_B (was B)

**Prometheus counters:**
- `rps_model_games_by_alias_total{alias="Production"}` = 20 (continues from 20)
- `rps_model_games_by_alias_total{alias="B"}` = 30 (continues from 30)

**New games played:**
- Production plays → counter goes 20→21→22...
- B plays (now model_C) → counter goes 30→31→32...

**"Since swap" values:**
- Last swap baseline: Production=10, B=15
- Current deltas: Production=20-10=10 games, B=30-15=15 games
- After reorder: deltas continue, Production=21-10=11, B=31-15=16

**Key point:** The baseline doesn't change on challenger reorders, so deltas keep growing.

### Scenario: Production Swap (Production ↔ B)

**Before:**
- Production: model_A (20 total games, 15 wins) 
- B: model_C (35 total games, 28 wins)

**B has better win rate → swap:**
- Production: model_C (was B)
- B: model_A (was Production)

**Prometheus counters:**
- `rps_model_games_by_alias_total{alias="Production"}` = 20 (continues from 20)
- `rps_model_games_by_alias_total{alias="B"}` = 35 (continues from 35)

**New baseline recorded:**
- Production baseline = 20 (current counter value)
- B baseline = 35 (current counter value)

**New games played:**
- Production plays (now model_C) → counter goes 20→21→22...
- B plays (now model_A) → counter goes 35→36→37...

**"Since swap" values:**
- New baseline: Production=20, B=35  
- Current deltas: Production=21-20=1, B=36-35=1
- Gauges show: 1 game each since swap

**Key point:** The baseline updates to current values, making deltas start from 0.

---

## Appendix: Log Evidence

### Training Job Output (rps-trainer-29348790)

```
# Phase 1: Training completes, hypothesis test runs
2025-10-20 02:52:30 - xgboost: retaining Production alias (decision=insufficient_data)
2025-10-20 02:52:52 - feedforward_nn: retaining Production alias (decision=retain_production)
2025-10-20 02:53:13 - multinomial_logistic: retaining Production alias (decision=retain_production)

# Phase 2: Auto-promotion runs, final summary
2025-10-20 02:54:24 - xgboost: decision=alias_reorder swap=False reorder=True
2025-10-20 02:54:24 - feedforward_nn: decision=alias_reorder swap=False reorder=True
2025-10-20 02:54:24 - multinomial_logistic: decision=alias_reorder swap=False reorder=True
```

**The bug:** Phase 1 correctly identifies "insufficient_data" and "retain_production", but Phase 2 overwrites this to "alias_reorder" in the final summary and database storage.

---

**End of Review**


#### The Problem (in deployed Oct 17 images):
The old code had this pattern around line 543 of `scripts/auto_promote_models.py`:

```python
# OLD BUGGY CODE (deployed):
if test_result.should_swap():
    decision = "swap_production_b"
elif reorder_applied:
    decision = "alias_reorder"  # ❌ OVERWRITES the test_result.decision
else:
    decision = test_result.decision
```

This means:
- If Production/B swap AND alias reorder both happen → decision becomes "alias_reorder" ❌
- If only alias reorder happens → decision becomes "alias_reorder" ✅
- If neither happens → decision is retain_production/insufficient_data ✅

#### The Fix (in current repo, Oct 18):
```python
# NEW CORRECT CODE (in repo):
# CHANGED 2025-10-18: Don't change decision field for alias reordering
# This way Grafana filters for swap_production_b/retain_production/insufficient_data
# will work cleanly without seeing "alias_reorder" noise
decision = test_result.decision
# Note: reorder_applied boolean field still shows if challengers were reshuffled
```

This preserves the actual hypothesis test decision:
- `swap_production_b` - B is better, swap them
- `retain_production` - Production is better or tied, keep it
- `insufficient_data` - Not enough games to decide

The `reorder_applied` boolean field separately tracks whether challenger aliases (B/shadow1/shadow2) were reshuffled.

### 2. Counter Reset Logic

#### Current Behavior (lines 505-517 of auto_promote_models.py):

```python
if test_result.should_swap():
    cycles_since_swap = 0
    production_games_since_swap = 0
    b_games_since_swap = 0
    ticker_note = "Ticker reset after Production/B swap"
else:
    cycles_since_swap = ticker_state.cycles_since_swap + 1
    production_games_since_swap = production_stats_window.total
    b_games_since_swap = b_stats_window.total
    ticker_note = (...)
```

**This is CORRECT.** Counters reset ONLY when `test_result.should_swap()` returns True, which means:
- A Production ↔ B swap actually occurred
- The decision was "swap_production_b"

**Alias reordering alone does NOT reset counters** - this is intentional and correct!

#### Why This Design Makes Sense:

The counters track:
- How many games Production has played since last swap
- How many games B has played since last swap  
- How many promotion cycles have run since last swap

When B, shadow1, shadow2 shuffle amongst themselves:
- Production is unchanged → Production counter continues
- The **model** currently in the B slot changes, but we're tracking the **slot performance**, not the model identity
- This continues accumulating evidence until a real Production swap occurs

### 3. Database Evidence

Query of most recent 30 promotion events shows:

```
xgboost (most recent first):
- Oct 20 02:52:41 - decision: "alias_reorder", cycles: 9, prod_games: 1, b_games: 2
- Oct 20 02:22:34 - decision: "alias_reorder", cycles: 8, prod_games: 1, b_games: 2
- Oct 20 01:52:57 - decision: "alias_reorder", cycles: 7, prod_games: 1, b_games: 1
- ...continuing back...
- Oct 19 22:37:44 - decision: "swap_production_b", cycles: 0, prod_games: 0, b_games: 0 ✅

feedforward_nn:
- Oct 20 02:53:02 - decision: "alias_reorder", cycles: 138, prod_games: 27, b_games: 33
- Oct 20 02:22:56 - decision: "alias_reorder", cycles: 137, prod_games: 27, b_games: 33
- ...all showing "alias_reorder" with high cycle counts

multinomial_logistic:
- Oct 20 02:53:24 - decision: "alias_reorder", cycles: 74, prod_games: 19, b_games: 26
- Oct 20 02:23:17 - decision: "alias_reorder", cycles: 73, prod_games: 18, b_games: 26
- ...all showing "alias_reorder" with high cycle counts
```

**Observation:** Only xgboost has had a swap recently (Oct 19). The other two models have been steadily incrementing their cycle counters, meaning Production is consistently better than B (retain_production decision) but challenger reshuffling is still happening.

### 4. Training Job Logs Analysis

From `rps-trainer-29348790` (completed Oct 20 02:54):

```
2025-10-20 02:52:30 - xgboost: retaining Production alias (decision=insufficient_data)
2025-10-20 02:52:52 - feedforward_nn: retaining Production alias (decision=retain_production)
2025-10-20 02:53:13 - multinomial_logistic: retaining Production alias (decision=retain_production)

2025-10-20 02:54:24 - xgboost: decision=alias_reorder swap=False reorder=True
2025-10-20 02:54:24 - feedforward_nn: decision=alias_reorder swap=False reorder=True
2025-10-20 02:54:24 - multinomial_logistic: decision=alias_reorder swap=False reorder=True
```

**This confirms the bug!**

The script logs show:
1. First phase: Training completes, logs "decision=insufficient_data" or "decision=retain_production" ✅
2. Second phase: Auto-promotion runs, but final summary says "decision=alias_reorder" ❌

This is the OLD buggy code behavior where `decision` gets overwritten!

### 5. Deployed vs Repository Code Mismatch

**Deployed Images (outdated):**
- App: `ghcr.io/jimmyrisk/rps-app:dev-20251017e`
- Trainer: `ghcr.io/jimmyrisk/rps-trainer:dev-20251017d`

**Repository Code:**
- Last commit: `b3c846e` (Oct 17, but contains Oct 18 dated changes)
- Contains fix comment: "CHANGED 2025-10-18: Don't change decision field for alias reordering"

**Git log shows:** No commits to `auto_promote_models.py` since Oct 18 changes, meaning the fix is in the repo but NOT in the deployed images!

---

## Counter Reset Behavior Deep Dive

### Question: Do counters reset when B/shadow1/shadow2 reshuffle?

**Answer: NO, and this is correct by design.**

### Prometheus Counter Architecture

The counters are keyed by **alias name**, not model identity:

```python
# From app/metrics.py
_MODEL_ALIAS_GAMES_TOTAL_COUNTER = get_counter(
    "rps_model_games_by_alias_total",
    "Total games by model, alias, and difficulty (Production/B only)",
    ["model", "alias", "difficulty"]  # ← Alias is the label, not model version!
)
```

Example: `rps_model_games_by_alias_total{model="xgboost", alias="B", difficulty="standard"}`

This counter tracks:
- **All games played by whichever model is currently in the B slot**
- When shadow1 → B promotion happens, the **counter continues accumulating**
- The counter represents "B slot performance" not "shadow1 model performance"

### Why This Design?

The auto-promotion system compares:
- Production **slot** performance
- B **slot** performance

It doesn't care which specific model version is in each slot. It cares about:
- "Is the B slot beating the Production slot?"
- "If yes, swap them"

After the swap:
- The model that was in B → now in Production
- The model that was in Production → now in B
- **Both counters reset to 0** because we need fresh data about the new arrangement

But when just B/shadow1/shadow2 reshuffle:
- Production is unchanged
- B slot gets a new model, but we **don't reset** because we want to keep accumulating evidence
- If the new B model performs better, it will gradually improve the B win rate
- Eventually B might beat Production, triggering a swap

### Expected Counter Behavior After Different Events

#### Scenario 1: Production ↔ B Swap
```
Before swap:
  Production: 20 games, 15 wins (75%)
  B: 25 games, 22 wins (88%)
  → B is better, trigger swap

After swap:
  Production: 0 games, 0 wins (was B, reset) ✅
  B: 0 games, 0 wins (was Production, reset) ✅
  cycles_since_swap: 0 ✅
```

#### Scenario 2: B ↔ shadow1 Swap (Challenger Reorder)
```
Before reorder:
  Production: 20 games, 15 wins
  B: 25 games, 10 wins (40%)
  shadow1: (tracked via accuracy, not games)

B performing poorly, but shadow1 has better accuracy
→ Swap B ↔ shadow1

After reorder:
  Production: 20 games, 15 wins (unchanged) ✅
  B: 25 games, 10 wins (continues accumulating!) ✅
  cycles_since_swap: (unchanged, keep counting) ✅
```

**Why continue accumulating?**
- We want to give the new B model a chance to prove itself
- If we reset, we'd never accumulate enough evidence for a Production swap
- The counters represent "slot performance over time" not "current model performance"

---

## Related Code Cross-Check

### 1. Where Counters Are Incremented

**Per-game recording** (`app/model_serving.py:740`):
```python
from app.metrics import record_model_alias_game_result
record_model_alias_game_result(
    model=model_type,
    alias=alias,  # ← The alias that was selected for this game
    difficulty_mode=difficulty_mode,
    won=bot_won,
)
```

**Metrics implementation** (`app/metrics.py:386`):
```python
def record_model_alias_game_result(model: str, alias: str, difficulty_mode: str, won: bool) -> None:
    """Record game result for Production/B aliases only."""
    try:
        if alias in ["Production", "B"]:
            difficulty = _canonical_difficulty_label(difficulty_mode)
            # Always increment total games
            _MODEL_ALIAS_GAMES_TOTAL_COUNTER.labels(model=model, alias=alias, difficulty=difficulty).inc()
            
            # Increment wins or losses
            if won:
                _MODEL_ALIAS_GAME_WINS_COUNTER.labels(model=model, alias=alias, difficulty=difficulty).inc()
            else:
                _MODEL_ALIAS_GAME_LOSSES_COUNTER.labels(model=model, alias=alias, difficulty=difficulty).inc()
```

**Key observation:** These are **Prometheus Counters**, which only go up. They are NOT reset by application code.

### 2. How "Reset" Actually Works

Prometheus counters don't actually reset. Instead, the auto-promotion system:

1. **Records a baseline** when a swap occurs (in `promotion_events` table):
   ```sql
   decision = 'swap_production_b'
   production_wins = 15
   production_total_games = 20
   b_wins = 22
   b_total_games = 25
   ```

2. **Calculates deltas** on subsequent cycles (`app/promotion_store.py:calculate_ticker_snapshot`):
   ```python
   production_games_since_swap = _delta(
       current_production_total,
       baseline.production_total_games
   )
   ```

3. **Reports deltas** to Prometheus gauges (`app/metrics.py:update_games_since_swap`):
   ```python
   _PROMOTION_GAMES_SINCE_SWAP_GAUGE.labels(
       model=model, 
       alias=alias
   ).set(float(max(games_since_swap, 0)))
   ```

So "reset to 0" means:
- The **gauge** displays 0
- The underlying **counter** continues incrementing
- Future calculations subtract the baseline

### 3. Database Schema for Promotion Events

```sql
CREATE TABLE promotion_events (
    id INTEGER PRIMARY KEY,
    created_ts_utc TEXT,
    model_type TEXT,
    decision TEXT,  -- 'swap_production_b', 'retain_production', 'insufficient_data'
    z_statistic REAL,
    p_value REAL,
    production_wins INTEGER,
    production_total_games INTEGER,
    b_wins INTEGER,
    b_total_games INTEGER,
    reorder_applied INTEGER,  -- Boolean: did B/shadow1/shadow2 reshuffle?
    alias_rankings_json TEXT,
    alias_accuracies_json TEXT,
    alias_assignments_json TEXT,
    production_games_since_swap INTEGER,  -- Delta from last swap
    b_games_since_swap INTEGER,           -- Delta from last swap
    promotion_cycles_since_swap INTEGER,  -- Cycles since last swap
    reason TEXT,
    source TEXT,
    payload_json TEXT
)
```

The `promotion_store.py` module queries this table to find the most recent `swap_production_b` event and uses its counters as the baseline.

---

## Answers to Specific Questions

### Q1: Should "alias_reorder" only appear when there was NO prod/B swap?

**A:** In the FIXED code (current repo): YES ✅

The `decision` field should only ever be one of:
- `swap_production_b` - Production and B were swapped
- `retain_production` - Production was kept (higher/equal win rate)
- `insufficient_data` - Not enough games to decide

The `reorder_applied` boolean separately tracks whether challengers reshuffled.

So valid combinations are:
- `decision: swap_production_b, reorder_applied: true` - Both happened
- `decision: swap_production_b, reorder_applied: false` - Only swap, no reshuffle
- `decision: retain_production, reorder_applied: true` - Kept Production, but reshuffled B/shadow1/shadow2
- `decision: retain_production, reorder_applied: false` - Kept Production, no changes
- `decision: insufficient_data, reorder_applied: true` - Not enough data, but reshuffled challengers anyway
- `decision: insufficient_data, reorder_applied: false` - Not enough data, no changes

**In deployed code (buggy):** NO ❌
- If `reorder_applied: true`, the decision field gets overwritten to "alias_reorder"
- This loses information about whether Production was also swapped

### Q2: Are counters reset when there is an alias_reorder between B/shadow1/shadow2?

**A:** NO, and this is correct ✅

The counters track **slot performance**, not model identity. When challengers reshuffle:
- Production slot continues accumulating
- B slot continues accumulating with the new model
- Counters only reset when Production ↔ B swap occurs

**Rationale:**
- If we reset on every challenger reshuffle, we'd never accumulate enough evidence
- We want to measure "Is anything in the challenger pool better than Production?"
- By continuing accumulation, a newly-promoted B model gradually changes the B slot's win rate

### Q3: Does the reset also happen with prod/B swap?

**A:** YES ✅ (lines 505-510 of auto_promote_models.py)

```python
if test_result.should_swap():
    cycles_since_swap = 0
    production_games_since_swap = 0
    b_games_since_swap = 0
    ticker_note = "Ticker reset after Production/B swap"
```

When Production and B swap:
1. A new row is inserted into `promotion_events` with `decision='swap_production_b'`
2. The current counter values are recorded as the new baseline
3. The ticker gauges are set to 0
4. Future calculations subtract this baseline from current counters

### Q4: Is the cluster code different from the repo?

**A:** YES ❌ - This is the root cause!

**Deployed:**
- App: `ghcr.io/jimmyrisk/rps-app:dev-20251017e`
- Trainer: `ghcr.io/jimmyrisk/rps-trainer:dev-20251017d`
- Date: October 17, 2025

**Repository:**
- Last commit: `b3c846e` (Oct 17 but contains Oct 18 changes)
- Contains fix: "CHANGED 2025-10-18: Don't change decision field for alias reordering"

The fix was committed after the images were built and pushed!

---

## Evidence Summary Table

| Component | Expected Behavior | Observed Behavior | Status |
|-----------|-------------------|-------------------|---------|
| `decision` field | Should be hypothesis test result | Shows "alias_reorder" in logs | ❌ Bug in deployed code |
| `reorder_applied` field | Boolean for challenger reshuffling | Working correctly | ✅ |
| Counter reset on prod/B swap | Should reset to 0 | Code logic is correct | ✅ |
| Counter reset on challenger reorder | Should NOT reset | Code logic is correct | ✅ |
| Database records | Should show swap events | Found 1 xgboost swap on Oct 19 | ✅ |
| Deployed image version | Should be recent | Oct 17 (before Oct 18 fix) | ❌ |
| Repository code | Should have fix | Has fix dated Oct 18 | ✅ |

---

## Recommendations

### Immediate Actions

1. **Rebuild and redeploy images** to pick up the Oct 18 fix:
   ```bash
   ./scripts/build_push_deploy.sh --tag latest --push --deploy
   ```

2. **Verify fix deployment:**
   ```bash
   kubectl -n mlops-poc describe pod -l app=rps-app | grep "Image:"
   kubectl -n mlops-poc describe cronjob rps-trainer | grep "Image:"
   # Should show newer timestamps, not dev-20251017
   ```

3. **Monitor next training cycle** (runs every 30 minutes) and check logs:
   ```bash
   kubectl -n mlops-poc logs -f job/<next-trainer-job>
   # Look for final summary lines showing decision=retain_production or decision=swap_production_b
   # Should NOT see decision=alias_reorder anymore
   ```

4. **Verify database records** after next few cycles:
   ```bash
   kubectl -n mlops-poc exec deploy/rps-app -- python -c "
   from app.db import connect
   conn = connect()
   cursor = conn.execute('''
     SELECT created_ts_utc, model_type, decision, reorder_applied 
     FROM promotion_events 
     ORDER BY created_ts_utc DESC 
     LIMIT 10
   ''')
   for row in cursor.fetchall():
     print(row)
   "
   ```

### Testing Plan

After deployment, validate:

1. **Decision field correctness:**
   - Should see `retain_production` when Production is better
   - Should see `swap_production_b` when B is better
   - Should see `insufficient_data` when not enough games
   - Should NEVER see `alias_reorder` as a decision anymore

2. **Reorder flag correctness:**
   - Should see `reorder_applied=True` when challengers reshuffled
   - Should see `reorder_applied=False` when they didn't

3. **Counter reset behavior:**
   - When `decision=swap_production_b`: both prod and B counters → 0
   - When `decision=retain_production` with `reorder_applied=True`: counters continue accumulating
   - When `decision=insufficient_data`: counters continue accumulating

4. **Grafana panels:**
   - Promotion decision panels should show clean swap/retain/insufficient data
   - No more "alias_reorder" noise
   - Counter reset events should be visible when swaps occur

### Documentation Updates

1. Update `.github/copilot-instructions.md` to emphasize:
   - The importance of rebuilding images when `scripts/auto_promote_models.py` changes
   - The trainer CronJob pulls whatever image tag is current, doesn't auto-rebuild

2. Add to operations runbook:
   - How to verify deployed image versions match repository
   - Standard procedure for deploying auto-promotion changes

3. Update Grafana dashboard guide:
   - Explain the decision field values
   - Explain the reorder_applied boolean
   - Document the ticker/counter relationship

---

## Technical Deep Dive: Why the Old Code Was Wrong

### The Logical Flaw

The old code tried to use `decision` to represent two different things:

1. **Hypothesis test result**: Is B better than Production?
2. **Action taken**: Did we reshuffle aliases?

These are orthogonal concepts:
- You can swap Production/B AND reshuffle challengers
- You can keep Production AND reshuffle challengers
- You can swap Production/B and NOT reshuffle challengers
- etc.

By overwriting `decision` with "alias_reorder", the old code lost information about the hypothesis test result.

### The Fix

The new code separates these concerns:

```python
# Hypothesis test result (3 possible values)
decision = test_result.decision  # swap_production_b | retain_production | insufficient_data

# Action taken (boolean)
reorder_applied = (True | False)
```

Now you have complete information:
- What did the statistical test conclude?
- What alias changes were made?

### Why This Matters for Metrics

Prometheus metrics and Grafana dashboards were designed expecting `decision` to be the hypothesis test result:

```promql
# Count Production swaps over time
rate(rps_model_promotion_events_total{decision="swap_production_b"}[5m])

# Count Production retentions over time
rate(rps_model_promotion_events_total{decision="retain_production"}[5m])
```

When `decision` was "alias_reorder", these queries broke because:
- Swaps were miscategorized as reorders
- Can't distinguish "kept Production because it's better" from "kept Production but reshuffled"
- Lost the ability to track swap frequency

With the fix:
- `decision` is always the hypothesis test result
- `reorder_applied` boolean tracks challenger reshuffling
- Grafana queries work correctly
- Human readers understand what happened

---

## Conclusion

The promotion system logic is **fundamentally sound** in the current repository:

✅ Hypothesis testing is correct  
✅ Counter reset logic is correct  
✅ Alias reordering logic is correct  
✅ Database persistence is correct  
✅ Metrics recording is correct  

The only issue is that **deployed images are outdated** and contain the pre-fix code where `decision` gets overwritten.

**Next step:** Rebuild and redeploy images to get the Oct 18 fix into production.

**Expected outcome after fix:**
- Database will show proper decision values (swap_production_b, retain_production, insufficient_data)
- Logs will clearly show when Production swaps occur vs when only challengers reshuffle
- Grafana panels will correctly categorize promotion events
- Counter reset behavior will be observable in telemetry

---

## Appendix: Code Snippets for Verification

### Check Decision Distribution in Database

```bash
kubectl -n mlops-poc exec deploy/rps-app -- python -c "
from app.db import connect
from collections import Counter
conn = connect()
cursor = conn.execute('SELECT decision FROM promotion_events')
decisions = [row[0] for row in cursor.fetchall()]
print(Counter(decisions))
"
```

Expected after fix:
```
Counter({
  'retain_production': 400,
  'insufficient_data': 50,
  'swap_production_b': 3,
  'alias_reorder': 0  # ← Should be 0 after fix!
})
```

### Check Recent Swap Events

```bash
kubectl -n mlops-poc exec deploy/rps-app -- python -c "
from app.db import connect
conn = connect()
cursor = conn.execute('''
  SELECT created_ts_utc, model_type, production_total_games, b_total_games 
  FROM promotion_events 
  WHERE decision = \"swap_production_b\"
  ORDER BY created_ts_utc DESC 
  LIMIT 5
''')
for row in cursor.fetchall():
  print(row)
"
```

### Monitor Next Training Job

```bash
# Wait for next training job (CronJob runs every 30 mins)
kubectl -n mlops-poc get jobs -w

# When new job appears, follow logs
kubectl -n mlops-poc logs -f job/rps-trainer-XXXXX | grep -E "(decision=|swap_production|alias_reorder)"
```

Look for final summary lines like:
```
xgboost: decision=retain_production swap=False reorder=True
feedforward_nn: decision=swap_production_b swap=True reorder=True
multinomial_logistic: decision=insufficient_data swap=False reorder=False
```

Should NOT see `decision=alias_reorder` anywhere!

---

**End of Review**
