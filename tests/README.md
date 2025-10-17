# Tests Directory

## Quick Reference

```bash
# Local tests (no cluster)
./tests/run_tests.sh

# Cluster validation (after deploy)
./tests/test_cluster_e2e.sh [bot_names...]
```

## Active Suites

### Core (fast, offline)

| File | Purpose | Runtime |
|------|---------|---------|
| `test_essential.py` | Core offline tests | ~5s |
| `test_critical_endpoints.py` | API contract validation | ~10s |
| `test_model_manager_metrics.py` | Prometheus gauge regression (`pytest`) | ~5s |

### Extended integration

| File | Purpose | Notes |
|------|---------|-------|
| `test_comprehensive_games.py` | All bot×difficulty combos (18 games) | ~90s, offline |
| `test_features_db_integration.py` | Feature extraction + DB contract | offline |
| `test_phased_validation_v2.py` | Three-phase gating harness | configurable via CLI |
| `test_e2e_metrics.py` | **Cluster metrics pipeline** | Requires deployed API |

### Experiment & manual checks

| File | Coverage |
|------|----------|
| `test_danger_penalty.py` | Multinomial logistic danger penalty unit tests |
| `test_experiment_integration.py` | Hyperparameter experiment orchestration |
| `test_alias_continuation.py` | Alias-preserving continuation training smoke tests |
| `test_frontend_integration.py` | FastAPI TestClient simulation of UI flows |
| `test_javascript_simulation.py` | Mirrors historical JS regressions |
| `test_full_game_metrics.py` | Long-running Prometheus + DB audit |
| `test_new_legacy_policies.py` | Legacy policy smoke tests against live API |

### Archived stubs

`test_integration.py`, `test_validation.py`, `test_phase3_comprehensive.py`, and every
file under `tests/archive/` now raise immediately. They are retained only so links in
historical documentation remain valid—run the v2 phased suite or the dedicated modules
listed above instead.

## Scripts

- **`run_tests.sh`** - Curated local bundle (core + targeted integration)
- **`test_cluster_e2e.sh`** - Cluster validation with health checks
- **`quick_validation_test.sh`** - Phase 1 smoke + critical endpoints

## Usage

### After Code Changes
```bash
./tests/run_tests.sh
```

### After Cluster Deploy
```bash
# After completing a Docker rollout:
./tests/test_cluster_e2e.sh brian  # Quick test
./tests/test_cluster_e2e.sh        # Full test
```

### Environment Variables
```bash
# test_e2e_metrics.py
API_BASE=https://mlops-rps.uk  # Default
TEST_BOTS="brian,forrest,logan"  # Default

# Example: test specific bot
TEST_BOTS="brian" python3 tests/test_e2e_metrics.py
```

See `docs/operations.md` for full testing guide.

