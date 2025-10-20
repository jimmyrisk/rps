# Deployment Notes: Legacy Bot Game Downsampling (2025-10-19)

## Summary
Implemented legacy bot game downsampling to shift training data composition from bot-dominated to human-dominated gameplay. This change reduces legacy bot game frequency by 5x both in data collection and training data usage.

## Changes Made

### 1. Configuration (`app/config.py`)
- Added `TRAINING_LEGACY_GAME_STRIDE` config parameter (default: 5)
- Added `get_training_legacy_game_stride()` helper function
- Configurable via environment variable: `TRAINING_LEGACY_GAME_STRIDE=N`

### 2. Training Data Loading (`trainer/base_model.py`)
- Modified `load_data()` to downsample legacy bot games
- **Logic:**
  - Identifies legacy bot games by `player_name` matching: Ace, Bob, Cal, Dan, Edd, Fox, Gus, Hal
  - Keeps only every Nth legacy game (configurable, default N=5)
  - **Preserves 100% of human games** (no downsampling)
  - Handles edge case of zero legacy games gracefully
- **Output:** Logs downsampling statistics during training

### 3. Legacy Gameplay CronJob (`infra/k8s/22-legacy-gameplay-cronjob.yaml`)
- **Before:** `*/30 * * * *` (every 30 minutes)
- **After:** `*/150 * * * *` (every 150 minutes = 2.5 hours)
- **Reduction:** 5x less frequent (aligned with downsampling stride)

### 4. Training Gating Logic (`trainer/train_all_aliases.py`)
- Added clarifying comment: thresholds represent raw DB counts **before** downsampling
- No numerical changes needed (gating happens before downsampling, which is correct)

### 5. Documentation Updates
- `docs/operations.md`: Added "Training data composition" section
- `docs/file_index.md`: Updated CronJob schedule documentation
- `scripts/play_legacy_game.sh`: Updated comment with new schedule

## Testing
- ✅ All essential tests pass (`tests/test_essential.py`)
- ✅ Downsampling logic handles zero legacy games gracefully
- ✅ Feature count remains 50 (no breaking changes)
- ✅ YAML validation passes

## Deployment Steps
1. Build and push new Docker images:
   ```bash
   ./scripts/build_push_deploy.sh --tag latest --push --deploy
   ```

2. Update CronJob manifest:
   ```bash
   kubectl apply -f infra/k8s/22-legacy-gameplay-cronjob.yaml -n mlops-poc
   ```

3. Verify deployment:
   ```bash
   # Check app rollout
   kubectl -n mlops-poc rollout status deployment/rps-app
   
   # Check CronJob schedule
   kubectl -n mlops-poc get cronjob legacy-gameplay -o jsonpath='{.spec.schedule}'
   # Should output: */150 * * * *
   
   # Verify downsampling in trainer logs (next training run)
   kubectl -n mlops-poc logs -l app=rps-trainer --tail=100 | grep "Legacy game downsampling"
   # Expected output similar to:
   # 🎲 Legacy game downsampling (stride=5):
   #    Legacy bot games: kept 40/200 (20.0% of legacy)
   #    Human games: kept 100/100 (100% preserved)
   #    Total events: 1400 (down from 2000)
   ```

4. Spot-check the production dataset to confirm the 1-in-5 retention window:
   ```bash
   kubectl -n mlops-poc exec deploy/rps-app -- python - <<'PY'
   from trainer.base_model import BaseRPSModel
   from app.legacy_models import LEGACY_POLICIES, LEGACY_DISPLAY_NAMES

   _LEGACY_TOKENS = {token.lower() for token in LEGACY_POLICIES}
   _LEGACY_NAMES = {name.lower() for name in LEGACY_DISPLAY_NAMES.values()}

   class _Probe(BaseRPSModel):
      def __init__(self):
         super().__init__(model_name="probe", model_type="probe")
      def needs_feature_scaling(self):
         return True
      def get_hyperparameters(self):
         return {}
      def run_cross_validation(self, *args, **kwargs):
         return [], 0.0, 0.0
      def train_final_model(self, *args, **kwargs):
         pass
      def predict(self, X):
         return []
      def create_pyfunc_model(self, **kwargs):
         pass

   probe = _Probe()
   events, _ = probe.load_data()
   def _is_legacy(player):
      if not isinstance(player, str):
         return False
      lowered = player.strip().lower()
      if not lowered:
         return False
      if lowered in _LEGACY_TOKENS or lowered in _LEGACY_NAMES:
         return True
      for token in ''.join(ch if ch.isalpha() else ' ' for ch in lowered).split():
          if token in _LEGACY_TOKENS:
             return True
      return False

   legacy = events[events["player_name"].apply(_is_legacy)]
   ratio = (len(legacy) / len(events) * 100.0) if len(events) else 0.0
   print(f"Total events: {len(events)} | Legacy events kept: {len(legacy)} ({ratio:.1f}% of events)")
   PY
   ```

## Expected Impact

### Immediate Effects
- **Data collection:** Legacy games generated 5x less frequently (every 2.5 hours vs every 30 min)
- **Training data:** Only 20% of legacy bot games used (1 in 5 kept)
- **Human data:** 100% preserved (no change)

### Long-term Effects
- Training datasets will gradually become more human-dominated
- Legacy bot patterns will have less influence on model behavior
- Model performance should better reflect human player patterns

## Rollback Plan
If issues arise:

1. Revert CronJob schedule:
   ```bash
   kubectl -n mlops-poc edit cronjob legacy-gameplay
   # Change schedule back to: */30 * * * *
   ```

2. Disable downsampling:
   ```bash
   kubectl -n mlops-poc set env deployment/rps-app TRAINING_LEGACY_GAME_STRIDE=1
   kubectl -n mlops-poc set env cronjob/rps-trainer TRAINING_LEGACY_GAME_STRIDE=1
   ```

3. Or revert to previous image:
   ```bash
   kubectl -n mlops-poc rollout undo deployment/rps-app
   ```

## Configuration Reference

### Environment Variables
- `TRAINING_LEGACY_GAME_STRIDE=5` (default) - Keep 1 in 5 legacy games
- `TRAINING_LEGACY_GAME_STRIDE=1` - Disable downsampling (keep all games)
- `TRAINING_LEGACY_GAME_STRIDE=10` - More aggressive (keep 1 in 10)

### Legacy Bot Names (for reference)
Games with these `player_name` values are considered legacy bots:
- Ace, Bob, Cal, Dan (original 4)
- Edd, Fox, Gus, Hal (newer additions)

All other games (including null/empty player_name) are treated as human games.

## Notes
- Downsampling preserves chronological ordering
- Game-stratified cross-validation still works (entire games kept or removed)
- No changes to feature extraction or model architecture
- MIN_NEW_ROWS_FOR_TRAINING and MIN_TOTAL_ROWS thresholds unchanged (they check raw DB, which is correct)
