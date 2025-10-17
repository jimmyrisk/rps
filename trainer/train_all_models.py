#!/usr/bin/env python3
"""
Unified training orchestrator that can train all three models with consistent interface.
Includes data-driven training triggers to only train when sufficient new data is available.
Implements SEQUENTIAL training with garbage collection to prevent OOM.
"""
import os
import sys
import gc
import time
import sqlite3
import argparse
from pathlib import Path
from datetime import datetime, timedelta

# Add trainer directory to path
sys.path.append(str(Path(__file__).parent))

from train_mnlogit_torch import MNLogitRPSModel
from train_xgboost import XGBoostRPSModel  
from train_feedforward import FeedforwardRPSModel


def check_training_needed(db_path: str, min_new_rows: int = 50, min_total_rows: int = 300) -> tuple[bool, dict]:
    """
    Check if training is needed based on new data availability.
    Excludes test games from count.
    Returns (should_train, stats_dict)
    """
    # Get training data cutoff date from environment
    training_since = os.getenv("TRAINING_DATA_SINCE_DATE", "2025-10-03T00:00:00+00:00")
    
    print(f"🔍 DEBUG: check_training_needed called with db_path={db_path}")
    print(f"🔍 DEBUG: min_new_rows={min_new_rows}, min_total_rows={min_total_rows}")
    
    try:
        print(f"🔍 DEBUG: training_since={training_since}")
        
        with sqlite3.connect(db_path) as conn:
            # Get total non-test rows within time window
            total_rows = conn.execute(
                """SELECT COUNT(*) FROM events e
                   INNER JOIN games g ON e.game_id = g.id
                   WHERE g.is_test = 0
                   AND (g.created_ts_utc IS NULL OR g.created_ts_utc >= ?)""",
                (training_since,)
            ).fetchone()[0]
            
            print(f"🔍 DEBUG: total_rows={total_rows}")
            
            # FIX: Use created_ts_utc from games table (properly formatted UTC timestamps)
            # instead of e.ts which has inconsistent format
            # Check last 10 minutes (since CronJob runs every 5 minutes)
            cutoff_time = datetime.now() - timedelta(minutes=10)
            cutoff_iso = cutoff_time.strftime('%Y-%m-%dT%H:%M:%S') + '+00:00'
            
            # DEBUG: Print what we're comparing
            print(f"   DEBUG: Cutoff timestamp: {cutoff_iso}")
            print(f"   DEBUG: Training since: {training_since}")
            
            # Count new non-test rows since cutoff within time window
            # Use game.created_ts_utc which is properly formatted
            new_rows = conn.execute(
                """SELECT COUNT(*) FROM events e
                   INNER JOIN games g ON e.game_id = g.id
                   WHERE g.is_test = 0 
                   AND g.created_ts_utc IS NOT NULL
                   AND g.created_ts_utc > ?
                   AND g.created_ts_utc >= ?""",
                (cutoff_iso, training_since)
            ).fetchone()[0]
            
            # DEBUG: Check without the training_since filter
            new_rows_no_filter = conn.execute(
                """SELECT COUNT(*) FROM events e
                   INNER JOIN games g ON e.game_id = g.id
                   WHERE g.is_test = 0 
                   AND g.created_ts_utc IS NOT NULL
                   AND g.created_ts_utc > ?""",
                (cutoff_iso,)
            ).fetchone()[0]
            print(f"   DEBUG: New rows (no training_since filter): {new_rows_no_filter}")
            
            # Count test games for reporting
            test_games = conn.execute(
                "SELECT COUNT(*) FROM games WHERE is_test = 1"
            ).fetchone()[0]
            
            stats = {
                "total_rows": total_rows,
                "new_rows_since_cutoff": new_rows,
                "min_new_rows_threshold": min_new_rows,
                "min_total_rows_threshold": min_total_rows,
                "cutoff_time": cutoff_iso,
                "training_data_since": training_since,
                "test_games_excluded": test_games
            }
            
            should_train = (total_rows >= min_total_rows and new_rows >= min_new_rows)
            
            return should_train, stats
            
    except Exception as e:
        print(f"⚠️  Error checking training requirements: {e}")
        # Default to training if we can't check
        return True, {"error": str(e)}


def train_all_models(models_to_train=None, force_training=False):
    """Train all specified models or all models if none specified"""
    
    # Check if training is needed based on data availability
    data_path = os.getenv("DATA_PATH", "/data")
    db_path = os.path.join(data_path, "rps.db")
    min_new_rows = int(os.getenv("MIN_NEW_ROWS_FOR_TRAINING", "50"))
    min_total_rows = int(os.getenv("MIN_TOTAL_ROWS", "300"))
    
    if not force_training:
        should_train, stats = check_training_needed(db_path, min_new_rows, min_total_rows)
        
        print(f"📊 Data-driven training check:")
        print(f"   Total rows: {stats.get('total_rows', 'unknown')}")
        print(f"   New rows: {stats.get('new_rows_since_cutoff', 'unknown')}")
        print(f"   Test games excluded: {stats.get('test_games_excluded', 0)}")
        print(f"   Training data since: {stats.get('training_data_since', 'N/A')[:10]}")
        print(f"   Thresholds: {min_new_rows} new, {min_total_rows} total")
        print(f"   Training needed: {'✅ Yes' if should_train else '❌ No'}")
        
        if not should_train:
            print(f"\n⏭️  Skipping training - insufficient new data")
            return {"status": "skipped", "reason": "insufficient_new_data", "stats": stats}
    else:
        print("🔥 Force training enabled - skipping data checks")
    
    available_models = {
        'mnlogit': MNLogitRPSModel,
        'xgboost': XGBoostRPSModel,
        'feedforward': FeedforwardRPSModel
    }
    
    if models_to_train is None:
        models_to_train = list(available_models.keys())
    
    print(f"🚀 Training models: {', '.join(models_to_train)}")
    print(f"⚠️  SEQUENTIAL mode: Training one model at a time to prevent OOM")
    
    results = {}
    for model_name in models_to_train:
        if model_name not in available_models:
            print(f"⚠️  Unknown model: {model_name}. Available: {list(available_models.keys())}")
            continue
            
        print(f"\n{'='*60}")
        print(f"Training {model_name.upper()} (sequential mode)")
        print(f"{'='*60}")
        
        try:
            model_class = available_models[model_name]
            model = model_class()
            model.train()
            results[model_name] = "SUCCESS"
            print(f"✅ {model_name} training completed successfully")
            
            # CRITICAL: Force garbage collection between models to prevent OOM
            del model  # Explicitly delete model object
            gc.collect()  # Force Python garbage collection
            
            # Give system time to release memory
            if model_name != models_to_train[-1]:  # Don't wait after last model
                print(f"⏳ Waiting 5 seconds for memory cleanup...")
                time.sleep(5)
                print(f"✅ Memory cleanup complete, ready for next model")
            
        except Exception as e:
            print(f"❌ {model_name} training failed: {str(e)}")
            results[model_name] = f"FAILED: {str(e)}"
            
            # Still clean up memory even on failure
            try:
                del model
                gc.collect()
            except:
                pass
    
    print(f"\n{'='*50}")
    print("TRAINING SUMMARY")
    print(f"{'='*50}")
    for model_name, status in results.items():
        status_emoji = "✅" if status == "SUCCESS" else "❌"
        print(f"{status_emoji} {model_name}: {status}")
    
    return results


def compare_models():
    """Print comparison guide for the trained models"""
    print(f"\n{'='*50}")
    print("MODEL COMPARISON GUIDE")
    print(f"{'='*50}")
    print("Check your MLflow UI to compare models:")
    print("1. Open MLflow UI in browser")
    print("2. Navigate to the 'rps-bot' experiment") 
    print("3. Compare runs by cv_mean_acc and test_accuracy")
    print("4. Look for consistent validation patterns")
    print("\nKey metrics to compare:")
    print("• cv_mean_acc ± cv_std_acc (cross-validation performance)")
    print("• test_accuracy (final holdout performance)")
    print("• Per-class accuracies (rock_acc, paper_acc, scissors_acc)")
    print("• Training time and model complexity")


def main():
    parser = argparse.ArgumentParser(description="Train RPS models with data-driven triggers")
    parser.add_argument(
        "--models", 
        nargs="+", 
        choices=["mnlogit", "xgboost", "feedforward"],
        help="Models to train. If not specified, trains all models."
    )
    parser.add_argument(
        "--compare-only", 
        action="store_true",
        help="Only show comparison guide without training"
    )
    parser.add_argument(
        "--force", 
        action="store_true",
        help="Force training even if insufficient new data"
    )
    
    args = parser.parse_args()
    
    # Check for environment variable override for force training
    force_training_env = os.getenv("FORCE_TRAINING", "").lower() in ("true", "1", "yes")
    force_training = args.force or force_training_env
    
    if force_training_env:
        print("🔥 Force training enabled via FORCE_TRAINING environment variable")
    
    if args.compare_only:
        compare_models()
        return
    
    # Train models
    results = train_all_models(args.models, force_training=force_training)
    
    # Handle skipped training
    if isinstance(results, dict) and results.get("status") == "skipped":
        print(f"✅ Training orchestrator completed - no training needed")
        sys.exit(0)
    
    # Show comparison guide
    compare_models()
    
    # Exit with error code if any models failed
    failed_models = [name for name, status in results.items() if status != "SUCCESS"]
    if failed_models:
        print(f"\n⚠️  {len(failed_models)} model(s) failed training")
        sys.exit(1)
    else:
        print(f"\n🎉 All {len(results)} model(s) trained successfully!")
        
        # Assign all 4 aliases (Production, B, shadow1, shadow2) to recent versions
        print(f"\n{'='*50}")
        print("ASSIGNING ALIASES TO ALL MODEL VERSIONS")
        print(f"{'='*50}")
        try:
            from trainer.assign_aliases import main as assign_aliases_main
            assign_aliases_main()
        except Exception as e:
            print(f"⚠️  Alias assignment failed: {e}")
            print("Models are trained but aliases not updated. Run assign_aliases.py manually.")


if __name__ == "__main__":
    main()