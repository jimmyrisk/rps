#!/usr/bin/env python3
"""
Post-training script to register all 4 aliases for newly trained models.
This runs after train_all_models.py completes successfully.
"""
import os
import sys
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime

# Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "https://dagshub.com/jimmyrisk/rps.mlflow")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "rps-bot-production")

# Model registry names
MODEL_NAMES = {
    "xgboost": "xgboost",
    "feedforward": "feedforward_nn",
    "mnlogit": "multinomial_logistic",
}

# All 4 aliases to maintain
ALIASES = ["Production", "B", "shadow1", "shadow2"]


def setup_mlflow():
    """Configure MLflow client"""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return MlflowClient()


def get_recent_model_versions(client, model_name, count=4):
    """Get the N most recent versions of a model"""
    try:
        all_versions = client.search_model_versions(f"name='{model_name}'")
        # Sort by version number (descending)
        sorted_versions = sorted(all_versions, key=lambda v: int(v.version), reverse=True)
        return sorted_versions[:count]
    except Exception as e:
        print(f"❌ Error fetching versions for {model_name}: {e}")
        return []


def assign_aliases(client, model_name, versions):
    """Assign all 4 aliases to the most recent 4 versions"""
    if len(versions) < len(ALIASES):
        print(f"⚠️  Only {len(versions)} versions available for {model_name}, need {len(ALIASES)}")
        # Assign what we can
        aliases_to_assign = ALIASES[:len(versions)]
    else:
        aliases_to_assign = ALIASES
    
    print(f"\n📌 Assigning aliases for {model_name}:")
    for alias, version in zip(aliases_to_assign, versions):
        try:
            client.set_registered_model_alias(
                name=model_name,
                alias=alias,
                version=version.version
            )
            print(f"  ✅ {alias:12} → version {version.version}")
        except Exception as e:
            print(f"  ❌ Failed to set {alias} → v{version.version}: {e}")


def main():
    """Main execution"""
    print("=" * 80)
    print("POST-TRAINING ALIAS ASSIGNMENT")
    print("=" * 80)
    print(f"MLflow URI: {MLFLOW_TRACKING_URI}")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Time: {datetime.now().isoformat()}")
    print()
    
    client = setup_mlflow()
    
    # Process each model type
    for model_key, model_name in MODEL_NAMES.items():
        print(f"\n{'='*80}")
        print(f"Model: {model_name} ({model_key})")
        print(f"{'='*80}")
        
        # Get recent versions
        versions = get_recent_model_versions(client, model_name, count=len(ALIASES))
        
        if not versions:
            print(f"❌ No versions found for {model_name}")
            continue
        
        print(f"Found {len(versions)} version(s):")
        for v in versions:
            print(f"  - Version {v.version} (run: {v.run_id[:8]}...)")
        
        # Assign aliases
        assign_aliases(client, model_name, versions)
    
    print("\n" + "=" * 80)
    print("✅ Alias assignment complete!")
    print("=" * 80)
    print("\nCurrent alias assignments:")
    
    # Show final state
    for model_key, model_name in MODEL_NAMES.items():
        print(f"\n{model_name}:")
        model_details = client.get_registered_model(model_name)
        if hasattr(model_details, 'aliases') and model_details.aliases:
            for alias, version in model_details.aliases.items():
                print(f"  {alias:12} → version {version}")
        else:
            # Fallback: check each alias individually
            for alias in ALIASES:
                try:
                    version_info = client.get_model_version_by_alias(model_name, alias)
                    print(f"  {alias:12} → version {version_info.version}")
                except:
                    print(f"  {alias:12} → (not set)")


if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
