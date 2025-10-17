#!/usr/bin/env python3
"""
Register experiment runs as models and set up aliases for A/B testing.
This completes Phase 1 of the model storage migration.
"""
import os
import mlflow
from mlflow.tracking import MlflowClient

# Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "https://dagshub.com/jimmyrisk/rps.mlflow")
EXPERIMENT_NAME = "rps-bot-production"

# Model type mapping: code_name -> (mlflow_registry_name, experiment_tag_value)
MODEL_MAPPING = {
    "xgboost": ("xgboost", "xgboost"),
    "feedforward_nn": ("feedforward_nn", "feedforward"),
    "multinomial_logistic": ("multinomial_logistic", "multinomial_logistic"),
}

# Aliases to create (for A/B testing and shadow predictions)
ALIASES = ["Production", "B", "shadow1", "shadow2"]

def setup_mlflow():
    """Configure MLflow client"""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return MlflowClient()

def get_latest_runs_by_model_type(client, experiment_id, model_tag_value, count=4):
    """Get the latest N runs for a specific model type"""
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.model_type = '{model_tag_value}'",
        order_by=["start_time DESC"],
        max_results=count
    )
    return runs

def register_model_from_run(client, run_id, model_name):
    """Register a model from an experiment run"""
    try:
        # Register the model
        model_uri = f"runs:/{run_id}/model"
        result = mlflow.register_model(model_uri, model_name)
        print(f"  ✅ Registered version {result.version} from run {run_id[:8]}")
        return result
    except Exception as e:
        print(f"  ❌ Failed to register: {e}")
        return None

def set_alias_for_version(client, model_name, version, alias):
    """Set an alias for a model version"""
    try:
        client.set_registered_model_alias(model_name, alias, version)
        print(f"  ✅ Set alias '{alias}' -> version {version}")
        return True
    except Exception as e:
        print(f"  ❌ Failed to set alias '{alias}': {e}")
        return False

def main():
    """Main registration process"""
    print("🚀 Starting Phase 1: Model Registration and Alias Setup")
    print(f"   MLflow URI: {MLFLOW_TRACKING_URI}")
    print(f"   Experiment: {EXPERIMENT_NAME}")
    print()
    
    client = setup_mlflow()
    
    # Get experiment ID
    try:
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        if not experiment:
            print(f"❌ Experiment '{EXPERIMENT_NAME}' not found!")
            return
        experiment_id = experiment.experiment_id
        print(f"📊 Found experiment: {EXPERIMENT_NAME} (ID: {experiment_id})")
        print()
    except Exception as e:
        print(f"❌ Error getting experiment: {e}")
        return
    
    # Process each model type
    for model_type, (registry_name, tag_value) in MODEL_MAPPING.items():
        print(f"📦 Processing {model_type} (registry name: {registry_name}):")
        
        # Get latest runs
        runs = get_latest_runs_by_model_type(client, experiment_id, tag_value, count=len(ALIASES))
        if len(runs) < len(ALIASES):
            print(f"  ⚠️  Warning: Only found {len(runs)} runs, need {len(ALIASES)} for all aliases")
        
        # Check if model already registered
        try:
            existing_versions = client.search_model_versions(f"name='{registry_name}'")
            if existing_versions:
                print(f"  ℹ️  Model already registered with {len(existing_versions)} versions")
                # Just set aliases on existing versions
                for i, alias in enumerate(ALIASES):
                    if i < len(existing_versions):
                        version = existing_versions[i].version
                        set_alias_for_version(client, registry_name, version, alias)
                print()
                continue
        except Exception:
            # Model doesn't exist, will register it
            pass
        
        # Register models from runs
        registered_versions = []
        for run in runs:
            result = register_model_from_run(client, run.info.run_id, registry_name)
            if result:
                registered_versions.append(result.version)
        
        # Set aliases
        print(f"  📌 Setting up aliases:")
        for i, alias in enumerate(ALIASES):
            if i < len(registered_versions):
                set_alias_for_version(client, registry_name, registered_versions[i], alias)
            else:
                print(f"  ⚠️  Skipping '{alias}' - not enough registered versions")
        
        print()
    
    print("=" * 60)
    print("✅ Phase 1 Complete!")
    print()
    print("Next steps:")
    print("  1. Verify aliases with: kubectl exec deployment/rps-app -- python -c '...'")
    print("  2. Proceed to Phase 2: Populate local storage")
    print("=" * 60)

if __name__ == "__main__":
    main()
