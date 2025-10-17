#!/usr/bin/env python3
"""
Batch promote specific model versions to aliases in MLflow.

This script promotes the specified model versions to their respective aliases:

XGBoost:
- Production: v104 (config07)
- B: v99 (config02)
- shadow1: v103 (config06)
- shadow2: v105 (config08)

Feedforward NN:
- Production: v248 (config04)
- B: v246 (config02)
- shadow1: v249 (config05)
- shadow2: v250 (config06)

Multinomial Logistic:
- Production: v93 (config06)
- B: v99 (config12)
- shadow1: v96 (config09)
- shadow2: v97 (config10)
"""
import os
import sys
import mlflow
from pathlib import Path
from mlflow.tracking import MlflowClient

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_mlflow_tracking_uri

# Model promotions configuration
PROMOTIONS = {
    'rps_bot_xgboost': {
        'Production': {'version': '104', 'config': 'config07'},
        'B': {'version': '99', 'config': 'config02'},
        'shadow1': {'version': '103', 'config': 'config06'},
        'shadow2': {'version': '105', 'config': 'config08'},
    },
    'rps_bot_feedforward': {
        'Production': {'version': '248', 'config': 'config04'},
        'B': {'version': '246', 'config': 'config02'},
        'shadow1': {'version': '249', 'config': 'config05'},
        'shadow2': {'version': '250', 'config': 'config06'},
    },
    'rps_bot_mnlogit': {
        'Production': {'version': '93', 'config': 'config06'},
        'B': {'version': '99', 'config': 'config12'},
        'shadow1': {'version': '96', 'config': 'config09'},
        'shadow2': {'version': '97', 'config': 'config10'},
    },
}


def promote_models():
    """Promote models to specified aliases"""
    # Setup MLflow
    mlflow_uri = get_mlflow_tracking_uri()
    mlflow.set_tracking_uri(mlflow_uri)
    client = MlflowClient()
    
    print("🚀 Starting Batch Model Promotion")
    print(f"   MLflow URI: {mlflow_uri}")
    print()
    
    success_count = 0
    fail_count = 0
    
    for model_name, aliases in PROMOTIONS.items():
        print(f"📦 Processing {model_name}:")
        
        for alias, info in aliases.items():
            version = info['version']
            config = info['config']
            
            try:
                # Set the alias to point to the specified version
                client.set_registered_model_alias(model_name, alias, version)
                print(f"  ✅ Set {alias} -> v{version} ({config})")
                success_count += 1
            except Exception as e:
                print(f"  ❌ Failed to set {alias} -> v{version}: {e}")
                fail_count += 1
        
        print()
    
    print(f"📊 Summary: {success_count} successes, {fail_count} failures")
    
    if fail_count > 0:
        print("\n⚠️  Some promotions failed. Please check MLflow registry and try again.")
        return False
    else:
        print("\n✅ All models promoted successfully!")
        return True


if __name__ == "__main__":
    success = promote_models()
    sys.exit(0 if success else 1)
