"""
Model management route handlers.
Handles model loading, unloading, staging, and A/B testing configuration.
"""
from fastapi import APIRouter, HTTPException
from app.model_serving import get_model_manager, AVAILABLE_MODELS

router = APIRouter(prefix="/models")

@router.get("")
def list_models():
    """List available models and their status"""
    manager = get_model_manager()
    available = manager.list_available_models()
    loaded_info = manager.get_all_model_info()
    usage_info = manager.model_usage
    
    result = {}
    for model_type, config in available.items():
        result[model_type] = {
            "name": config["name"],
            "description": config["description"],
            "loaded": manager.is_model_loaded(model_type),
            "info": loaded_info.get(model_type, {}),
            "usage": usage_info.get(model_type, {}),
        }
    return result

@router.post("/{model_type}/load")
def load_model(model_type: str, force_reload: bool = False):
    """Load a specific model"""
    manager = get_model_manager()
    success = manager.load_model(model_type, force_reload=force_reload)
    
    if success:
        info = manager.get_model_info(model_type)
        return {"success": True, "model_type": model_type, "info": info}
    else:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {model_type}")

@router.delete("/{model_type}")
def unload_model(model_type: str):
    """Unload a specific model to free memory"""
    manager = get_model_manager()
    success = manager.unload_model(model_type)
    
    if success:
        return {"success": True, "model_type": model_type, "message": "Model unloaded"}
    else:
        raise HTTPException(status_code=404, detail=f"Model not loaded: {model_type}")

@router.get("/stages")
def list_all_model_stages():
    """List all models and their current stages"""
    manager = get_model_manager()
    all_stages = {}
    
    for model_config in AVAILABLE_MODELS.values():
        model_name = model_config["name"]
        all_stages[model_name] = manager.list_model_stages(model_name)
    
    return {"model_stages": all_stages}

@router.get("/production")
def get_production_models():
    """Get all models currently in Production stage"""
    manager = get_model_manager()
    return {"production_models": manager.get_production_models()}

@router.get("/ab_test/stats")
def get_ab_test_stats():
    """Expose aggregate A/B testing metrics."""
    manager = get_model_manager()
    return manager.get_ab_test_stats()

@router.post("/{model_name}/promote")
def promote_model(model_name: str, stage: str = "Production"):
    """Promote a model to the specified stage"""
    manager = get_model_manager()
    
    try:
        current_production = manager.get_production_model_version(model_name)
        success = manager.promote_to_production(model_name, stage)
        
        if success:
            return {
                "success": True,
                "model_name": model_name,
                "promoted_to": stage,
                "previous_production": current_production
            }
        else:
            raise HTTPException(status_code=500, detail=f"Failed to promote {model_name} to {stage}")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Promotion failed: {str(e)}")

@router.post("/{model_name}/demote")
def demote_model(model_name: str):
    """Demote a model from Production stage"""
    manager = get_model_manager()
    
    try:
        current_stage = manager.get_current_stage(model_name)
        if current_stage != "Production":
            raise HTTPException(status_code=400, detail=f"{model_name} is not in Production stage")
        
        success = manager.demote_from_production(model_name)
        
        if success:
            return {
                "success": True,
                "model_name": model_name,
                "demoted_from": "Production",
                "current_stage": "Staging"
            }
        else:
            raise HTTPException(status_code=500, detail=f"Failed to demote {model_name} from Production")
            
    except HTTPException:
        raise
    except Exception as e:
        current_stage = getattr(e, 'current_stage', 'Unknown')
        raise HTTPException(status_code=500, detail=f"Failed to demote {model_name} from {current_stage}")

@router.post("/reload/{model_type}/{alias}")
def reload_single_model(model_type: str, alias: str, sync_minio: bool = True):
    """
    Reload a single model@alias without reloading all 12 models.
    
    Args:
        model_type: xgboost, feedforward_nn, or multinomial_logistic
        alias: Production, B, shadow1, or shadow2
        sync_minio: If True, sync this specific model to MinIO first
    
    Returns:
        Success status and model info
    """
    import logging
    from app.model_serving import get_model_manager
    
    logger = logging.getLogger(__name__)
    manager = get_model_manager()
    
    # Validate inputs
    valid_types = ["xgboost", "feedforward_nn", "multinomial_logistic"]
    valid_aliases = ["Production", "B", "shadow1", "shadow2"]
    
    if model_type not in valid_types:
        raise HTTPException(status_code=400, detail=f"Invalid model_type: {model_type}. Must be one of {valid_types}")
    if alias not in valid_aliases:
        raise HTTPException(status_code=400, detail=f"Invalid alias: {alias}. Must be one of {valid_aliases}")
    
    # Step 1: Sync to MinIO (if requested)
    minio_sync_success = True
    if sync_minio:
        logger.info(f"Syncing {model_type}@{alias} to MinIO...")
        try:
            from app.minio_sync import sync_single_model_to_minio
            from mlflow.tracking import MlflowClient
            
            # Get model info from MLflow
            client = MlflowClient()
            type_to_name = {
                'xgboost': 'rps_bot_xgboost',
                'feedforward_nn': 'rps_bot_feedforward',
                'multinomial_logistic': 'rps_bot_mnlogit',
            }
            model_name = type_to_name[model_type]
            
            mv = client.get_model_version_by_alias(model_name, alias)
            run_id = mv.run_id
            version = mv.version
            
            minio_sync_success = sync_single_model_to_minio(model_type, alias, run_id, version)
            
            if minio_sync_success:
                logger.info(f"✅ MinIO sync completed for {model_type}@{alias}")
            else:
                logger.warning(f"⚠️  MinIO sync failed for {model_type}@{alias}")
                
        except Exception as e:
            logger.error(f"MinIO sync error: {e}", exc_info=True)
            minio_sync_success = False
    
    # Step 2: Clear cache for this specific model@alias
    cache_key = f"{model_type}@{alias}"
    if cache_key in manager.loaded_models:
        del manager.loaded_models[cache_key]
        logger.info(f"Cleared cache for {cache_key}")
    if cache_key in manager.model_metadata:
        del manager.model_metadata[cache_key]
    
    # Step 3: Force reload from MinIO/MLflow
    try:
        # Load model with specified alias
        model = manager.load_model_with_alias(model_type, alias, force_reload=True)
        if model is not None:
            logger.info(f"✅ Reloaded {model_type}@{alias}")
            info = manager.model_metadata.get(cache_key, {})
            return {
                "success": True,
                "minio_sync_success": minio_sync_success,
                "model_type": model_type,
                "alias": alias,
                "info": info,
                "message": f"Successfully reloaded {model_type}@{alias}"
            }
        else:
            raise HTTPException(status_code=500, detail=f"Failed to reload {model_type}@{alias}")
    except Exception as e:
        logger.error(f"Failed to reload {model_type}@{alias}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Reload failed: {str(e)}")


@router.post("/reload")
def reload_all_models(sync_minio: bool = True):
    """
    Force reload of all models from MLflow registry.
    
    When sync_minio=True (default):
    1. Syncs promoted models (Production/B/shadow1/shadow2) to MinIO
    2. Removes old models from MinIO
    3. Clears model cache
    4. Reloads all models from MinIO
    
    This ensures MinIO always has the latest promoted models.
    """
    import logging
    
    logger = logging.getLogger(__name__)
    manager = get_model_manager()
    
    # Step 1: Sync promoted models to MinIO (if requested)
    minio_sync_success = True
    synced = 0
    skipped = 0
    errors = 0
    
    if sync_minio:
        logger.info("Syncing promoted models to MinIO...")
        try:
            from app.minio_sync import sync_promoted_models_to_minio
            synced, skipped, errors = sync_promoted_models_to_minio(clean=True, force=False)
            
            if errors == 0:
                logger.info(f"MinIO sync completed: {synced} synced, {skipped} skipped")
                minio_sync_success = True
            else:
                logger.warning(f"MinIO sync had errors: {synced} synced, {skipped} skipped, {errors} errors")
                minio_sync_success = False
        except Exception as e:
            logger.error(f"MinIO sync error: {e}", exc_info=True)
            minio_sync_success = False
    
    # Step 2: Clear the loaded models cache to force fresh load
    manager.loaded_models.clear()
    manager.model_metadata.clear()
    logger.info("Cleared model cache, forcing reload from MLflow/MinIO")
    
    # Step 3: Reload all models (will load from MinIO if available)
    results = manager.load_all_models()
    
    reloaded = [model for model, success in results.items() if success]
    failed = [model for model, success in results.items() if not success]
    
    return {
        "success": len(failed) == 0 and minio_sync_success,
        "minio_sync_success": minio_sync_success,
        "minio_synced": synced,
        "minio_skipped": skipped,
        "minio_errors": errors,
        "reloaded": reloaded,
        "failed": failed,
        "message": f"Reloaded {len(reloaded)} models, {len(failed)} failures. MinIO: {synced} synced, {skipped} already present, {errors} errors"
    }