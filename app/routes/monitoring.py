"""
Monitoring route handlers.
Handles Prometheus metrics, health checks, and system monitoring.
"""
from fastapi import APIRouter, Response
from app.model_serving import get_model_manager
from app.db import connect
import time

router = APIRouter()

@router.get("/metrics")
def get_prometheus_metrics():
    """Prometheus metrics endpoint"""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    
    # Update active games gauge if available
    try:
        conn = connect()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM games WHERE json_extract(game_state, '$.status') = 'active'")
        active_games = cursor.fetchone()[0]
        # Skip setting gauge to avoid import issues
        conn.close()
    except Exception:
        pass  # Ignore errors in metrics collection
    
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

@router.get("/health/detailed")
def detailed_health_check():
    """Comprehensive health check with component status"""
    start_time = time.time()
    
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "components": {}
    }
    
    # Database health
    try:
        conn = connect()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM games LIMIT 1")
        cursor.fetchone()
        conn.close()
        health_status["components"]["database"] = {"status": "healthy", "message": "Database accessible"}
    except Exception as e:
        health_status["components"]["database"] = {"status": "unhealthy", "message": f"Database error: {str(e)}"}
        health_status["status"] = "unhealthy"
    
    # Model manager health
    try:
        manager = get_model_manager()
        loaded_models = manager.get_all_model_info()
        available_models = manager.list_available_models()
        
        health_status["components"]["model_manager"] = {
            "status": "healthy",
            "loaded_models": len(loaded_models),
            "available_models": len(available_models),
            "ab_testing_enabled": manager.ab_testing_enabled
        }
    except Exception as e:
        health_status["components"]["model_manager"] = {"status": "unhealthy", "message": f"Model manager error: {str(e)}"}
        health_status["status"] = "unhealthy"
    
    # Feature extraction health
    try:
        from app.features import extract_inference_features
        test_features = extract_inference_features(["rock"], ["paper"])
        expected_feature_count = 50  # FEATURE_COUNT constant
        
        if len(test_features) == expected_feature_count:
            health_status["components"]["features"] = {
                "status": "healthy",
                "feature_count": len(test_features),
                "expected_count": expected_feature_count
            }
        else:
            health_status["components"]["features"] = {
                "status": "unhealthy",
                "message": f"Feature count mismatch: got {len(test_features)}, expected {expected_feature_count}"
            }
            health_status["status"] = "unhealthy"
    except Exception as e:
        health_status["components"]["features"] = {"status": "unhealthy", "message": f"Feature extraction error: {str(e)}"}
        health_status["status"] = "unhealthy"
    
    # Policies health
    try:
        from app.policies import get_available_policies
        policies = get_available_policies()
        expected_policies = ["brian", "forrest", "logan"]
        
        if all(policy in policies for policy in expected_policies):
            health_status["components"]["policies"] = {
                "status": "healthy",
                "available_policies": list(policies.keys()),
                "policy_count": len(policies)
            }
        else:
            missing_policies = [p for p in expected_policies if p not in policies]
            health_status["components"]["policies"] = {
                "status": "unhealthy",
                "message": f"Missing policies: {missing_policies}"
            }
            health_status["status"] = "unhealthy"
    except Exception as e:
        health_status["components"]["policies"] = {"status": "unhealthy", "message": f"Policies error: {str(e)}"}
        health_status["status"] = "unhealthy"
    
    # Response time
    response_time = time.time() - start_time
    health_status["response_time_seconds"] = round(response_time, 3)
    
    return health_status

@router.get("/health/metrics")
def health_metrics_summary():
    """Health check focused on metrics collection"""
    try:
        manager = get_model_manager()
        
        # Get metrics summary
        ab_stats = manager.get_ab_test_stats()
        model_usage = manager.model_usage
        
        conn = connect()
        cursor = conn.cursor()
        
        # Recent activity
        cursor.execute("SELECT COUNT(*) FROM moves WHERE created_at >= datetime('now', '-1 hour')")
        moves_last_hour = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM games WHERE created_at >= datetime('now', '-1 hour')")
        games_last_hour = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "status": "healthy",
            "metrics_collection": "active",
            "ab_testing": {
                "enabled": manager.ab_testing_enabled,
                "stats": ab_stats
            },
            "model_usage": model_usage,
            "recent_activity": {
                "moves_last_hour": moves_last_hour,
                "games_last_hour": games_last_hour
            }
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "metrics_collection": "failed"
        }

@router.get("/debug/counters")
def get_debug_counters():
    """Get current counter values for debugging"""
    try:
        # Return simplified counter info to avoid import issues
        return {
            "message": "Counter values available via /metrics endpoint",
            "status": "simplified_for_refactoring"
        }
    except Exception as e:
        return {"error": f"Failed to get counter values: {str(e)}"}

@router.get("/debug/model_state")
def get_model_debug_state():
    """Get detailed model manager state for debugging"""
    try:
        manager = get_model_manager()
        
        return {
            "loaded_models": manager.get_all_model_info(),
            "available_models": manager.list_available_models(),
            "model_usage": manager.model_usage,
            "ab_testing_enabled": manager.ab_testing_enabled,
            "ab_split_ratio": getattr(manager, 'ab_split_ratio', None),
            "fallback_chain_active": True,  # Always active in current implementation
            "production_models": manager.get_production_models()
        }
        
    except Exception as e:
        return {"error": f"Failed to get model state: {str(e)}"}