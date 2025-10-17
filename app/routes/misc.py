"""
Miscellaneous route handlers.
Handles legacy endpoints, debugging utilities, and special functions.
"""
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from app.model_serving import get_model_manager
from app.features import extract_inference_features
from app.game_utils import outcome
from app.db import connect
from app.policies import list_policies
import uuid
import random

router = APIRouter()

class LegacyGameRequest(BaseModel):
    player_name: str = "Player"
    policy: str = "brian"
    difficulty_mode: str = "normal"

class LegacyMoveRequest(BaseModel):
    player_move: str
    policy: str = "brian"
    difficulty_mode: str = "normal"
    player_history: List[str] = []
    bot_history: List[str] = []

@router.post("/game")
def legacy_start_game(request: LegacyGameRequest):
    """Legacy endpoint for starting games (deprecated, use /start_game)"""
    try:
        game_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())
        
        return {
            "game_id": game_id,
            "session_id": session_id,
            "player_name": request.player_name,
            "policy": request.policy,
            "difficulty_mode": request.difficulty_mode,
            "status": "ready",
            "message": f"Legacy game started. Consider using /start_game for new integrations.",
            "api_version": "legacy"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Legacy game start failed: {str(e)}")

@router.post("/move")
def legacy_make_move(request: LegacyMoveRequest):
    """Legacy endpoint for making moves (deprecated, use /play)"""
    try:
        valid_moves = ["rock", "paper", "scissors"]
        if request.player_move.lower() not in valid_moves:
            raise HTTPException(status_code=400, detail=f"Invalid move. Must be one of: {valid_moves}")
        
        player_move = request.player_move.lower()
        
        # Extract features and get prediction
        features = extract_inference_features(request.player_history, request.bot_history)
        manager = get_model_manager()
        
        prediction_result = manager.predict_move(
            policy=request.policy,
            difficulty_mode=request.difficulty_mode,
            features=features
        )
        
        bot_move = prediction_result["prediction"]
        result = outcome(player_move, bot_move)
        
        # Record move in database
        try:
            conn = connect()
            cursor = conn.cursor()
            
            # Create a legacy session entry
            session_id = str(uuid.uuid4())
            
            cursor.execute("""
                INSERT INTO moves (
                    game_id, player_move, bot_move, result,
                    bot_policy, difficulty_mode, model_type,
                    session_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """, (
                f"legacy_{session_id}",
                player_move,
                bot_move,
                result,
                request.policy,
                request.difficulty_mode,
                prediction_result.get("model_info", {}).get("model_type", "unknown"),
                session_id
            ))
            
            conn.commit()
            conn.close()
        except Exception:
            pass  # Don't fail the request if database write fails
        
        return {
            "player_move": player_move,
            "bot_move": bot_move,
            "result": result,
            "policy": request.policy,
            "difficulty_mode": request.difficulty_mode,
            "model_info": prediction_result.get("model_info", {}),
            "message": f"You played {player_move}, bot played {bot_move} - {result}!",
            "api_version": "legacy",
            "recommendation": "Consider migrating to /start_game and /play endpoints"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Legacy move failed: {str(e)}")

@router.get("/debug/random_game")
def debug_random_game(moves: int = 10, policy: str = "brian"):
    """Generate a random game for testing purposes"""
    try:
        valid_moves = ["rock", "paper", "scissors"]
        player_history = []
        bot_history = []
        game_log = []
        
        manager = get_model_manager()
        
        for round_num in range(moves):
            # Random player move
            player_move = random.choice(valid_moves)
            
            # Get bot prediction
            features = extract_inference_features(player_history, bot_history)
            prediction_result = manager.predict_move(
                policy=policy,
                difficulty_mode="normal",
                features=features
            )
            
            bot_move = prediction_result["prediction"]
            result = outcome(player_move, bot_move)
            
            # Update histories
            player_history.append(player_move)
            bot_history.append(bot_move)
            
            game_log.append({
                "round": round_num + 1,
                "player_move": player_move,
                "bot_move": bot_move,
                "result": result,
                "model_info": prediction_result.get("model_info", {})
            })
        
        # Calculate final scores
        player_wins = sum(1 for move in game_log if move["result"] == "player")
        bot_wins = sum(1 for move in game_log if move["result"] == "bot")
        ties = sum(1 for move in game_log if move["result"] == "tie")
        
        return {
            "game_type": "debug_random",
            "policy": policy,
            "total_moves": moves,
            "final_scores": {
                "player": player_wins,
                "bot": bot_wins,
                "ties": ties
            },
            "game_log": game_log,
            "player_history": player_history,
            "bot_history": bot_history
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Debug game failed: {str(e)}")

@router.get("/version")
def get_api_version():
    """Get API version and system information"""
    try:
        manager = get_model_manager()
        
        return {
            "api_version": "2.0",
            "system": "RPS Quest MLOps Platform",
            "description": "Kubernetes-native MLOps platform with rock-paper-scissors gameplay",
            "features": {
                "ab_testing": manager.ab_testing_enabled,
                "model_fallback_chain": True,
                "feature_count": 50,
                "policies": ["brian", "forrest", "logan"],
                "difficulty_modes": ["normal"]
            },
            "endpoints": {
                "modern": ["/start_game", "/play"],
                "legacy": ["/game", "/move"],
                "analytics": ["/leaderboard", "/games", "/stats/*"],
                "models": ["/models", "/models/*/load", "/models/production"],
                "monitoring": ["/metrics", "/health/*"]
            },
            "deployment": {
                "type": "kubernetes",
                "config_method": "configmaps",
                "mlflow_backend": "dagsHub",
                # Marker used for deployment verification during 2025-10-11 update flow
                "verification_token": "ab-split-verify-2025-10-11"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Version check failed: {str(e)}")

@router.get("/policies")
def get_policies():
    """Legacy endpoint for listing policies (redirects to /predict/policies logic)"""
    try:
        policies = list_policies()
        
        # Return just the array that the frontend expects
        return policies
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Policies listing failed: {str(e)}")

@router.get("/round_points")
def round_points(game_id: str):
    """Get round points for damage calculation"""
    try:
        from app.features import deterministic_round_points
        from app.game_utils import get_next_round_number
        
        conn = connect()
        cur = conn.cursor()
        round_no = get_next_round_number(cur, game_id)
        conn.close()
        
        return {"round_no": round_no, "round_points": deterministic_round_points(game_id, round_no)}
        
    except Exception as e:
        if 'conn' in locals():
            conn.close()
        raise HTTPException(status_code=500, detail=f"Round points calculation failed: {str(e)}")