"""
Prediction route handlers.
Handles immediate predictions without game state tracking.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from app.model_serving import get_model_manager
from app.features import extract_features_stateless
from app.policies import list_policies

router = APIRouter(prefix="/predict")

class PredictionRequest(BaseModel):
    policy: str
    difficulty_mode: str = "normal"
    player_history: List[str] = []
    bot_history: List[str] = []
    result_history: List[str] = []
    round_values: Optional[Dict[str, float]] = None
    session_id: Optional[str] = None
    ab_assignment: Optional[str] = None

class GameContextRequest(BaseModel):
    player_move: str
    opponent_history: List[str] = []
    context: Dict[str, Any] = {}

@router.post("")
def make_prediction(request: PredictionRequest):
    """Make a move prediction for a given policy and game context"""
    from app.policies import get_model_and_difficulty_for_policy, choose_bot_move
    
    manager = get_model_manager()
    
    # Extract features for prediction (stateless)
    features_df = extract_features_stateless(
        user_moves=request.player_history, 
        bot_moves=request.bot_history,
        results=request.result_history if request.result_history else None,
        round_values=request.round_values,
        difficulty_mode=request.difficulty_mode
    )
    
    try:
        # Get model type and difficulty for the policy
        model_type, _ = get_model_and_difficulty_for_policy(request.policy)
        
        # Determine alias (use provided or select based on A/B testing)
        if request.ab_assignment:
            alias = request.ab_assignment
        else:
            # Default to Production for stateless predictions
            alias = "Production"
        
        # Get prediction from model
        prediction = manager.predict_with_alias(model_type, alias, features_df)
        
        if not prediction:
            raise HTTPException(status_code=500, detail="Model prediction failed")
        
        # Get predicted move (already in the prediction)
        predicted_move = prediction.get("pick", "rock")
        
        # Convert probabilities list to dict
        probs_list = prediction.get("probabilities", [0.33, 0.33, 0.34])
        probs_dict = dict(zip(["rock", "paper", "scissors"], probs_list))
        
        return {
            "prediction": predicted_move,
            "probabilities": probs_dict,
            "policy": request.policy,
            "difficulty_mode": request.difficulty_mode,
            "model_info": {
                "model_type": model_type,
                "alias": alias,
                "version": prediction.get("version"),
            },
            "features_used": len(features_df.columns)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/batch")
def batch_predictions(requests: List[PredictionRequest]):
    """Make predictions for multiple contexts"""
    from app.policies import get_model_and_difficulty_for_policy
    
    manager = get_model_manager()
    results = []
    
    for req in requests:
        try:
            features_df = extract_features_stateless(
                user_moves=req.player_history,
                bot_moves=req.bot_history,
                difficulty_mode=req.difficulty_mode
            )
            
            model_type, _ = get_model_and_difficulty_for_policy(req.policy)
            alias = req.ab_assignment if req.ab_assignment else "Production"
            
            prediction = manager.predict_with_alias(model_type, alias, features_df)
            
            if not prediction:
                raise Exception("Model prediction failed")
            
            predicted_move = prediction.get("pick", "rock")
            probs_list = prediction.get("probabilities", [0.33, 0.33, 0.34])
            probs_dict = dict(zip(["rock", "paper", "scissors"], probs_list))
            
            results.append({
                "success": True,
                "prediction": predicted_move,
                "probabilities": probs_dict,
                "policy": req.policy,
                "difficulty_mode": req.difficulty_mode,
                "model_info": {
                    "model_type": model_type,
                    "alias": alias,
                    "version": prediction.get("version"),
                }
            })
            
        except Exception as e:
            results.append({
                "success": False,
                "policy": req.policy,
                "error": str(e)
            })
    
    return {"results": results}

@router.post("/context")
def predict_from_context(request: GameContextRequest):
    """Predict next move based on game context"""
    from app.policies import get_model_and_difficulty_for_policy
    
    manager = get_model_manager()
    
    # Convert context to features if needed
    player_history = request.context.get("player_history", [])
    bot_history = request.context.get("bot_history", [])
    difficulty_mode = request.context.get("difficulty_mode", "normal")
    
    features_df = extract_features_stateless(
        user_moves=player_history,
        bot_moves=bot_history,
        difficulty_mode=difficulty_mode
    )
    
    try:
        # Use default policy if not specified in context
        policy = request.context.get("policy", "brian")
        
        model_type, _ = get_model_and_difficulty_for_policy(policy)
        alias = "Production"
        
        prediction = manager.predict_with_alias(model_type, alias, features_df)
        
        if not prediction:
            raise Exception("Model prediction failed")
        
        predicted_move = prediction.get("pick", "rock")
        probs_list = prediction.get("probabilities", [0.33, 0.33, 0.34])
        probs_dict = dict(zip(["rock", "paper", "scissors"], probs_list))
        
        return {
            "prediction": predicted_move,
            "probabilities": probs_dict,
            "context_policy": policy,
            "context_difficulty": difficulty_mode,
            "model_info": {
                "model_type": model_type,
                "alias": alias,
                "version": prediction.get("version"),
            },
            "player_move": request.player_move,
            "features_extracted": len(features_df.columns)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Context prediction failed: {str(e)}")

@router.get("/policies")
def list_prediction_policies():
    """List available policies for predictions"""
    policies = list_policies()
    
    return {
        "policies": policies,
        "difficulty_modes": ["normal"],
        "description": "Available policies for predictions (easy mode retired)"
    }

@router.get("/features/example")
def get_feature_example():
    """Get an example of extracted features for debugging"""
    # Example game history
    player_history = ["rock", "paper", "scissors", "rock"]
    bot_history = ["scissors", "rock", "paper", "scissors"]
    
    features_df = extract_features_stateless(
        user_moves=player_history,
        bot_moves=bot_history,
        difficulty_mode="normal"
    )
    features = features_df.values[0]
    
    return {
        "example_player_history": player_history,
        "example_bot_history": bot_history,
        "extracted_features": features.tolist() if hasattr(features, 'tolist') else list(features),
        "feature_count": len(features),
        "description": "Example feature extraction for prediction debugging"
    }