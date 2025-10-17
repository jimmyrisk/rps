# app/policies.py - ML Model-based policies with Value Optimization for RPS Quest

import os
import json
import random
import logging
from typing import Dict, List, Optional, Any, Tuple

import pandas as pd
import requests

from app.model_serving import get_model_manager
# Use the canonical legacy policies directly
from app.legacy_models import choose_legacy_bot_move, list_legacy_policies, opening_move_if_any

logger = logging.getLogger(__name__)

MOVES: List[str] = ["rock", "paper", "scissors"]

# Policy to model mapping - each policy uses a specific ML model for probabilities
# Format: policy_name -> (model_type, difficulty_mode)
POLICY_MODEL_MAP: Dict[str, Tuple[str, str]] = {
    # Standard policies
    "brian": ("feedforward_nn", "standard"),
    "forrest": ("xgboost", "standard"),
    "logan": ("multinomial_logistic", "standard"),
    
    # Easy mode policies
    "brian_easy": ("feedforward_nn", "easy"),
    "forrest_easy": ("xgboost", "easy"),
    "logan_easy": ("multinomial_logistic", "easy"),
}

# Model types (for ML model loading only)
MODEL_TYPES = ["feedforward_nn", "xgboost", "multinomial_logistic"]

ML_POLICY_FALLBACK: Dict[str, str] = {
    "brian": "bob",
    "forrest": "cal", 
    "logan": "dan",
    "brian_easy": "bob",
    "forrest_easy": "cal",
    "logan_easy": "dan",
}

def beater(move: str) -> str:
    """Return the move that beats the given move"""
    return {"rock": "paper", "paper": "scissors", "scissors": "rock"}[move]

def loser(move: str) -> str:
    """Return the move that loses to the given move"""
    return {"rock": "scissors", "paper": "rock", "scissors": "paper"}[move]

def _normalize_probs(probabilities) -> Dict[str, float]:
    """
    Accepts:
      - list/tuple like [p_rock, p_paper, p_scissors]
      - dict with 'probabilities' key
      - dict keyed by move names
    Returns a dict {rock, paper, scissors} with sanitized, normalized values.
    """
    probs_list: List[float]

    if isinstance(probabilities, dict):
        if "probabilities" in probabilities:
            vals = probabilities["probabilities"]
            if isinstance(vals, (list, tuple)):
                probs_list = list(vals)
            else:
                # nested format, try to flatten
                probs_list = [vals.get(m, 0.0) for m in MOVES]
        else:
            # direct dict keyed by move names
            probs_list = [probabilities.get(m, 0.0) for m in MOVES]
    elif isinstance(probabilities, (list, tuple)):
        probs_list = list(probabilities)
    else:
        probs_list = [1.0 / len(MOVES)] * len(MOVES)

    # Ensure exactly 3 values
    while len(probs_list) < len(MOVES):
        probs_list.append(0.0)
    probs_list = probs_list[:len(MOVES)]

    # Normalize to sum to 1
    total = sum(probs_list)
    if total <= 0:
        probs_list = [1.0 / len(MOVES)] * len(MOVES)
    else:
        probs_list = [p / total for p in probs_list]

    return dict(zip(MOVES, probs_list))


def _predict_via_http(model_name: str, features: pd.DataFrame) -> Optional[Dict[str, float]]:
    """
    Attempt to get predictions from a remote HTTP endpoint.
    """
    api_base = os.getenv("RPS_PREDICT_API_BASE")
    if not api_base:
        return None

    try:
        # Convert features to JSON-serializable format
        feature_data = features.to_dict(orient="records")[0] if not features.empty else {}
        
        response = requests.post(
            f"{api_base}/predict/{model_name}",
            json={"features": feature_data},
            timeout=5.0,
        )
        
        if response.status_code == 200:
            data = response.json()
            probs = data
        return _normalize_probs(probs)
    except Exception:
        return None

def get_model_probabilities(
    model_name: str,
    features: pd.DataFrame,
    alias: Optional[str] = None,
) -> Tuple[Dict[str, float], str]:
    """
    Get probability distribution over user's next move from specified ML model.
    
    This function ONLY gets raw probabilities from the model - no difficulty adjustments.
    Difficulty adjustments happen in the policy layer via expected value calculations.

    Preference order:
      1) Remote HTTP endpoint (if RPS_PREDICT_API_BASE is set)
      2) In-process ModelManager
    """
    manager = get_model_manager()
    alias_key = f"{model_name}@{alias}" if alias else None

    def _record_fallback_source(source: str, error: Optional[str] = None) -> None:
        if alias_key:
            manager.record_prediction_source(alias_key, source, error)
        manager.record_prediction_source(model_name, source, error)

    if os.getenv("RPS_FORCE_MODEL_FALLBACKS", "false").lower() == "true":
        _record_fallback_source("forced_fallback", "RPS_FORCE_MODEL_FALLBACKS enabled")
        return {"rock": 1 / 3, "paper": 1 / 3, "scissors": 1 / 3}, "forced_fallback"

    http_probs = None if alias else _predict_via_http(model_name, features)
    if http_probs is not None:
        manager.record_prediction_source(model_name, "http", None)
        return http_probs, "http"

    # 2) In-process model manager
    try:
        if alias:
            prediction_result = manager.predict_with_alias(model_name, alias, features)
        else:
            prediction_result = manager.predict(model_name, features)

        if prediction_result and "probabilities" in prediction_result:
            source = prediction_result.get("source", "mlflow")
            manager.record_prediction_source(model_name, source, None)
            return _normalize_probs(prediction_result["probabilities"]), source
        elif prediction_result and "probs" in prediction_result:
            source = prediction_result.get("source", "mlflow")
            manager.record_prediction_source(model_name, source, None)
            return _normalize_probs(prediction_result["probs"]), source
    except Exception as e:
        # Log the error for debugging
        print(f"Model prediction error for {model_name}: {e}")
        if alias_key:
            manager.record_prediction_source(alias_key, "mlflow_error", str(e))
        manager.record_prediction_source(model_name, "mlflow_error", str(e))

    # 3) Pattern-based fallback "personalities"
    _record_fallback_source("fallback", "heuristic policy engaged")
    return {"rock": 1 / 3, "paper": 1 / 3, "scissors": 1 / 3}, "fallback"

def expected_value(
    our_move: str,
    opponent_probs: Dict[str, float],
    round_values: Dict[str, float],
    our_score: float,
    opponent_score: float,
    target_score: float = 10.0,
) -> float:
    beaten_move = loser(our_move)
    losing_move = beater(our_move)

    p_beat = opponent_probs.get(beaten_move, 0.0)
    p_lose = opponent_probs.get(losing_move, 0.0)
    p_tie  = opponent_probs.get(our_move, 0.0)

    our_damage = round_values.get(our_move, 1.0)
    opponent_damage = round_values.get(losing_move, 1.0)

    win_bonus = 10.0 if (our_score + our_damage) >= target_score else 0.0
    lose_penalty_bonus = 10.0 if (opponent_score + opponent_damage) >= target_score else 0.0

    beat_value = p_beat * (our_damage + win_bonus)
    lose_value = p_lose * (opponent_damage + lose_penalty_bonus)

    # New tie value: 0.5 * (my points - their points) / 10, weighted by p_tie
    denom = target_score if target_score else 10.0
    tie_value = p_tie * 0.5 * ((our_score - opponent_score) / denom)
    return beat_value - lose_value + tie_value

def select_policy_for_difficulty(base_policy: str, easy_mode: bool) -> str:
    """
    Select the appropriate policy based on base policy and difficulty mode.
    
    Args:
        base_policy: Base policy name ("brian", "forrest", "logan") 
        easy_mode: Whether easy mode is enabled
        
    Returns:
        Full policy name ("brian", "brian_easy", etc.)
    """
    if not easy_mode:
        return base_policy
    
    easy_policy = f"{base_policy}_easy"
    if easy_policy in POLICY_MODEL_MAP:
        return easy_policy
    else:
        # Fallback to standard policy if easy variant doesn't exist
        return base_policy

def get_model_and_difficulty_for_policy(policy: str) -> Tuple[str, str]:
    """
    Get the model type and difficulty mode for a given policy.
    
    Returns:
        Tuple of (model_type, difficulty_mode)
    """
    if policy in POLICY_MODEL_MAP:
        return POLICY_MODEL_MAP[policy]
    else:
        # Default fallback
        return ("feedforward_nn", "standard")

def value_optimizer_with_policy(
    policy: str,
    features: pd.DataFrame,
    our_score: float = 0.0,
    opponent_score: float = 0.0,
    round_values: Dict[str, float] = None,
    alias: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Optimize move selection using model predictions and expected value calculations.
    
    This implements the correct architecture:
    1. Get raw probabilities from model (no difficulty adjustment)
    2. Apply difficulty adjustment in policy layer via expected value calculation

    Args:
        policy: Policy name including difficulty variant
        features: Feature vector for prediction
        our_score: Bot's current score
        opponent_score: User's current score
        round_values: Round value multipliers
        alias: Optional model alias (Production/B/etc.) for A/B testing
    """
    if round_values is None:
        round_values = {"rock": 1.0, "paper": 1.0, "scissors": 1.0}

    model_type, difficulty_mode = get_model_and_difficulty_for_policy(policy)
    
    # Get raw probabilities from model (same for both easy and normal modes)
    opponent_probs, probability_source = get_model_probabilities(
        model_type,
        features,
        alias=alias,
    )

    # Calculate expected values for all moves using raw probabilities
    move_values: Dict[str, float] = {
        m: expected_value(m, opponent_probs, round_values, our_score, opponent_score, target_score=10.0)
        for m in MOVES
    }

    # Apply difficulty-based move selection
    if difficulty_mode == "easy":
        # Easy mode: Weighted random selection based on EV rankings
        # Import here to avoid circular dependency
        from app.config import get_easy_mode_probabilities
        worst_prob, middle_prob, best_prob = get_easy_mode_probabilities()
        
        # Sort moves by expected value (ascending: worst to best)
        sorted_moves = sorted(move_values.items(), key=lambda x: x[1])
        worst_move = sorted_moves[0][0]
        middle_move = sorted_moves[1][0]
        best_move = sorted_moves[2][0]
        
        # Weighted random selection
        import random
        rand = random.random()
        if rand < worst_prob:
            selected_move = worst_move
        elif rand < worst_prob + middle_prob:
            selected_move = middle_move
        else:
            selected_move = best_move
    else:
        # Standard mode: Always pick best EV move (greedy)
        selected_move = max(move_values, key=move_values.get)

    return {
        "move": selected_move,
        "probabilities": opponent_probs,  # Always use raw probabilities
        "base_probabilities": opponent_probs,
        "move_values": move_values,
        "model": model_type,
        "policy": policy,
        "difficulty_mode": difficulty_mode,
        "probability_source": probability_source,
        "model_alias": alias,
        "model_key": f"{model_type}@{alias}" if alias else model_type,
    }

def _record_policy_metrics(policy: str, move: str, model_type: str, difficulty_mode: str):
    """
    Record policy-level metrics for Grafana monitoring.
    
    This enables tracking win rates separately for easy vs standard policies.
    """
    try:
        from app.metrics import record_policy_prediction
        record_policy_prediction(policy, model_type, difficulty_mode, move)
    except Exception as e:
        # Don't let metrics recording break the game
        print(f"Failed to record policy metrics: {e}")


def get_all_model_predictions(
    policy: str,
    features: pd.DataFrame,
    game_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Get predictions from all 4 aliases for the given policy's model type.
    
    This is used for tracking shadow model accuracy alongside the active model.
    
    Args:
        policy: Policy name (determines model type)
        features: Feature vector for prediction
        game_id: Game ID for A/B assignment
        session_id: Session ID for A/B assignment
        
    Returns:
        Dict mapping alias names to prediction results:
        {
            "Production": {"probabilities": {...}, "predicted_move": "rock", ...},
            "B": {"probabilities": {...}, "predicted_move": "paper", ...},
            "shadow1": {"probabilities": {...}, "predicted_move": "scissors", ...},
            "shadow2": {"probabilities": {...}, "predicted_move": "rock", ...}
        }
    """
    from app.model_serving import get_model_manager
    
    # Get model type for this policy
    model_type, _ = get_model_and_difficulty_for_policy(policy)
    manager = get_model_manager()
    
    # Determine which alias is active for this game
    active_alias = manager.select_model_alias_for_policy(model_type, game_id, session_id)
    
    all_predictions = {}
    
    # Get predictions from all 4 aliases
    for alias in ["Production", "B", "shadow1", "shadow2"]:
        try:
            # Load model with this alias
            manager.load_model_with_alias(model_type, alias)
            
            # Get prediction from this alias
            # We need to call the model directly, not through choose_bot_move
            result = manager.predict(
                model_type=model_type,
                features=features,
                alias=alias
            )
            
            if result and "probabilities" in result:
                # Determine which move would be selected
                probs = result["probabilities"]
                if isinstance(probs, dict):
                    predicted_move = max(probs, key=probs.get)
                elif isinstance(probs, (list, tuple)):
                    moves = ["rock", "paper", "scissors"]
                    predicted_move = moves[probs.index(max(probs))]
                else:
                    predicted_move = "rock"  # Fallback
                
                all_predictions[alias] = {
                    "probabilities": probs,
                    "predicted_move": predicted_move,
                    "is_active": (alias == active_alias),
                    "model_type": model_type,
                }
        except Exception as e:
            # If alias doesn't exist or fails, skip it
            logger.debug(f"Could not get prediction from {model_type}@{alias}: {e}")
            all_predictions[alias] = {
                "probabilities": None,
                "predicted_move": None,
                "is_active": (alias == active_alias),
                "model_type": model_type,
                "error": str(e)
            }
    
    return all_predictions

def _infer_round_no_if_missing(cur, game_id: Optional[str]) -> Optional[int]:
    """
    If a DB cursor is available but no round_no was provided, infer the current round
    from the events table for this game (len(events)+1). Returns None if not possible.
    """
    if not cur or not game_id:
        return None
    try:
        rows = cur.execute(
            "SELECT COUNT(1) FROM events WHERE game_id=?",
            (game_id,),
        ).fetchone()
        if rows and rows[0] is not None:
            return int(rows[0]) + 1
    except Exception:
        pass
    return None

def choose_bot_move(
    policy: str,
    features: Optional[pd.DataFrame],
    easy_mode: bool = False,
    our_score: float = 0.0,
    opponent_score: float = 0.0,
    round_values: Dict[str, float] = None,
    game_id: Optional[str] = None,
    session_id: Optional[str] = None,
    *,
    cur: Optional[Any] = None,
    round_no: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Choose bot move based on policy type and return detailed context.

    NEW ARCHITECTURE: Separate policies for each model×difficulty combination.
    
    Args:
        policy: Base policy name ("brian", "forrest", "logan", "ace", "bob", "cal", "dan")
        features: Feature vector for ML model prediction
        easy_mode: Whether easy mode is enabled (selects policy variant)
        our_score: Bot's current score
        opponent_score: User's current score
        round_values: Current round point values for each move
        game_id: Optional game identifier for A/B assignments and legacy logic
        session_id: Optional session identifier for deterministic splits
        cur: Optional database cursor for legacy policies
        round_no: Optional round number for opening sequences

    Returns:
        Dict containing move, probabilities, policy metadata, model info, and metrics data.
    """
    policy = (policy or "").lower()
    if round_values is None:
        round_values = {"rock": 1.0, "paper": 1.0, "scissors": 1.0}

    # Handle opening moves (gambits)
    if cur and game_id and round_no:
        opening_move = opening_move_if_any(cur, game_id, int(round_no))
        if opening_move:
            return {
                "move": opening_move,
                "probabilities": {m: 1 / 3 for m in MOVES},
                "move_values": {},
                "model": "gambit",
                "policy": policy,
                "model_requested": None,
                "ab_test_model": None,
                "prediction_source": "gambit",
                "probability_source": "gambit",
                "legacy_round_no": int(round_no),
                "model_info": {
                    "model_type": "gambit",
                    "model_alias": None,
                    "difficulty_mode": "easy" if easy_mode else "standard",
                    "policy": policy,
                    "probability_source": "gambit",
                },
                "ab_assignment": None,
            }

    # Handle legacy deterministic policies
    if policy in list_legacy_policies():
        if round_no is None:
            inferred = _infer_round_no_if_missing(cur, game_id)
            round_no = inferred if inferred is not None else 1

        if cur and game_id:
            try:
                legacy_move = choose_legacy_bot_move(
                    policy=policy,
                    game_id=game_id,
                    round_no=int(round_no),
                    cur=cur,
                    round_pts=round_values,
                )
            except Exception:
                legacy_move = random.choice(MOVES)
        else:
            # Lightweight fallback for legacy policies
            if policy == "ace":
                legacy_move = random.choice(MOVES)
            elif policy == "bob":
                best = max(round_values, key=round_values.get)
                beat = beater(best)
                other = next(m for m in MOVES if m not in (best, beat))
                r = random.random()
                legacy_move = best if r < 0.60 else (beat if r < 0.90 else other)
            elif policy in ["cal", "dan"]:
                best = max(round_values, key=round_values.get)
                beat = beater(best)
                other = next(m for m in MOVES if m not in (best, beat))
                r = random.random()
                legacy_move = best if r < 0.60 else (beat if r < 0.90 else other)
            else:
                legacy_move = random.choice(MOVES)

        return {
            "move": legacy_move,
            "probabilities": {m: 1 / 3 for m in MOVES},
            "move_values": {},
            "model": f"legacy:{policy}",
            "policy": policy,
            "model_requested": None,
            "ab_test_model": None,
            "prediction_source": "legacy",
            "probability_source": "legacy",
            "legacy_round_no": int(round_no) if round_no is not None else None,
            "model_info": {
                "model_type": "legacy",
                "model_alias": policy,
                "difficulty_mode": "easy" if easy_mode else "standard",
                "policy": policy,
                "probability_source": "legacy",
            },
            "ab_assignment": None,
        }

    # Handle ML-driven policies with new architecture
    effective_policy = select_policy_for_difficulty(policy, easy_mode)
    model_type, difficulty_mode = get_model_and_difficulty_for_policy(effective_policy)
    
    if features is None:
        raise ValueError("features must be provided for ML-driven policies once gambits finish")

    # Import here to avoid circular imports
    manager = get_model_manager()
    
    # Determine A/B testing alias for the base model type
    selected_alias = manager.select_model_alias_for_policy(model_type, game_id, session_id)
    
    # Get model metadata for version information
    cache_key = f"{model_type}@{selected_alias}"
    model_metadata = manager.model_metadata.get(cache_key, {})
    model_version = model_metadata.get("version", "unknown")
    model_run_id = model_metadata.get("run_id")

    # Use the value optimizer with the effective policy
    try:
        result = value_optimizer_with_policy(
            effective_policy,
            features,
            our_score,
            opponent_score,
            round_values,
            alias=selected_alias,
        )
        
        # Add A/B testing and alias information
        result.update({
            "selected_alias": selected_alias,
            "effective_policy": effective_policy,
            "base_policy": policy,
            "model_requested": model_type,
            "ab_test_model": f"{model_type}@{selected_alias}" if selected_alias != "Production" else None,
            "model_alias": result.get("model_alias", selected_alias),
            "model_key": result.get("model_key", f"{model_type}@{selected_alias}"),
            # Package model info for API response (NOW WITH VERSION!)
            "model_info": {
                "model_type": model_type,
                "model_alias": selected_alias,
                "model_version": model_version,
                "model_run_id": model_run_id[:8] if model_run_id else None,
                "difficulty_mode": difficulty_mode,
                "policy": effective_policy,
                "probability_source": result.get("probability_source", "unknown"),
            },
            # Package A/B assignment info for API response
            "ab_assignment": {
                "alias": selected_alias,
                "model_key": f"{model_type}@{selected_alias}",
                "is_variant": selected_alias != "Production",
            },
        })
        
        # Record policy-level metrics for Grafana
        _record_policy_metrics(effective_policy, result["move"], model_type, difficulty_mode)
        
        return result
        
    except Exception as e:
        # Fallback handling for failed ML predictions
        fallback_policy = ML_POLICY_FALLBACK.get(policy, "ace")
        
        if cur and game_id:
            fallback_round = int(round_no or _infer_round_no_if_missing(cur, game_id) or 1)
            legacy_move = choose_legacy_bot_move(
                policy=fallback_policy,
                game_id=game_id,
                round_no=fallback_round,
                cur=cur,
                round_pts=round_values,
            )
        else:
            legacy_move = random.choice(MOVES)

        manager.record_prediction_source(f"{model_type}@{selected_alias}", "legacy_fallback", f"using {fallback_policy}")
        
        return {
            "move": legacy_move,
            "probabilities": {m: 1/3 for m in MOVES},
            "move_values": {},
            "model": f"legacy:{fallback_policy}",
            "policy": policy,
            "effective_policy": effective_policy,
            "base_policy": policy,
            "model_requested": model_type,
            "fallback_policy": fallback_policy,
            "prediction_source": "legacy_fallback",
            "probability_source": "legacy_fallback",
            "error": str(e),
            "model_info": {
                "model_type": model_type,
                "model_alias": selected_alias,
                "model_version": model_version,
                "model_run_id": model_run_id[:8] if model_run_id else None,
                "difficulty_mode": "easy" if easy_mode else "standard",
                "policy": effective_policy,
                "probability_source": "legacy_fallback",
                "error": str(e),
            },
            "ab_assignment": {
                "alias": selected_alias,
                "model_key": f"{model_type}@{selected_alias}",
                "is_variant": selected_alias != "Production",
                "fallback": True,
            },
        }


def list_policies() -> List[str]:
    """Return list of user-facing opponents (base policies only)"""
    # Only return base policies, not the _easy variants
    base_policies = ["brian", "forrest", "logan"]
    return base_policies

def list_all_policies() -> List[str]:
    """Return list of ALL available policies (ML + legacy, for internal use only)"""
    ml_policies = list(POLICY_MODEL_MAP.keys())
    legacy_policies = list_legacy_policies()
    return ml_policies + legacy_policies

def get_policy_display_names() -> Dict[str, str]:
    """Return mapping of user-facing opponent names to display names"""
    return {
        # ML Models (user-facing)
        "brian": "Brian (Neural Network)",
        "forrest": "Forrest (XGBoost)",
        "logan": "Logan (Logistic Regression)",
        # Legacy models are internal-only and not included here
    }