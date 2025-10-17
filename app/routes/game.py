"""
Game flow route handlers.
Handles the modern API pattern for game state management.
"""
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging
import uuid
import json
from datetime import datetime, timezone
from app.model_serving import get_model_manager
from app.db import connect
from app.game_utils import outcome
from app.features import extract_inference_features
from app.features import deterministic_round_points
from app.gambits import pick_opening
from app.policies import choose_bot_move, get_model_and_difficulty_for_policy, select_policy_for_difficulty

logger = logging.getLogger(__name__)

router = APIRouter()


def _generate_default_player_name(session_id: Optional[str]) -> str:
    """Generate a placeholder display name using the session id."""
    if not session_id:
        return "Player"

    clean = "".join(ch for ch in session_id if ch.isalnum())
    code = (clean[:3] or session_id[:3]).upper()
    return f"Player-{code}" if code else "Player"


class StartGameRequest(BaseModel):
    player_name: Optional[str] = "Player"
    policy: str = "brian"
    difficulty_mode: str = "normal"
    session_id: Optional[str] = None
    mark_test_game: Optional[bool] = False
    is_test_game: Optional[bool] = None  # legacy alias used by some scripts
    is_test_data: Optional[bool] = None  # legacy alias used by some scripts
    is_test: Optional[bool] = None       # legacy alias used by some scripts

class PlayMoveRequest(BaseModel):
    game_id: str
    user_move: str  # Changed from player_move to match frontend
    easy_mode: Optional[bool] = None  # Frontend sends this but we don't need it (it's in game state)

@router.post("/start_game")
def start_game(request: StartGameRequest, http_request: Request):
    """Start a new game session with specified policy and difficulty"""
    try:
        # Generate game and session IDs
        game_id = str(uuid.uuid4())
        session_id = request.session_id or str(uuid.uuid4())

        # Resolve display name
        raw_player_name = (request.player_name or "").strip()
        player_name = raw_player_name if raw_player_name and raw_player_name.lower() != "player" else _generate_default_player_name(session_id)

        # Generate opening gambit sequence
        gambit_name, gambit_moves = pick_opening(game_id)
        gambit_seq_json = json.dumps(gambit_moves)
        
        requested_difficulty = (request.difficulty_mode or "normal").strip().lower()
        easy_requested = requested_difficulty == "easy"
        if easy_requested:
            logger.info(
                "Easy difficulty requested for session %s; coercing to normal mode",
                session_id,
            )

        effective_difficulty_mode = "normal"

        # Get model manager for A/B assignment
        manager = get_model_manager()

        # Determine model type and A/B alias for this game
        effective_policy = select_policy_for_difficulty(request.policy, False)
        model_type, _ = get_model_and_difficulty_for_policy(effective_policy)
        selected_alias = manager.select_model_alias_for_policy(model_type, game_id, session_id)
        
        # Eager load all 4 model aliases from MinIO for fast inference during gameplay
        # This prevents slow first-round delays when models load lazily
        logger.info(f"Preloading all {model_type} aliases for game {game_id[:8]}...")
        for alias in ["Production", "B", "shadow1", "shadow2"]:
            if not manager.is_model_loaded(f"{model_type}@{alias}"):
                logger.info(f"  Loading {model_type}@{alias} from MinIO...")
                manager.load_model_with_alias(model_type, alias)
            else:
                logger.debug(f"  {model_type}@{alias} already loaded (cached)")
        logger.info(f"All {model_type} aliases ready for game")
        
        # Initialize game state in database
        conn = connect()
        cursor = conn.cursor()

        # Get current UTC timestamp
        utc_now = datetime.now(timezone.utc).isoformat()

        mark_as_test = any(
            flag is True
            for flag in (
                request.mark_test_game,
                request.is_test_game,
                request.is_test_data,
                request.is_test,
            )
        )

        cursor.execute("""
            INSERT INTO games (
                id, session_id, bot_policy, bot_model_version,
                user_score, bot_score,
                created_ts, easy_mode, created_ts_utc, bot_model_alias,
                is_test, opening_seq, opening_name,
                rock_pts, paper_pts, scissors_pts,
                player_name
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            game_id,
            session_id,
            request.policy,
            "mlflow",  # model version placeholder
            0.0,       # user_score
            0.0,       # bot_score
            utc_now,   # created_ts (use same timestamp as created_ts_utc)
            0,  # easy_mode flag is deprecated and always disabled
            utc_now,   # created_ts_utc
            selected_alias,  # bot_model_alias (Production, B, etc.)
            1 if mark_as_test else 0,  # is_test (0 = real game, 1 = test game)
            gambit_seq_json,  # opening_seq (gambit moves for rounds 1-3)
            gambit_name,      # opening_name (gambit strategy name)
            1.0, 1.0, 1.0,    # rock/paper/scissors_pts - legacy columns, default to 1.0
            player_name  # player_name for leaderboards
        ))
        
        conn.commit()
        conn.close()
        
        return {
            "game_id": game_id,
            "session_id": session_id,
            "player_name": player_name,
            "bot_policy": request.policy,  # Changed from "policy" to "bot_policy" for frontend compatibility
            "policy": request.policy,      # Keep both for backward compatibility
            "difficulty_mode": effective_difficulty_mode,
            "bot_model_alias": selected_alias,  # Return which model variant is assigned
            "model_type": model_type,  # Return model type for reference
            "status": "active",
            "message": (
                f"Game started against {request.policy} bot in {effective_difficulty_mode} mode"
                if not easy_requested
                else f"Game started against {request.policy} bot; easy mode is retired and was coerced to normal"
            ),
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start game: {str(e)}")

@router.post("/play")
def play_move(request: PlayMoveRequest):
    """Play a move in an existing game"""
    try:
        # Validate move
        valid_moves = ["rock", "paper", "scissors"]
        if request.user_move.lower() not in valid_moves:
            raise HTTPException(status_code=400, detail=f"Invalid move. Must be one of: {valid_moves}")
        
        player_move = request.user_move.lower()
        
        # Get game state from database
        conn = connect()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT session_id, bot_policy, easy_mode, user_score, bot_score
            FROM games WHERE id = ?
        """, (request.game_id,))
        
        game_data = cursor.fetchone()
        if not game_data:
            raise HTTPException(status_code=404, detail="Game not found")
        
        session_id, policy, easy_mode, user_score, bot_score = game_data
        difficulty_mode = "easy" if easy_mode else "normal"
        
        # Get move history from events table
        cursor.execute("""
            SELECT user_move, bot_move FROM events 
            WHERE game_id = ? ORDER BY step_no
        """, (request.game_id,))
        
        history_rows = cursor.fetchall()
        player_history = [row[0] for row in history_rows]
        bot_history = [row[1] for row in history_rows]
        
        # Get next step number
        step_no = len(history_rows) + 1
        
        # Get round points for this round (MUST be calculated before bot move selection!)
        round_points = deterministic_round_points(request.game_id, step_no)
        
        # Extract features for current round
        features_df = extract_inference_features(cursor, request.game_id, step_no)
        manager = get_model_manager()
        
        # Choose bot move using policy system (with actual round point values)
        bot_result = choose_bot_move(
            policy=policy,
            features=features_df if len(features_df) > 0 else None,
            easy_mode=(difficulty_mode == "easy"),
            our_score=bot_score,
            opponent_score=user_score,
            round_values=round_points,  # Pass actual point values for EV calculation!
            game_id=request.game_id,
            session_id=session_id,
            cur=cursor,
            round_no=step_no
        )
        
        bot_move = bot_result["move"]
        
        # ======================================================================
        # METRICS: Record prediction correctness for ALL model aliases
        # ======================================================================
        # For each user action, all 4 aliases (Production, B, shadow1, shadow2) should
        # make predictions for testing purposes. Only the active alias (Production or B)
        # determines the actual bot move and can win/lose the game. Shadow models just
        # track prediction accuracy for potential promotion decisions.
        try:
            # Only record accuracy after gambit phase (first 3 moves use predetermined sequences)
            if step_no > 3 and len(features_df) > 0:
                from app.metrics import record_model_alias_prediction
                from app.policies import get_model_and_difficulty_for_policy
                
                # Get the actual alias used for this game (for bot action)
                cursor.execute("SELECT bot_model_alias FROM games WHERE id = ?", (request.game_id,))
                alias_row = cursor.fetchone()
                active_alias = alias_row[0] if alias_row else "Production"
                
                # Get model type for this policy
                model_type, _ = get_model_and_difficulty_for_policy(policy)
                
                # Predict with ALL 4 aliases to track accuracy
                all_aliases = ["Production", "B", "shadow1", "shadow2"]
                for alias in all_aliases:
                    try:
                        # Get prediction from this alias
                        alias_prediction = manager.predict_with_alias(model_type, alias, features_df)
                        
                        if alias_prediction and "probabilities" in alias_prediction:
                            # Get the predicted move (already computed by the model adapter)
                            predicted_move = alias_prediction.get("pick")
                            
                            if predicted_move:
                                # Check if this alias predicted the user's move correctly
                                is_correct = (predicted_move == player_move)
                                
                                # Record prediction correctness for this alias
                                record_model_alias_prediction(
                                    model=model_type,
                                    alias=alias,
                                    correct=is_correct
                                )
                    except Exception as e:
                        logger.warning(f"Failed to get prediction from {model_type}@{alias}: {e}")
        except Exception as e:
            logger.warning(f"Failed to record model alias prediction metrics: {e}")
        
        # Determine winner (returns "win", "lose", or "draw" from user's perspective)
        result = outcome(player_move, bot_move)
        
        # ======================================================================
        # METRICS: Record policy-level action results for Prometheus/Grafana
        # ======================================================================
        try:
            from app.metrics import record_policy_action_result
            from app.policies import get_model_and_difficulty_for_policy
            
            model_type, _ = get_model_and_difficulty_for_policy(policy)
            
            # Convert result from user perspective to bot perspective for bot metrics
            if result == "win":
                bot_result_label = "loss"
            elif result == "lose":
                bot_result_label = "win"
            else:
                bot_result_label = "tie"
            
            logger.info(f"Recording action result: policy={policy}, model={model_type}, difficulty={difficulty_mode}, result={bot_result_label}")
            record_policy_action_result(
                policy=policy,
                model_type=model_type,
                difficulty_mode=difficulty_mode,
                result=bot_result_label
            )
        except Exception as e:
            logger.warning(f"Failed to record policy action result metrics: {e}")
        
        # Calculate score deltas based on round outcome and actual point values
        if result == "win":
            user_delta = round_points.get(player_move, 1.0)
            bot_delta = 0.0
        elif result == "lose":
            user_delta = 0.0
            bot_delta = round_points.get(bot_move, 1.0)
        else:  # draw
            tie_penalty = 0.5
            user_delta = tie_penalty
            bot_delta = tie_penalty
        
        # Update scores
        new_user_score = user_score + user_delta
        new_bot_score = bot_score + bot_delta

        # Check win condition (game ends at 10 points)
        TARGET_SCORE = 10.0
        EPSILON = 1e-6

        def reached_target(score: float) -> bool:
            return score >= TARGET_SCORE or (TARGET_SCORE - score) <= EPSILON

        user_reached = reached_target(new_user_score)
        bot_reached = reached_target(new_bot_score)

        if user_reached and new_user_score < TARGET_SCORE:
            new_user_score = TARGET_SCORE
        if bot_reached and new_bot_score < TARGET_SCORE:
            new_bot_score = TARGET_SCORE

        game_finished = user_reached or bot_reached
        winner = None
        if game_finished:
            if user_reached and bot_reached:
                # Both hit 10 on same round - higher score wins, or tie
                if new_user_score > new_bot_score:
                    winner = "user"
                elif new_bot_score > new_user_score:
                    winner = "bot"
                else:
                    winner = "tie"
            elif user_reached:
                winner = "user"
            else:
                winner = "bot"
        
        # Get lagged round points for feature extraction
        # Fetch the previous 3 rounds' points
        cursor.execute("""
            SELECT round_rock_pts, round_paper_pts, round_scissors_pts
            FROM events 
            WHERE game_id = ? 
            ORDER BY step_no DESC 
            LIMIT 3
        """, (request.game_id,))
        
        lag_rows = cursor.fetchall()
        lag1 = lag_rows[0] if len(lag_rows) > 0 else (1.0, 1.0, 1.0)
        lag2 = lag_rows[1] if len(lag_rows) > 1 else (1.0, 1.0, 1.0)
        lag3 = lag_rows[2] if len(lag_rows) > 2 else (1.0, 1.0, 1.0)
        
        # Insert event record WITH round points for lag-3 feature extraction
        cursor.execute("""
            INSERT INTO events (
                game_id, ts, user_move, bot_move, result,
                user_delta, bot_delta, user_score, bot_score, step_no,
                created_ts_utc,
                round_rock_pts, round_paper_pts, round_scissors_pts,
                lag1_rock_pts, lag1_paper_pts, lag1_scissors_pts,
                lag2_rock_pts, lag2_paper_pts, lag2_scissors_pts,
                lag3_rock_pts, lag3_paper_pts, lag3_scissors_pts
            ) VALUES (?, datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            request.game_id, player_move, bot_move, result,
            user_delta, bot_delta, new_user_score, new_bot_score, step_no,
            datetime.now(timezone.utc).isoformat(),  # UTC timestamp
            round_points["rock"], round_points["paper"], round_points["scissors"],  # Current round points
            lag1[0], lag1[1], lag1[2],  # Lag 1 points
            lag2[0], lag2[1], lag2[2],  # Lag 2 points
            lag3[0], lag3[1], lag3[2],  # Lag 3 points
        ))
        
        # Update game scores
        cursor.execute("""
            UPDATE games SET user_score = ?, bot_score = ? WHERE id = ?
        """, (new_user_score, new_bot_score, request.game_id))
        
        cursor.execute(
            """
            SELECT COUNT(*) FROM events
            WHERE game_id = ? AND result IN ('draw', 'tie')
            """,
            (request.game_id,)
        )
        tie_row = cursor.fetchone()
        tie_count = tie_row[0] if tie_row else 0

        # Mark game as finished if someone won
        if game_finished:
            cursor.execute("""
                UPDATE games SET finished_ts = datetime('now'), winner = ? WHERE id = ?
            """, (winner, request.game_id))
            
            # ======================================================================
            # METRICS: Record game-level result for the ACTIVE model only
            # ======================================================================
            # METRICS: Record game-level result for policy tracking
            # ======================================================================
            try:
                from app.model_serving import record_game_result
                from app.policies import get_model_and_difficulty_for_policy
                from app.metrics import record_policy_game_result, record_game_completion
                
                # Get which model alias was actually used for this game
                cursor.execute("""
                    SELECT bot_model_alias FROM games WHERE id = ?
                """, (request.game_id,))
                alias_row = cursor.fetchone()
                active_alias = alias_row[0] if alias_row else "Production"
                
                # Get model type for this policy
                model_type, _ = get_model_and_difficulty_for_policy(policy)
                
                # Record game completion (always increment total games counter)
                record_game_completion(
                    policy=policy,
                    model_type=model_type,
                    difficulty_mode=difficulty_mode
                )
                
                # Record game result (only for active model, not shadows)
                bot_won = (winner == "bot")
                record_game_result(
                    model_type=model_type,
                    alias=active_alias,
                    bot_won=bot_won,
                    policy=policy,
                    difficulty_mode=difficulty_mode
                )
                
                # Record policy-level game result for Prometheus/Grafana
                game_result = "win" if bot_won else "loss"
                record_policy_game_result(
                    policy=policy,
                    model_type=model_type,
                    difficulty_mode=difficulty_mode,
                    result=game_result
                )
            except Exception as e:
                logger.warning(f"Failed to record game metrics: {e}")
        
        conn.commit()
        conn.close()
        
        return {
            "game_id": request.game_id,
            "round": step_no,
            "user_move": player_move,  # Frontend expects user_move
            "player_move": player_move,  # Keep for backward compatibility
            "bot_move": bot_move,
            "result": result,
            "user_delta": user_delta,  # Frontend needs this for damage display
            "bot_delta": bot_delta,    # Frontend needs this for damage display
            "user_score": new_user_score,  # Frontend expects flat fields
            "bot_score": new_bot_score,    # Frontend expects flat fields
            "round_points": round_points,  # Frontend needs this for damage palette
            "finished": game_finished,  # Game over flag
            "winner": winner,  # "user", "bot", "tie", or None
            "scores": {
                "player": new_user_score,
                "bot": new_bot_score,
                "ties": tie_count
            },
            "policy": policy,
            "difficulty_mode": difficulty_mode,
            "model_info": bot_result.get("model_info", {}),
            "ab_assignment": bot_result.get("ab_assignment"),
            # Debug information (full prediction details)
            "debug": {
                "probabilities": bot_result.get("probabilities", {}),
                "base_probabilities": bot_result.get("base_probabilities", {}),
                "move_values": bot_result.get("move_values", {}),
                "expected_values": bot_result.get("expected_values", {}),
                "probability_source": bot_result.get("probability_source", "unknown"),
                "model_type": bot_result.get("model", "unknown"),
                "selected_move": bot_move,
                "effective_policy": bot_result.get("effective_policy", policy),
                # Include feature vector for validation/inspection
                "features": features_df.iloc[0].values.tolist() if features_df is not None and len(features_df) > 0 else None,
                "feature_count": len(features_df.iloc[0]) if features_df is not None and len(features_df) > 0 else 0,
                "round_no": step_no,
                "model_info": bot_result.get("model_info", {})
            },
            "message": f"You played {player_move}, bot played {bot_move} - {result}!"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to play move: {str(e)}")