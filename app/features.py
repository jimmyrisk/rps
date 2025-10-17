# app/features.py - Unified Feature Engineering for RPS MLOps Pipeline
"""
Consolidates all feature engineering logic for both training and inference.
Eliminates duplication and ensures consistency across the MLOps pipeline.
"""
import pandas as pd
import numpy as np
import hashlib
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Tuple, Optional

# Constants
MOVES = ["rock", "paper", "scissors"]
RESULTS = ["win", "lose", "draw"]
FEATURE_COUNT = 50

# For round point generation
TENTHS = [Decimal(x).quantize(Decimal("0.1")) for x in ["1.1","1.2","1.3","1.4","1.5"]]

# Utility functions
def _beats(x: str) -> str:
    """What move beats x"""
    return {"rock":"paper", "paper":"scissors", "scissors":"rock"}[x]

def _beaten_by(x: str) -> str:
    """What move x beats"""
    return {"rock":"scissors", "paper":"rock", "scissors":"paper"}[x]

def _fav_from_points(r: float, p: float, s: float) -> str:
    """Determine favored move from point values (tie-breaker: rock > paper > scissors)"""
    vals = {"rock": float(r or 0.0), "paper": float(p or 0.0), "scissors": float(s or 0.0)}
    return max(MOVES, key=lambda m: (vals[m], {"rock":2,"paper":1,"scissors":0}[m]))

def _q01(x: Decimal) -> float:
    """Quantize to 0.1 and return float (no precision jitter)"""
    return float(x.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))

def deterministic_round_points(game_id: str, round_no: int) -> Dict[str, float]:
    """
    Generate deterministic point values for a round.
    Moved from main.py to consolidate game logic.
    """
    # 1) Deterministically pick which move is the low one (gets 1.0)
    seed_low = f"{game_id}:{round_no}:low".encode()
    n_low = int.from_bytes(hashlib.sha256(seed_low).digest()[:2], "big")
    low_move = MOVES[n_low % 3]

    # 2) Deterministically pick the middle value base ∈ {1.1, 1.2, 1.3, 1.4, 1.5}
    seed_base = f"{game_id}:{round_no}:base".encode()
    n_base = int.from_bytes(hashlib.sha256(seed_base).digest()[:2], "big")
    base = TENTHS[n_base % len(TENTHS)]  # Decimal
    best = (base + Decimal("0.5")).quantize(Decimal("0.1"))

    # 3) Assign values per rules:
    #    - low_move -> 1.0
    #    - the move that low_move beats -> BEST (base + 0.5)
    #    - the remaining move -> base
    prey = _beaten_by(low_move)  # the one low_move beats
    mid_move = next(m for m in MOVES if m not in (low_move, prey))

    return {
        low_move: 1.0,
        mid_move: _q01(base),
        prey: _q01(best),
    }

def get_feature_columns() -> List[str]:
    """Return the canonical 50-feature column order"""
    return [
        # User move history (9 features)
        "user_-1_rock", "user_-1_paper", "user_-1_scissors",
        "user_-2_rock", "user_-2_paper", "user_-2_scissors", 
        "user_-3_rock", "user_-3_paper", "user_-3_scissors",
        # Bot move history (9 features)
        "bot_-1_rock", "bot_-1_paper", "bot_-1_scissors",
        "bot_-2_rock", "bot_-2_paper", "bot_-2_scissors",
        "bot_-3_rock", "bot_-3_paper", "bot_-3_scissors",
        # Result history (9 features)
        "res_-1_win", "res_-1_lose", "res_-1_draw",
        "res_-2_win", "res_-2_lose", "res_-2_draw", 
        "res_-3_win", "res_-3_lose", "res_-3_draw",
        # Current round points (3 features)
        "rock_pts", "paper_pts", "scissors_pts",
        # Lagged points (9 features)
        "points_-1_rock", "points_-1_paper", "points_-1_scissors",
        "points_-2_rock", "points_-2_paper", "points_-2_scissors",
        "points_-3_rock", "points_-3_paper", "points_-3_scissors", 
        # User tendencies (3 features)
        "user_tend_rock", "user_tend_paper", "user_tend_scissors",
        # Favored tendencies (3 features) 
        "tend_favored", "tend_fav_beater", "tend_fav_beater_beater",
        # Score context (4 features)
        "score_diff", "user_pts_to_win", "bot_pts_to_win", "step_no",
        # Easy mode (1 feature)
    "easy_mode"  # Deprecated: kept for legacy compatibility, always 0 in new games
    ]

# ================================================================================
# TRAINING PIPELINE - For batch model training
# ================================================================================

def build_training_dataset(events: pd.DataFrame, games: pd.DataFrame = None,
                          lookback: int = 3, min_label_step: int = 4, 
                          target_score: float = 10.0) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build training dataset from historical events.
    
    Returns:
        X: Features DataFrame (49 columns)
        y: Target Series (user moves to predict)
    """
    if events is None or len(events) == 0:
        return pd.DataFrame(columns=get_feature_columns()), pd.Series(dtype="category")

    # Validate required columns
    base_required = {"id","game_id","user_move","bot_move","result","user_score","bot_score"}
    missing = [c for c in base_required if c not in events.columns]
    if missing:
        raise ValueError(f"events missing columns: {missing}")

    if events.columns.duplicated().any():
        events = events.loc[:, ~events.columns.duplicated()]

    sort_priority = ["created_ts_utc", "game_id", "step_no", "id"]
    sort_columns = [col for col in sort_priority if col in events.columns]

    if sort_columns:
        ev = events.sort_values(sort_columns).reset_index(drop=True)
    else:
        ev = events.reset_index(drop=True)

    # Ensure/repair step_no
    if "step_no" not in ev.columns:
        ev["step_no"] = ev.groupby("game_id").cumcount() + 1
    else:
        ev["step_no"] = ev["step_no"].fillna(ev.groupby("game_id").cumcount() + 1).astype(int)

    # Fill optional numeric columns with 0.0
    numeric_optional = [
        "round_rock_pts","round_paper_pts","round_scissors_pts",
        *[f"lag{k}_{m}_pts" for k in (1,2,3) for m in MOVES],
    ]
    for col in numeric_optional:
        if col not in ev.columns:
            ev[col] = 0.0
    ev[numeric_optional] = ev[numeric_optional].fillna(0.0).astype(float)

    rows, labels = [], []

    for game_id, game_df in ev.groupby("game_id", sort=False):
        game_df = game_df.copy()
        user_move_counts = {m: 0 for m in MOVES}
        history = []

        for _, row in game_df.iterrows():
            step = int(row["step_no"])

            if step >= min_label_step and len(history) > 0:
                # Extract features for this prediction point
                features = _extract_features_from_history(
                    history, user_move_counts, row, game_id, step, target_score, lookback
                )
                rows.append(features)
                labels.append(row["user_move"])

            # Add current row to history for next iteration
            history.append({
                "user_move": row["user_move"],
                "bot_move": row["bot_move"], 
                "result": row["result"],
                "user_score": float(row["user_score"]),
                "bot_score": float(row["bot_score"]),
                "round_pts": {
                    "rock": float(row.get("round_rock_pts", 0.0)),
                    "paper": float(row.get("round_paper_pts", 0.0)),
                    "scissors": float(row.get("round_scissors_pts", 0.0)),
                },
            })
            
            if row["user_move"] in user_move_counts:
                user_move_counts[row["user_move"]] += 1

    # Convert to DataFrame with proper column order
    X = pd.DataFrame(rows, columns=get_feature_columns()).fillna(0.0)
    for col in X.columns:
        X[col] = X[col].astype(float)
    
    y = pd.Series(labels, dtype="category")
    return X, y

# ================================================================================
# INFERENCE PIPELINE - For real-time predictions
# ================================================================================

def extract_inference_features(cursor, game_id: str, round_no: int, 
                             target_score: float = 10.0) -> pd.DataFrame:
    """
    Extract features for real-time inference from database.
    
    Args:
        cursor: Database cursor
        game_id: Current game ID
        round_no: Current round number (1-based)
        target_score: Points needed to win
        
    Returns:
        Single-row DataFrame with 50 features matching training schema
    """
    # Get historical game events for this game BEFORE the current round
    # CRITICAL: Only use events that happened BEFORE round_no to avoid data leakage
    events = cursor.execute("""
        SELECT user_move, bot_move, result, step_no, user_score, bot_score,
               round_rock_pts, round_paper_pts, round_scissors_pts,
               lag1_rock_pts, lag1_paper_pts, lag1_scissors_pts,
               lag2_rock_pts, lag2_paper_pts, lag2_scissors_pts,
               lag3_rock_pts, lag3_paper_pts, lag3_scissors_pts
        FROM events 
        WHERE game_id = ? AND step_no < ?
        ORDER BY step_no ASC
    """, (game_id, round_no)).fetchall()
    
    # Get easy_mode from games table (deprecated flag retained for legacy data parity)
    game_row = cursor.execute("""
        SELECT easy_mode FROM games WHERE id = ?
    """, (game_id,)).fetchone()
    easy_mode = int(game_row[0]) if game_row and game_row[0] is not None else 0
    
    # Build history for feature extraction
    history = []
    user_move_counts = {"rock": 0, "paper": 0, "scissors": 0}
    
    for event in events:
        user_move, bot_move, result, step_no, user_score, bot_score = event[:6]
        round_pts = event[6:9]  # rock, paper, scissors pts for that round
        
        history.append({
            "user_move": user_move,
            "bot_move": bot_move,
            "result": result,
            "user_score": float(user_score or 0),
            "bot_score": float(bot_score or 0),
            "round_pts": {
                "rock": float(round_pts[0] or 0.0),
                "paper": float(round_pts[1] or 0.0), 
                "scissors": float(round_pts[2] or 0.0)
            }
        })
        
        if user_move in user_move_counts:
            user_move_counts[user_move] += 1

    # Create a synthetic "current round" for feature extraction
    current_round_pts = deterministic_round_points(game_id, round_no)
    current_row = {
        "round_rock_pts": current_round_pts["rock"],
        "round_paper_pts": current_round_pts["paper"],
        "round_scissors_pts": current_round_pts["scissors"],
    "easy_mode": float(easy_mode)
    }
    
    # Extract features
    features = _extract_features_from_history(
        history, user_move_counts, current_row, game_id, round_no, target_score, lookback=3
    )
    
    # Convert to single-row DataFrame
    return pd.DataFrame([features], columns=get_feature_columns()).astype(float)

# ================================================================================
# SHARED FEATURE EXTRACTION LOGIC
# ================================================================================

def _extract_features_from_history(history: List[Dict], user_move_counts: Dict[str, int],
                                 current_row: Dict, game_id: str, step_no: int,
                                 target_score: float, lookback: int = 3) -> Dict[str, float]:
    """
    Core feature extraction logic shared between training and inference.
    
    Args:
        history: List of previous game events
        user_move_counts: Running count of user moves 
        current_row: Current round data
        game_id: Game identifier
        step_no: Current step number
        target_score: Points needed to win
        lookback: How many previous moves to consider
        
    Returns:
        Feature dictionary with all 50 features
    """
    def _pad_lookback(seq, pad_token, k):
        """Pad sequence to length k"""
        return [pad_token] * (k - len(seq)) + seq
    
    # Extract lookback sequences
    u_hist = [h["user_move"] for h in history]
    b_hist = [h["bot_move"] for h in history]
    r_hist = [h["result"] for h in history]
    
    u_k = _pad_lookback(u_hist[-lookback:], "none", lookback)
    b_k = _pad_lookback(b_hist[-lookback:], "none", lookback)
    r_k = _pad_lookback(r_hist[-lookback:], "none", lookback)
    
    # Initialize feature dictionary
    features = {}
    
    # 1. User move history features
    for i, move in enumerate(u_k, 1):
        for m in MOVES:
            features[f"user_-{i}_{m}"] = 1.0 if move == m else 0.0
    
    # 2. Bot move history features
    for i, move in enumerate(b_k, 1):
        for m in MOVES:
            features[f"bot_-{i}_{m}"] = 1.0 if move == m else 0.0
    
    # 3. Result history features
    for i, result in enumerate(r_k, 1):
        for r in RESULTS:
            features[f"res_-{i}_{r}"] = 1.0 if result == r else 0.0
    
    # 4. Current round point values
    features["rock_pts"] = float(current_row.get("round_rock_pts", 0.0))
    features["paper_pts"] = float(current_row.get("round_paper_pts", 0.0))
    features["scissors_pts"] = float(current_row.get("round_scissors_pts", 0.0))
    
    # 5. Lagged point features
    for k in (1, 2, 3):
        for m in MOVES:
            if len(history) >= k:
                features[f"points_-{k}_{m}"] = float(history[-k]["round_pts"][m])
            else:
                # Use current round points as fallback
                features[f"points_-{k}_{m}"] = features[f"{m}_pts"]
    
    # 6. User tendency features
    total_moves = sum(user_move_counts.values()) or 1
    for m in MOVES:
        features[f"user_tend_{m}"] = user_move_counts[m] / total_moves
    
    # 7. Favored move tendency features
    fav_count = fav_beater_count = fav_bb_count = 0
    for h in history:
        rr, pp, ss = h["round_pts"]["rock"], h["round_pts"]["paper"], h["round_pts"]["scissors"]
        fav = _fav_from_points(rr, pp, ss)
        fav_beater = _beats(fav)
        fav_bb = _beats(fav_beater)
        
        user_move = h["user_move"]
        if user_move == fav:
            fav_count += 1
        elif user_move == fav_beater:
            fav_beater_count += 1
        elif user_move == fav_bb:
            fav_bb_count += 1
    
    denom = len(history) or 1
    features["tend_favored"] = fav_count / denom
    features["tend_fav_beater"] = fav_beater_count / denom  
    features["tend_fav_beater_beater"] = fav_bb_count / denom
    
    # 8. Score context features
    if history:
        latest = history[-1]
        user_score = latest["user_score"]
        bot_score = latest["bot_score"]
    else:
        user_score = bot_score = 0.0
        
    features["score_diff"] = float(user_score - bot_score)
    features["user_pts_to_win"] = max(0.0, float(target_score) - float(user_score))
    features["bot_pts_to_win"] = max(0.0, float(target_score) - float(bot_score))
    features["step_no"] = float(step_no)
    
    # 9. Easy mode feature (legacy compatibility flag)
    features["easy_mode"] = float(current_row.get("easy_mode", 0.0))
    
    return features


# Stateless Feature Extraction Helpers
# =============================================================================

def _backfill_score_history(
    user_moves: List[str],
    bot_moves: List[str],
    results: List[str],
    round_points: List[Dict[str, float]],
    final_user_score: float,
    final_bot_score: float,
) -> Tuple[List[float], List[float]]:
    """Reconstruct cumulative user/bot scores for each historical round.

    Args:
        user_moves: Historical user moves (one per round).
        bot_moves: Historical bot moves (aligned with ``user_moves``).
        results: Round outcomes from the user's perspective.
        round_points: Per-round point dictionaries for rock/paper/scissors.
        final_user_score: Known total user score after the final round.
        final_bot_score: Known total bot score after the final round.

    Returns:
        Two lists containing the cumulative user and bot scores *after* each round.
    """

    rounds = min(len(user_moves), len(bot_moves), len(round_points))
    if rounds == 0:
        return [], []

    # Ensure results length matches rounds
    norm_results = list(results[:rounds])
    if len(norm_results) < rounds:
        norm_results.extend(["draw"] * (rounds - len(norm_results)))

    user_scores: List[float] = []
    bot_scores: List[float] = []
    user_total = 0.0
    bot_total = 0.0

    for idx in range(rounds):
        round_result = norm_results[idx] or "draw"
        round_pts = round_points[idx] if idx < len(round_points) else {m: 1.0 for m in MOVES}
        # Normalise numeric values
        round_pts = {
            "rock": float(round_pts.get("rock", 1.0)),
            "paper": float(round_pts.get("paper", 1.0)),
            "scissors": float(round_pts.get("scissors", 1.0)),
        }

        if round_result == "win":
            move = user_moves[idx] if idx < len(user_moves) else "rock"
            user_total += round_pts.get(move, 1.0)
        elif round_result == "lose":
            move = bot_moves[idx] if idx < len(bot_moves) else "rock"
            bot_total += round_pts.get(move, 1.0)
        else:
            # Draws award a flat 0.5 to both players (see game logic)
            user_total += 0.5
            bot_total += 0.5

        user_scores.append(float(user_total))
        bot_scores.append(float(bot_total))

    # Align totals with provided scores to avoid drift
    if user_scores:
        user_diff = float(final_user_score) - user_scores[-1]
        if abs(user_diff) > 1e-6:
            user_scores[-1] = user_scores[-1] + user_diff

    if bot_scores:
        bot_diff = float(final_bot_score) - bot_scores[-1]
        if abs(bot_diff) > 1e-6:
            bot_scores[-1] = bot_scores[-1] + bot_diff

    return user_scores, bot_scores


def _build_stateless_history(
    user_moves: List[str],
    bot_moves: List[str],
    results: List[str],
    round_values: Dict[str, float],
    user_score: float,
    bot_score: float,
    round_points_history: Optional[List[Dict[str, float]]],
    user_scores_history: Optional[List[float]],
    bot_scores_history: Optional[List[float]],
) -> Tuple[List[Dict[str, float]], Dict[str, int]]:
    """Construct history rows and move counts for stateless feature extraction."""

    rounds = min(len(user_moves), len(bot_moves))
    if rounds == 0:
        return [], {m: 0 for m in MOVES}

    # Normalise results length
    norm_results = list(results[:rounds])
    if len(norm_results) < rounds:
        norm_results.extend(["draw"] * (rounds - len(norm_results)))

    # Prepare per-round point dictionaries
    default_points = {
        "rock": float(round_values.get("rock", 1.0)) if round_values else 1.0,
        "paper": float(round_values.get("paper", 1.0)) if round_values else 1.0,
        "scissors": float(round_values.get("scissors", 1.0)) if round_values else 1.0,
    }

    normalised_round_points: List[Dict[str, float]] = []
    for idx in range(rounds):
        round_pts = (
            round_points_history[idx]
            if round_points_history is not None and idx < len(round_points_history)
            else None
        )
        normalised_round_points.append({
            "rock": float((round_pts or default_points).get("rock", default_points["rock"])),
            "paper": float((round_pts or default_points).get("paper", default_points["paper"])),
            "scissors": float((round_pts or default_points).get("scissors", default_points["scissors"])),
        })

    # Determine cumulative score histories
    user_scores: Optional[List[float]] = None
    bot_scores: Optional[List[float]] = None

    if user_scores_history is not None and len(user_scores_history) >= rounds:
        user_scores = [float(score) for score in user_scores_history[:rounds]]

    if bot_scores_history is not None and len(bot_scores_history) >= rounds:
        bot_scores = [float(score) for score in bot_scores_history[:rounds]]

    if user_scores is None or bot_scores is None:
        calc_user_scores, calc_bot_scores = _backfill_score_history(
            user_moves=user_moves[:rounds],
            bot_moves=bot_moves[:rounds],
            results=norm_results,
            round_points=normalised_round_points,
            final_user_score=user_score,
            final_bot_score=bot_score,
        )
        if user_scores is None:
            user_scores = calc_user_scores
        if bot_scores is None:
            bot_scores = calc_bot_scores

    history: List[Dict[str, float]] = []
    user_move_counts = {m: 0 for m in MOVES}

    for idx in range(rounds):
        user_move = user_moves[idx]
        bot_move = bot_moves[idx]
        result = norm_results[idx]

        history.append({
            "user_move": user_move,
            "bot_move": bot_move,
            "result": result,
            "user_score": float(user_scores[idx]) if idx < len(user_scores) else float(user_score),
            "bot_score": float(bot_scores[idx]) if idx < len(bot_scores) else float(bot_score),
            "round_pts": normalised_round_points[idx],
        })

        if user_move in user_move_counts:
            user_move_counts[user_move] += 1

    return history, user_move_counts


# Stateless Feature Extraction for /predict Endpoint
# ================================================================================

def extract_features_stateless(
    user_moves: List[str],
    bot_moves: List[str],
    results: List[str] = None,
    round_values: Dict[str, float] = None,
    difficulty_mode: str = "normal",
    user_score: float = 0.0,
    bot_score: float = 0.0,
    target_score: float = 10.0,
    round_points_history: Optional[List[Dict[str, float]]] = None,
    user_scores_history: Optional[List[float]] = None,
    bot_scores_history: Optional[List[float]] = None,
) -> pd.DataFrame:
    """
    Extract features for stateless predictions (e.g., /predict endpoint).
    
    Args:
        user_moves: List of user's previous moves
        bot_moves: List of bot's previous moves
        results: List of round results (win/lose/draw from user perspective)
        round_values: Current round point values for rock/paper/scissors
        difficulty_mode: "normal" or "easy"
        user_score: Current user score
        bot_score: Current bot score
        target_score: Points needed to win
        
    Returns:
        Single-row DataFrame with 50 features
    """
    # Default to empty results if not provided
    if results is None:
        results = ["draw"] * len(user_moves)

    # Default to uniform point values if not provided
    if round_values is None:
        round_values = {"rock": 1.0, "paper": 1.0, "scissors": 1.0}

    history, user_move_counts = _build_stateless_history(
        user_moves=user_moves,
        bot_moves=bot_moves,
        results=results,
        round_values=round_values,
        user_score=user_score,
        bot_score=bot_score,
        round_points_history=round_points_history,
        user_scores_history=user_scores_history,
        bot_scores_history=bot_scores_history,
    )

    # Current round data
    current_row = {
        "round_rock_pts": round_values.get("rock", 1.0),
        "round_paper_pts": round_values.get("paper", 1.0),
        "round_scissors_pts": round_values.get("scissors", 1.0),
        "easy_mode": 1.0 if difficulty_mode == "easy" else 0.0
    }
    
    # Extract features using shared logic
    step_no = len(history) + 1  # Next step
    game_id = "stateless"  # Dummy game_id for stateless calls
    
    features = _extract_features_from_history(
        history=history,
        user_move_counts=user_move_counts,
        current_row=current_row,
        game_id=game_id,
        step_no=step_no,
        target_score=target_score,
        lookback=3
    )
    
    # Convert to DataFrame
    df = pd.DataFrame([features])
    
    # Validate feature count
    if len(df.columns) != FEATURE_COUNT:
        raise ValueError(f"Feature count mismatch: got {len(df.columns)}, expected {FEATURE_COUNT}")
    
    return df