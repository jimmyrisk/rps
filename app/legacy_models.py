# deterministic_models.py - Legacy-style bot policies (Ace, Bob, Cal, Dan, Edd, Fox, Gus, Hal)
# Cal cycles every 4 rounds between Bob and a "Bob-beater".
# Dan tapers the "best" probability based on the round's max point value.

import json
import random
import sqlite3
from collections import Counter
from typing import Dict, List, Optional

LEGACY_POLICIES: List[str] = [
    "ace",
    "bob",
    "cal",
    "dan",
    "edd",
    "fox",
    "gus",
    "hal",
]

LEGACY_DISPLAY_NAMES: Dict[str, str] = {
    "ace": "Ace",
    "bob": "Bob",
    "cal": "Cal",
    "dan": "Dan",
    "edd": "Edd",
    "fox": "Fox",
    "gus": "Gus",
    "hal": "Hal",
}

MOVES: List[str] = ["rock", "paper", "scissors"]


def normalize_legacy_policy_name(policy: str) -> str:
    """Return the canonical lowercase key for a legacy policy name."""
    return (policy or "").strip().lower()


def get_legacy_display_name(policy: str) -> str:
    """Return the display name for a legacy policy with proper capitalization."""
    key = normalize_legacy_policy_name(policy)
    return LEGACY_DISPLAY_NAMES.get(key, key.title() if key else "")


def normalize_probabilities(probs: Dict[str, float]) -> Dict[str, float]:
    """Return a normalized copy of a probability map."""
    total = sum(probs.values())
    if total <= 0:
        return {move: 1 / len(probs) for move in probs}
    return {move: value / total for move, value in probs.items()}


def weighted_choice(weights: Dict[str, float]) -> str:
    """Sample a move using the provided probability distribution."""
    normalized = normalize_probabilities(weights)
    r = random.random()
    cumulative = 0.0
    for move in MOVES:
        cumulative += normalized.get(move, 0.0)
        if r <= cumulative:
            return move
    # Fallback for rounding errors
    return random.choice(MOVES)


def beater(x: str) -> str:
    return {"rock": "paper", "paper": "scissors", "scissors": "rock"}[x]


def beaten_by(x: str) -> str:
    """Return the move that x beats"""
    return {"rock": "scissors", "paper": "rock", "scissors": "paper"}[x]


def other_move(a: str, b: str) -> str:
    return next(m for m in MOVES if m not in (a, b))


def outcome(my_move: str, opp_move: str) -> str:
    """Return outcome from my perspective: win, lose, or draw"""
    if my_move == opp_move:
        return "draw"
    return "win" if beater(opp_move) == my_move else "loss"


def opening_move_if_any(cur, game_id: str, round_no: int) -> Optional[str]:
    """
    If the games table has an opening sequence JSON array in 'opening_seq',
    return the move specified for this round (1-based). Otherwise None.
    """
    row = cur.execute("SELECT opening_seq FROM games WHERE id=?", (game_id,)).fetchone()
    if not row or not row[0]:
        return None
    try:
        seq = json.loads(row[0])
    except Exception:
        return None
    if 1 <= round_no <= len(seq):
        return seq[round_no - 1]
    return None


def policy_ace(
    round_pts: Dict[str, float],
    game_id: str,
    round_no: int,
    cur: Optional[sqlite3.Cursor] = None,
    in_memory_history: Optional[List[Dict]] = None,
) -> str:
    """Ace: Pure random strategy."""
    return random.choice(MOVES)


def policy_bob(
    round_pts: Dict[str, float],
    game_id: str,
    round_no: int,
    cur: Optional[sqlite3.Cursor] = None,
    in_memory_history: Optional[List[Dict]] = None,
) -> str:
    """Bob: Favors highest-value move, adapts after a losing streak."""
    best_move = max(round_pts, key=round_pts.get)

    history = get_game_history(cur, game_id, limit=3, in_memory_history=in_memory_history)
    if len(history) == 3 and all(h.get("result") == "lose" for h in history):
        losing_moves = [h.get("bot_move") for h in history if h.get("result") == "lose" and h.get("bot_move")]
        if losing_moves:
            most_common = Counter(losing_moves).most_common(1)[0][0]
            return most_common

    return best_move


def policy_cal(
    round_pts: Dict[str, float],
    game_id: str,
    round_no: int,
    cur: Optional[sqlite3.Cursor] = None,
    in_memory_history: Optional[List[Dict]] = None,
) -> str:
    """
    Cal: pick beater of highest-value move; if we lost 3 in a row, pick highest-value.
    """
    best = max(round_pts, key=round_pts.get)
    history = get_game_history(cur, game_id, limit=3, in_memory_history=in_memory_history)
    if len(history) == 3 and all(h.get("result") == "lose" for h in history):
        return best
    return beater(best)


def policy_dan(
    round_pts: Dict[str, float],
    game_id: str,
    round_no: int,
    cur: Optional[sqlite3.Cursor] = None,
    in_memory_history: Optional[List[Dict]] = None,
) -> str:
    """
    Dan: Bob-like, but the probability of choosing the current highest-value move
    tapers with how high that value is.

    Mapping:
      - If max round points == 2.0 -> choose 'best' with 60%
      - If max round points == 1.5 -> choose 'best' with 10%
    The difference from 60% flows to the 'other' move; 'beater(best)' stays 30%.

    This is a linear map from [1.5, 2.0] -> [0.10, 0.60].
    """
    best = max(round_pts, key=round_pts.get)
    beat = beater(best)
    oth = other_move(best, beat)

    # Clamp max value to [1.5, 2.0] for safety, then map to probability.
    max_val = float(round_pts.get(best, 1.5))
    clamped = max(1.5, min(2.0, max_val))

    # Linear map: p_best = 0.10 at 1.5 up to 0.60 at 2.0
    p_best = max(0.10, min(0.60, clamped - 1.40))
    p_beater = 0.30
    p_other = max(0.05, 1.0 - (p_best + p_beater))

    weights = {best: p_best, beat: p_beater, oth: p_other}

    history = get_game_history(cur, game_id, limit=10, in_memory_history=in_memory_history)
    if history:
        last = history[-1]
        score_diff = last["user_score"] - last["bot_score"]
        if last["result"] == "lose":
            weights[best] += 0.15
        elif last["result"] == "win":
            weights[oth] += 0.05

        if score_diff < 0:
            weights[beat] += 0.10
        elif score_diff > 0:
            weights[oth] += 0.05

        bot_moves = [h["bot_move"] for h in history if h["bot_move"]]
        if bot_moves:
            most_common, freq = Counter(bot_moves).most_common(1)[0]
            if freq >= max(2, len(bot_moves) // 2):
                counter = beater(most_common)
                weights[counter] += 0.10

    return weighted_choice(weights)


###############################################################################
# Helper: Get game history
###############################################################################


def get_game_history(
    cur: Optional[sqlite3.Cursor] = None, 
    game_id: str = "", 
    limit: Optional[int] = None,
    in_memory_history: Optional[List[Dict]] = None
) -> List[Dict]:
    """
    Fetch game history from events table OR use provided in-memory history.
    
    Args:
        cur: Database cursor (optional if in_memory_history provided)
        game_id: Game ID for database query
        limit: Limit number of records
        in_memory_history: Pre-loaded history list (bypass database)
    
    Returns list of dicts with: user_move, bot_move, result, user_score, bot_score, 
                                 round_rock_pts, round_paper_pts, round_scissors_pts
    """
    # If in-memory history provided, use it directly
    if in_memory_history is not None:
        history = in_memory_history[-limit:] if limit else in_memory_history
        return history
    
    # Otherwise query database
    if cur is None:
        return []
    
    query = """
        SELECT user_move, bot_move, result, user_score, bot_score,
               round_rock_pts, round_paper_pts, round_scissors_pts
        FROM events
        WHERE game_id = ?
        ORDER BY step_no DESC
    """
    if limit:
        query += f" LIMIT {limit}"
    
    rows = cur.execute(query, (game_id,)).fetchall()
    # Reverse to maintain chronological order (oldest first)
    rows = list(reversed(rows))
    return [
        {
            "user_move": r[0],
            "bot_move": r[1],
            "result": r[2],
            "user_score": r[3],
            "bot_score": r[4],
            "round_rock_pts": r[5],
            "round_paper_pts": r[6],
            "round_scissors_pts": r[7],
        }
        for r in rows
    ]


###############################################################################
# New Policy: Edd (value-first)
###############################################################################


def policy_edd(
    round_pts: Dict[str, float],
    game_id: str,
    round_no: int,
    cur: Optional[sqlite3.Cursor] = None,
    in_memory_history: Optional[List[Dict]] = None,
) -> str:
    """
    Edd: Value-first strategy
    - If just lost: pick highest-value move
    - If high-stakes round (any move ≥1.6): pick highest-value beater of opponent's most common
    - Otherwise: pick beater of opponent's most common move
    """
    history = get_game_history(cur, game_id, in_memory_history=in_memory_history)
    
    # Check if just lost
    just_lost = False
    if history:
        last_result = history[-1]["result"]
        # result is from user perspective: "win" means bot lost, "lose" means bot lost
        # We're the user here, so if result == "lose", we just lost
        just_lost = (last_result == "lose")
    
    # High stakes check
    high_stakes = any(v >= 1.6 for v in round_pts.values())
    
    if just_lost:
        # Pick highest-value move
        return max(round_pts.items(), key=lambda x: x[1])[0]
    
    if high_stakes:
        # Pick highest-value beater of opponent's most common
        if history:
            bot_moves = [h["bot_move"] for h in history]
            most_common_bot = max(set(bot_moves), key=bot_moves.count)
            beat_it = beater(most_common_bot)
            return beat_it
        else:
            # No history, pick highest value
            return max(round_pts.items(), key=lambda x: x[1])[0]
    
    # Default: beater of opponent's most common
    if history:
        bot_moves = [h["bot_move"] for h in history]
        most_common_bot = max(set(bot_moves), key=bot_moves.count)
        return beater(most_common_bot)
    else:
        # No history, pick highest value
        return max(round_pts.items(), key=lambda x: x[1])[0]


###############################################################################
# New Policy: Fox (protect leads)
###############################################################################


def policy_fox(
    round_pts: Dict[str, float],
    game_id: str,
    round_no: int,
    cur: Optional[sqlite3.Cursor] = None,
    in_memory_history: Optional[List[Dict]] = None,
) -> str:
    """
    Fox: Protect leads with safe plays
    - If leading: pick lowest-value move (minimize risk)
    - If behind: pick beater of opponent's least common move
    - If tied: pick move with median value
    """
    history = get_game_history(cur, game_id, in_memory_history=in_memory_history)
    
    # Get current scores
    if history:
        user_score = history[-1]["user_score"]
        bot_score = history[-1]["bot_score"]
    else:
        user_score = bot_score = 0
    
    if user_score > bot_score:
        # Leading: play safe (lowest value)
        return min(round_pts.items(), key=lambda x: x[1])[0]
    elif user_score < bot_score:
        # Behind: attack opponent's weakness
        if history:
            bot_moves = [h["bot_move"] for h in history]
            least_common_bot = min(set(bot_moves), key=bot_moves.count)
            return beater(least_common_bot)
        else:
            return random.choice(MOVES)
    else:
        # Tied: median value
        sorted_moves = sorted(round_pts.items(), key=lambda x: x[1])
        return sorted_moves[1][0]  # Middle value


###############################################################################
# New Policy: Gus (streak-based)
###############################################################################


def policy_gus(
    round_pts: Dict[str, float],
    game_id: str,
    round_no: int,
    cur: Optional[sqlite3.Cursor] = None,
    in_memory_history: Optional[List[Dict]] = None,
) -> str:
    """
    Gus: Streak-based reactions
    - If lost last 2 rounds: pick random move
    - If big edge in point values (max - min > 0.6): pick max-value move
    - Otherwise: beater of opponent's last move
    """
    history = get_game_history(cur, game_id, in_memory_history=in_memory_history)
    
    # Check for losing streak
    recent_losses = 0
    if len(history) >= 2:
        for h in history[-2:]:
            if h["result"] == "lose":
                recent_losses += 1
    
    if recent_losses >= 2:
        return random.choice(MOVES)
    
    # Big edge check
    point_values = list(round_pts.values())
    big_edge = (max(point_values) - min(point_values)) > 0.6
    
    if big_edge:
        return max(round_pts.items(), key=lambda x: x[1])[0]
    
    if history:
        last_bot_move = history[-1]["bot_move"]
        # Counter immediate repetition, especially after a loss
        if len(history) >= 2 and history[-2]["bot_move"] == last_bot_move:
            return beater(last_bot_move)

        bot_moves = [h["bot_move"] for h in history if h["bot_move"]]
        if bot_moves:
            counts = Counter(bot_moves)
            most_common, freq = counts.most_common(1)[0]
            ratio = freq / len(bot_moves)
            if ratio >= 0.5:
                return beater(most_common)

        return beater(last_bot_move)
    else:
        return random.choice(MOVES)


###############################################################################
# New Policy: Hal (scoring-based)
###############################################################################


def policy_hal(
    round_pts: Dict[str, float],
    game_id: str,
    round_no: int,
    cur: Optional[sqlite3.Cursor] = None,
    in_memory_history: Optional[List[Dict]] = None,
) -> str:
    """Hal: Prefers middle-value move unless a losing streak forces adaptation."""
    sorted_moves = sorted(round_pts.items(), key=lambda item: item[1])
    middle_move = sorted_moves[1][0] if len(sorted_moves) >= 3 else max(round_pts, key=round_pts.get)

    history = get_game_history(cur, game_id, limit=3, in_memory_history=in_memory_history)
    if len(history) == 3 and all(h.get("result") == "lose" for h in history):
        losing_moves = [h.get("bot_move") for h in history if h.get("result") == "lose" and h.get("bot_move")]
        if losing_moves:
            most_common = Counter(losing_moves).most_common(1)[0][0]
            return most_common

    return middle_move


###############################################################################
# Exported interface
###############################################################################


def choose_legacy_bot_move(*args, **kwargs) -> str:
    """Pick a move for the given legacy policy (supports legacy and keyword APIs)."""
    override_opening = kwargs.pop("opening_move_if_any", None)
    in_memory_history = kwargs.pop("in_memory_history", None)

    if args:
        # Legacy positional signature support
        legacy_policy_name = args[0]
        game_id = args[1]
        round_no = int(args[2])
        round_pts = args[3] if len(args) > 3 else kwargs.get("round_pts")
        cur = args[4] if len(args) > 4 else kwargs.get("cur")
        if len(args) > 5 and override_opening is None:
            override_opening = args[5]
        policy_name = legacy_policy_name
    else:
        policy_name = kwargs.pop("policy", None) or kwargs.pop("legacy_policy_name", None)
        game_id = kwargs.pop("game_id")
        round_no = int(kwargs.pop("round_no"))
        round_pts = kwargs.pop("round_pts", None)
        cur = kwargs.pop("cur", None)
        override_opening = kwargs.pop("override_opening", override_opening)

    if policy_name is None:
        raise ValueError("policy name is required for legacy move selection")
    
    # Allow either cursor OR in_memory_history (not both required)
    if cur is None and in_memory_history is None:
        raise ValueError("either cursor or in_memory_history is required for legacy policies")

    if round_pts is None:
        round_pts = {move: 1.0 for move in MOVES}

    if override_opening:
        return override_opening

    try:
        opening = opening_move_if_any(cur, game_id, round_no)
    except Exception:
        opening = None
    if opening:
        return opening

    lower = policy_name.lower()
    if lower == "ace":
        return policy_ace(round_pts, game_id, round_no, cur, in_memory_history)
    if lower == "bob":
        return policy_bob(round_pts, game_id, round_no, cur, in_memory_history)
    if lower == "cal":
        return policy_cal(round_pts, game_id, round_no, cur, in_memory_history)
    if lower == "dan":
        return policy_dan(round_pts, game_id, round_no, cur, in_memory_history)
    if lower == "edd":
        return policy_edd(round_pts, game_id, round_no, cur, in_memory_history)
    if lower == "fox":
        return policy_fox(round_pts, game_id, round_no, cur, in_memory_history)
    if lower == "gus":
        return policy_gus(round_pts, game_id, round_no, cur, in_memory_history)
    if lower == "hal":
        return policy_hal(round_pts, game_id, round_no, cur, in_memory_history)

    raise ValueError(f"Unknown legacy policy: {policy_name}")


def list_legacy_policies() -> List[str]:
    """Return a list of all available legacy policies."""
    return list(LEGACY_POLICIES)

