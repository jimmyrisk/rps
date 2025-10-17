# app/gambits.py
from bisect import bisect
import random
from typing import List

from app.features import MOVES, deterministic_round_points

# Name -> list of moves (any length)
GAMBITS = {
    "crescendo": ["paper", "scissors", "rock"],
    "denouement": ["rock", "scissors", "paper"],
    "paper-dolls": ["paper", "scissors", "scissors"],
    "fistful-of-dollars": ["rock", "paper", "paper"],
    "scissor-sandwich": ["paper", "scissors", "paper"],
    "bureaucrat": ["paper", "paper", "paper"],
    "avalanche": ["rock", "rock", "rock"],
    "toolbox": ["scissors", "scissors", "scissors"],
    "razor": ["scissors", "paper", "rock"],
    "switchblade": ["scissors", "rock", "paper"],
}

# Integer weights (percent-like) summing to 100
_WEIGHTS = [
    ("crescendo", 14),
    ("denouement", 12),
    ("paper-dolls", 12),
    ("fistful-of-dollars", 10),
    ("scissor-sandwich", 8),
    ("bureaucrat", 6),
    ("avalanche", 8),
    ("toolbox", 6),
    ("razor", 12),
    ("switchblade", 12),
]

# Precompute cumulative weights for O(log n) sampling
_NAMES = [name for name, _ in _WEIGHTS]
_CUM = []
_TOTAL = 0
for _, w in _WEIGHTS:
    _TOTAL += w
    _CUM.append(_TOTAL)

_RANK_MAP = {
    "rock": 0,       # Highest points for the round
    "scissors": 1,   # Second highest points
    "paper": 2,      # Third highest points
}

def _weighted_choice() -> str:
    # Sample an integer in [0, _TOTAL-1] and locate with bisect
    x = random.randrange(_TOTAL)
    i = bisect(_CUM, x)
    return _NAMES[i]

def _ordered_moves_for_round(game_id: str, round_no: int) -> List[str]:
    points = deterministic_round_points(game_id, round_no)
    return sorted(MOVES, key=lambda move: points.get(move, 0.0), reverse=True)


def _resolve_placeholder(move_placeholder: str, ordered_moves: List[str]) -> str:
    rank = _RANK_MAP.get(move_placeholder, 0)
    rank = max(0, min(rank, len(ordered_moves) - 1))
    return ordered_moves[rank]


def pick_opening(game_id: str) -> tuple[str, list[str]]:
    name = _weighted_choice()
    placeholders = GAMBITS[name]
    resolved_moves: List[str] = []

    for idx, placeholder in enumerate(placeholders, start=1):
        ordered_moves = _ordered_moves_for_round(game_id, idx)
        resolved_moves.append(_resolve_placeholder(placeholder, ordered_moves))

    return name, resolved_moves
