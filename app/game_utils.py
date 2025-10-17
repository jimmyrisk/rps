"""Shared gameplay utilities used across API endpoints and tests."""
from __future__ import annotations

from datetime import datetime, timezone
import sqlite3


def now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def beats(choice: str) -> str:
    """Return the move that defeats ``choice`` in Rock-Paper-Scissors."""
    return {"rock": "paper", "paper": "scissors", "scissors": "rock"}[choice]


def it_beats(choice: str) -> str:
    """Return the move that ``choice`` defeats."""
    return {"rock": "scissors", "paper": "rock", "scissors": "paper"}[choice]


def outcome(user: str, bot: str) -> str:
    """Compute the round outcome from the user's perspective."""
    if user == bot:
        return "draw"
    return "win" if beats(bot) == user else "lose"


def _event_count(cur: sqlite3.Cursor, game_id: str) -> int:
    row = cur.execute("SELECT COUNT(1) FROM events WHERE game_id=?", (game_id,)).fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def get_next_round_number(cur: sqlite3.Cursor, game_id: str) -> int:
    """Return the upcoming round number for the specified game."""
    return _event_count(cur, game_id) + 1


__all__ = [
    "now_iso",
    "beats",
    "it_beats",
    "outcome",
    "get_next_round_number",
]
