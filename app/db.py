"""Database utilities for the RPS service.

This module centralises SQLite connection handling and schema creation so the
FastAPI application and background jobs share the same bootstrap logic.
"""
from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_DB_FILENAME = "rps.db"
_DATA_ENV_VAR = "DATA_PATH"
_DB_PATH_ENV_VAR = "RPS_DB_PATH"

_DATA_DIR = Path(os.getenv(_DATA_ENV_VAR, "/data"))
_DATA_DIR.mkdir(parents=True, exist_ok=True)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_LOCAL_DB_FALLBACK = _PROJECT_ROOT / "local" / _DEFAULT_DB_FILENAME


def resolve_db_path(explicit: Optional[str | os.PathLike[str]] = None) -> Path:
    """Return the SQLite database path, honouring overrides and fallbacks.

    Priority order:
    1. ``explicit`` argument if provided.
    2. ``RPS_DB_PATH`` environment variable.
    3. ``DATA_PATH``/``/data`` directory shared across services.
    4. ``local/rps.db`` (useful for local development when shared volume is
       absent).
    """

    if explicit:
        path = Path(explicit)
    else:
        env_override = os.getenv(_DB_PATH_ENV_VAR)
        if env_override:
            path = Path(env_override)
        else:
            path = _DATA_DIR / _DEFAULT_DB_FILENAME

    if not path.exists() and _LOCAL_DB_FALLBACK.exists():
        path = _LOCAL_DB_FALLBACK

    # Ensure target directory exists to avoid runtime errors when SQLite tries
    # to create the file. ``resolve`` keeps relative paths predictable.
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create tables if they don't already exist."""

    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            user_name TEXT,
            created_ts TEXT
        );
        """
    )
    conn.execute(
        """
                CREATE TABLE IF NOT EXISTS games (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    bot_policy TEXT NOT NULL,
                    bot_model_version TEXT NOT NULL,
                    user_score REAL NOT NULL DEFAULT 0,
                    bot_score REAL NOT NULL DEFAULT 0,
                    winner TEXT,
                    created_ts TEXT NOT NULL,
                    start_ts TEXT,
                    finished_ts TEXT,
                    opening_name TEXT,
                    opening_seq TEXT,
                    easy_mode INTEGER DEFAULT 0  -- Deprecated flag retained for legacy compatibility
                );
        """
    )

    # Ensure legacy scoring columns exist so inserts remain compatible across
    # fresh databases and long-lived ones. These fields are still populated by
    # the application even though gameplay logic has moved to the events table.
    legacy_columns = (
        ("rock_pts", "REAL DEFAULT 1.0"),
        ("paper_pts", "REAL DEFAULT 1.0"),
        ("scissors_pts", "REAL DEFAULT 1.0"),
        ("user_rock_ct", "INTEGER DEFAULT 0"),
        ("user_paper_ct", "INTEGER DEFAULT 0"),
        ("user_scissors_ct", "INTEGER DEFAULT 0"),
        ("user_favored_pick_ct", "INTEGER DEFAULT 0"),
        ("user_favored_beater_pick_ct", "INTEGER DEFAULT 0"),
    )

    for column_name, column_def in legacy_columns:
        try:
            conn.execute(f"ALTER TABLE games ADD COLUMN {column_name} {column_def}")
        except sqlite3.OperationalError:
            # Column already exists
            pass

    # Add easy_mode column to existing games table if it doesn't exist (legacy, defaults to 0)
    try:
        conn.execute("ALTER TABLE games ADD COLUMN easy_mode INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        # Column already exists
        pass
    
    # Add is_test column for test data exclusion
    try:
        conn.execute("ALTER TABLE games ADD COLUMN is_test INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        # Column already exists
        pass
    
    # Add created_ts_utc column for precise time filtering
    try:
        conn.execute("ALTER TABLE games ADD COLUMN created_ts_utc TEXT")
    except sqlite3.OperationalError:
        # Column already exists
        pass
    
    # Add bot_model_alias to track which variant was used (Production, B, shadow1, shadow2)
    try:
        conn.execute("ALTER TABLE games ADD COLUMN bot_model_alias TEXT DEFAULT 'Production'")
    except sqlite3.OperationalError:
        # Column already exists
        pass
    
    # Add player_name for leaderboards
    try:
        conn.execute("ALTER TABLE games ADD COLUMN player_name TEXT")
    except sqlite3.OperationalError:
        # Column already exists
        pass
    
    # Fix legacy rock/paper/scissors_pts columns: set defaults if they exist
    # These were removed from schema but may still exist in production DB with NOT NULL constraints
    try:
        # Check if rock_pts exists and has values
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM games WHERE rock_pts IS NULL OR rock_pts = 0")
        null_count = cursor.fetchone()[0]
        if null_count > 0:
            # Set defaults for any NULL/0 values
            conn.execute("UPDATE games SET rock_pts = 1.0 WHERE rock_pts IS NULL OR rock_pts = 0")
            conn.execute("UPDATE games SET paper_pts = 1.0 WHERE paper_pts IS NULL OR paper_pts = 0")
            conn.execute("UPDATE games SET scissors_pts = 1.0 WHERE scissors_pts IS NULL OR scissors_pts = 0")
    except sqlite3.OperationalError:
        # Column doesn't exist (new database)
        pass
    
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        game_id TEXT NOT NULL,
        ts TEXT NOT NULL,
        user_move TEXT NOT NULL,
        bot_move TEXT NOT NULL,
        result TEXT NOT NULL,
        user_delta REAL NOT NULL,
        bot_delta REAL NOT NULL,
        user_score REAL NOT NULL,
        bot_score REAL NOT NULL,
        step_no INTEGER,
        round_rock_pts REAL, round_paper_pts REAL, round_scissors_pts REAL,
        lag1_rock_pts REAL, lag1_paper_pts REAL, lag1_scissors_pts REAL,
        lag2_rock_pts REAL, lag2_paper_pts REAL, lag2_scissors_pts REAL,
        lag3_rock_pts REAL, lag3_paper_pts REAL, lag3_scissors_pts REAL
        );
        """
    )
    
    # Add created_ts_utc column to events for precise filtering
    try:
        conn.execute("ALTER TABLE events ADD COLUMN created_ts_utc TEXT")
    except sqlite3.OperationalError:
        # Column already exists
        pass

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS promotion_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_ts_utc TEXT NOT NULL,
            model_type TEXT NOT NULL,
            decision TEXT NOT NULL,
            z_statistic REAL,
            p_value REAL,
            production_wins INTEGER,
            production_total_games INTEGER,
            b_wins INTEGER,
            b_total_games INTEGER,
            reorder_applied INTEGER NOT NULL DEFAULT 0,
            alias_rankings_json TEXT,
            alias_accuracies_json TEXT,
            alias_assignments_json TEXT,
            production_games_since_swap INTEGER,
            b_games_since_swap INTEGER,
            promotion_cycles_since_swap INTEGER,
            reason TEXT,
            source TEXT NOT NULL,
            payload_json TEXT NOT NULL
        );
        """
    )

    # Backfill newly added JSON columns for historical databases.
    try:
        conn.execute("ALTER TABLE promotion_events ADD COLUMN alias_accuracies_json TEXT")
    except sqlite3.OperationalError:
        pass

    try:
        conn.execute("ALTER TABLE promotion_events ADD COLUMN alias_assignments_json TEXT")
    except sqlite3.OperationalError:
        pass

    try:
        conn.execute("ALTER TABLE promotion_events ADD COLUMN production_games_since_swap INTEGER")
    except sqlite3.OperationalError:
        pass

    try:
        conn.execute("ALTER TABLE promotion_events ADD COLUMN b_games_since_swap INTEGER")
    except sqlite3.OperationalError:
        pass

    try:
        conn.execute("ALTER TABLE promotion_events ADD COLUMN promotion_cycles_since_swap INTEGER")
    except sqlite3.OperationalError:
        pass

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_promotion_events_created
        ON promotion_events (created_ts_utc);
        """
    )

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_promotion_events_model
        ON promotion_events (model_type, created_ts_utc);
        """
    )


def connect(explicit_path: Optional[str | os.PathLike[str]] = None) -> sqlite3.Connection:
    """Initialise and return a SQLite connection with the project schema."""

    db_path = resolve_db_path(explicit_path)
    try:
        conn = sqlite3.connect(db_path, check_same_thread=False)
    except sqlite3.Error:
        logger.exception("Failed to connect to SQLite database at %s", db_path)
        raise

    _ensure_schema(conn)
    return conn


DB = connect()

__all__ = ["DB", "connect", "resolve_db_path"]
