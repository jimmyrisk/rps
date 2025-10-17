"""
Analytics route handlers.
Handles leaderboards, game statistics, and performance metrics.
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List, Dict, Any

from app.player_filters import player_exclusion_reason, should_exclude_player
from app.db import connect
from app.config import get_metrics_since_date

router = APIRouter()

# Get the metrics cutoff date (filters old data from analytics)
METRICS_SINCE_DATE = get_metrics_since_date()


def _format_session_placeholder(session_id: Optional[str]) -> str:
    if not session_id:
        return "Player"
    clean = "".join(ch for ch in session_id if ch.isalnum())
    code = (clean[:3] or session_id[:3]).upper()
    return f"Player-{code}" if code else "Player"

@router.get("/leaderboard")
def get_leaderboard(limit: int = Query(10, ge=1, le=100)):
    """Return the top players by win rate for the active metrics window."""
    conn = connect()
    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                g.player_name,
                COUNT(*) AS total_games,
                SUM(CASE WHEN g.winner = 'user' THEN 1 ELSE 0 END) AS wins,
                SUM(CASE WHEN g.winner = 'bot' THEN 1 ELSE 0 END) AS losses
            FROM games g
            WHERE g.player_name IS NOT NULL
              AND TRIM(g.player_name) != ''
              AND g.winner IS NOT NULL
              AND g.is_test = 0
              AND g.created_ts_utc >= ?
            GROUP BY g.player_name
            HAVING total_games >= 5
            ORDER BY (wins * 1.0 / total_games) DESC, total_games DESC, g.player_name ASC
            LIMIT ?
            """,
            (METRICS_SINCE_DATE, limit),
        )

        results = cursor.fetchall()
        leaderboard: List[Dict[str, Any]] = []
        filtered_players: List[str] = []

        for row in results:
            player_name, total_games, wins, losses = row
            if should_exclude_player(player_name):
                filtered_players.append((player_name or "").strip())
                continue

            ties = max(total_games - wins - losses, 0)
            win_rate = round((wins / total_games) * 100.0, 2) if total_games else 0.0

            leaderboard.append(
                {
                    "player_name": player_name,
                    "total_games": total_games,
                    "wins": wins,
                    "losses": losses,
                    "ties": ties,
                    "win_rate": win_rate,
                }
            )

        for idx, entry in enumerate(leaderboard, start=1):
            entry["rank"] = idx

        return {
            "leaderboard": leaderboard,
            "since": METRICS_SINCE_DATE,
            "limit": limit,
            "filtered_out": sorted({name for name in filtered_players if name}),
        }
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to get leaderboard: {exc}") from exc
    finally:
        conn.close()

@router.get("/games")
def get_games(
    player_name: Optional[str] = None,
    policy: Optional[str] = None,
    easy_mode: Optional[bool] = None,
    difficulty_mode: Optional[str] = None,
    limit: int = Query(50, ge=1, le=500)
):
    """Return recent games with optional filtering.

    All responses exclude test games and respect the metrics cutoff window.
    """

    filters = ["g.is_test = 0", "g.created_ts_utc >= ?"]
    params: List[Any] = [METRICS_SINCE_DATE]

    if player_name:
        filters.append("g.player_name = ?")
        params.append(player_name)

    if policy:
        filters.append("g.bot_policy = ?")
        params.append(policy)

    # Accept legacy difficulty_mode parameter for backward compatibility
    effective_easy_mode: Optional[bool] = easy_mode
    if difficulty_mode is not None:
        normalized = difficulty_mode.strip().lower()
        if normalized in {"standard", "normal"}:
            effective_easy_mode = False
        elif normalized == "easy":
            effective_easy_mode = True
        else:
            raise HTTPException(status_code=400, detail="difficulty_mode must be 'standard' or 'easy'")

    if effective_easy_mode is not None:
        filters.append("g.easy_mode = ?")
        params.append(int(effective_easy_mode))

    where_sql = " AND ".join(filters)

    query = f"""
        SELECT
            g.id,
            g.player_name,
            g.bot_policy,
            g.bot_model_alias,
            g.easy_mode,
            g.user_score,
            g.bot_score,
            g.winner,
            g.session_id,
            g.created_ts_utc,
            g.finished_ts,
            (
                SELECT COUNT(*)
                FROM events e
                WHERE e.game_id = g.id
            ) AS round_count
        FROM games g
        WHERE {where_sql}
        ORDER BY COALESCE(g.finished_ts, g.created_ts) DESC
        LIMIT ?
    """

    params.append(limit)

    conn = connect()
    try:
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to get games: {exc}") from exc
    finally:
        conn.close()

    games = []
    for (
        game_id,
        raw_name,
        bot_policy,
        alias,
        easy_mode_flag,
        user_score,
        bot_score,
        winner,
        session_id,
        created_ts,
        finished_ts,
        round_count,
    ) in rows:
        display_name = (raw_name or "").strip() or _format_session_placeholder(session_id)
        games.append(
            {
                "game_id": game_id,
                "player_name": display_name,
                "policy": bot_policy,
                "model_alias": alias,
                "easy_mode": bool(easy_mode_flag),
                "user_score": user_score,
                "bot_score": bot_score,
                "winner": winner,
                "session_id": session_id,
                "created_ts": created_ts,
                "finished_ts": finished_ts,
                "rounds": round_count,
            }
        )

    return {"games": games, "count": len(games)}

@router.get("/stats/global")
def get_global_stats():
    """Get overall system statistics"""
    from datetime import datetime, timezone, timedelta

    conn = connect()
    cursor = conn.cursor()

    try:
        total_games = cursor.execute(
            """
            SELECT COUNT(*)
            FROM games g
            WHERE g.is_test = 0
              AND g.created_ts_utc >= ?
            """,
            (METRICS_SINCE_DATE,),
        ).fetchone()[0]

        total_rounds = cursor.execute(
            """
            SELECT COUNT(*)
            FROM events e
            JOIN games g ON g.id = e.game_id
            WHERE g.is_test = 0
              AND g.created_ts_utc >= ?
            """,
            (METRICS_SINCE_DATE,),
        ).fetchone()[0]

        event_totals = cursor.execute(
            """
            SELECT e.result, COUNT(*)
            FROM events e
            JOIN games g ON g.id = e.game_id
            WHERE g.is_test = 0
              AND g.created_ts_utc >= ?
            GROUP BY e.result
            """,
            (METRICS_SINCE_DATE,),
        ).fetchall()

        result_distribution = []
        for outcome, count in event_totals:
            percentage = round((count / total_rounds * 100.0), 2) if total_rounds else 0.0
            result_distribution.append(
                {
                    "result": outcome,
                    "count": count,
                    "percentage": percentage,
                }
            )

        policy_rows = cursor.execute(
            """
            SELECT
                g.bot_policy,
                g.easy_mode,
                COUNT(*) AS total_games,
                SUM(CASE WHEN g.winner = 'bot' THEN 1 ELSE 0 END) AS bot_wins,
                SUM(CASE WHEN g.winner = 'user' THEN 1 ELSE 0 END) AS user_wins,
                SUM(CASE WHEN g.winner = 'tie' THEN 1 ELSE 0 END) AS ties
            FROM games g
            WHERE g.is_test = 0
              AND g.winner IS NOT NULL
              AND g.created_ts_utc >= ?
            GROUP BY g.bot_policy, g.easy_mode
            ORDER BY g.bot_policy, g.easy_mode
            """,
            (METRICS_SINCE_DATE,),
        ).fetchall()

        policy_performance = []
        for policy, easy_flag, total, bot_wins, user_wins, ties in policy_rows:
            difficulty = "easy" if easy_flag else "standard"
            bot_win_rate = round((bot_wins / total) * 100.0, 2) if total else 0.0
            policy_performance.append(
                {
                    "policy": policy,
                    "difficulty_mode": difficulty,
                    "total_games": total,
                    "bot_wins": bot_wins,
                    "player_wins": user_wins,
                    "ties": ties,
                    "bot_win_rate": bot_win_rate,
                }
            )

        now = datetime.now(timezone.utc)
        cutoff_24h = (now - timedelta(hours=24)).isoformat()
        cutoff_7d = (now - timedelta(days=7)).isoformat()

        moves_last_24h = cursor.execute(
            """
            SELECT COUNT(*)
            FROM events e
            JOIN games g ON g.id = e.game_id
            WHERE g.is_test = 0
              AND g.created_ts_utc >= ?
              AND COALESCE(e.created_ts_utc, e.ts) >= ?
            """,
            (METRICS_SINCE_DATE, cutoff_24h),
        ).fetchone()[0]

        moves_last_week = cursor.execute(
            """
            SELECT COUNT(*)
            FROM events e
            JOIN games g ON g.id = e.game_id
            WHERE g.is_test = 0
              AND g.created_ts_utc >= ?
              AND COALESCE(e.created_ts_utc, e.ts) >= ?
            """,
            (METRICS_SINCE_DATE, cutoff_7d),
        ).fetchone()[0]

        return {
            "total_games": total_games,
            "total_rounds": total_rounds,
            "result_distribution": result_distribution,
            "policy_performance": policy_performance,
            "recent_activity": {
                "rounds_last_24h": moves_last_24h,
                "rounds_last_week": moves_last_week,
            },
        }
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to get global stats: {exc}") from exc
    finally:
        conn.close()

@router.get("/stats/policy/{policy}")
def get_policy_stats(
    policy: str,
    easy_mode: Optional[bool] = None,
):
    """Get detailed statistics for a specific policy"""

    conn = connect()
    cursor = conn.cursor()

    game_filters = ["g.bot_policy = ?", "g.is_test = 0", "g.created_ts_utc >= ?"]
    game_params: List[Any] = [policy, METRICS_SINCE_DATE]

    if easy_mode is not None:
        game_filters.append("g.easy_mode = ?")
        game_params.append(int(easy_mode))

    where_clause = " AND ".join(game_filters)

    try:
        totals = cursor.execute(
            f"""
            SELECT
                COUNT(*) AS total_games,
                SUM(CASE WHEN g.winner = 'bot' THEN 1 ELSE 0 END) AS bot_wins,
                SUM(CASE WHEN g.winner = 'user' THEN 1 ELSE 0 END) AS player_wins,
                SUM(CASE WHEN g.winner = 'tie' THEN 1 ELSE 0 END) AS ties
            FROM games g
            WHERE {where_clause}
            """,
            game_params,
        ).fetchone()

        if not totals or not totals[0]:
            raise HTTPException(status_code=404, detail=f"No data found for policy: {policy}")

        total_games, bot_wins, player_wins, ties = [int(x or 0) for x in totals]
        bot_win_rate = round((bot_wins / total_games) * 100.0, 2) if total_games else 0.0

        event_filters = ["g.bot_policy = ?", "g.is_test = 0", "g.created_ts_utc >= ?"]
        event_params: List[Any] = [policy, METRICS_SINCE_DATE]
        if easy_mode is not None:
            event_filters.append("g.easy_mode = ?")
            event_params.append(int(easy_mode))

        event_where = " AND ".join(event_filters)

        total_rounds = cursor.execute(
            f"""
            SELECT COUNT(*)
            FROM events e
            JOIN games g ON g.id = e.game_id
            WHERE {event_where}
            """,
            event_params,
        ).fetchone()[0]

        distribution_rows = cursor.execute(
            f"""
            SELECT e.bot_move, COUNT(*)
            FROM events e
            JOIN games g ON g.id = e.game_id
            WHERE {event_where}
            GROUP BY e.bot_move
            ORDER BY COUNT(*) DESC
            """,
            event_params,
        ).fetchall()

        move_distribution = []
        for move, count in distribution_rows:
            pct = round((count / total_rounds) * 100.0, 2) if total_rounds else 0.0
            move_distribution.append({"move": move, "count": count, "percentage": pct})

        difficulty_breakdown: List[Dict[str, Any]] = []
        if easy_mode is None:
            breakdown_rows = cursor.execute(
                """
                SELECT
                    g.easy_mode,
                    COUNT(*) AS total_games,
                    SUM(CASE WHEN g.winner = 'bot' THEN 1 ELSE 0 END) AS bot_wins
                FROM games g
                WHERE g.bot_policy = ?
                  AND g.is_test = 0
                  AND g.created_ts_utc >= ?
                GROUP BY g.easy_mode
                ORDER BY g.easy_mode
                """,
                (policy, METRICS_SINCE_DATE),
            ).fetchall()

            for easy_flag, total, bot_wins_easy in breakdown_rows:
                rate = round((bot_wins_easy / total) * 100.0, 2) if total else 0.0
                difficulty_breakdown.append(
                    {
                        "difficulty_mode": "easy" if easy_flag else "standard",
                        "total_games": total,
                        "bot_win_rate": rate,
                    }
                )

        return {
            "policy": policy,
            "easy_mode": easy_mode,
            "total_games": total_games,
            "total_rounds": total_rounds,
            "player_wins": player_wins,
            "bot_wins": bot_wins,
            "ties": ties,
            "bot_win_rate": bot_win_rate,
            "move_distribution": move_distribution,
            "difficulty_breakdown": difficulty_breakdown,
        }
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to get policy stats: {exc}") from exc
    finally:
        conn.close()

@router.get("/stats/player/{player_name}")
def get_player_stats(player_name: str):
    """Get detailed statistics for a specific player"""
    conn = connect()
    cursor = conn.cursor()

    try:
        totals = cursor.execute(
            """
            SELECT
                COUNT(*) AS total_games,
                SUM(CASE WHEN g.winner = 'user' THEN 1 ELSE 0 END) AS wins,
                SUM(CASE WHEN g.winner = 'bot' THEN 1 ELSE 0 END) AS losses,
                SUM(CASE WHEN g.winner = 'tie' THEN 1 ELSE 0 END) AS ties
            FROM games g
            WHERE g.player_name = ?
              AND g.is_test = 0
              AND g.created_ts_utc >= ?
            """,
            (player_name, METRICS_SINCE_DATE),
        ).fetchone()

        if not totals or not totals[0]:
            raise HTTPException(status_code=404, detail=f"No data found for player: {player_name}")

        total_games, wins, losses, ties = [int(x or 0) for x in totals]
        win_rate = round((wins / total_games) * 100.0, 2) if total_games else 0.0

        total_rounds = cursor.execute(
            """
            SELECT COUNT(*)
            FROM events e
            JOIN games g ON g.id = e.game_id
            WHERE g.player_name = ?
              AND g.is_test = 0
              AND g.created_ts_utc >= ?
            """,
            (player_name, METRICS_SINCE_DATE),
        ).fetchone()[0]

        policy_rows = cursor.execute(
            """
            SELECT
                g.bot_policy,
                g.easy_mode,
                COUNT(*) AS total_games,
                SUM(CASE WHEN g.winner = 'user' THEN 1 ELSE 0 END) AS wins
            FROM games g
            WHERE g.player_name = ?
              AND g.is_test = 0
              AND g.created_ts_utc >= ?
            GROUP BY g.bot_policy, g.easy_mode
            ORDER BY g.bot_policy, g.easy_mode
            """,
            (player_name, METRICS_SINCE_DATE),
        ).fetchall()

        policy_performance = []
        for policy, easy_flag, total, player_wins in policy_rows:
            rate = round((player_wins / total) * 100.0, 2) if total else 0.0
            policy_performance.append(
                {
                    "policy": policy,
                    "difficulty_mode": "easy" if easy_flag else "standard",
                    "total_games": total,
                    "wins": player_wins,
                    "win_rate": rate,
                }
            )

        move_rows = cursor.execute(
            """
            SELECT e.user_move, COUNT(*)
            FROM events e
            JOIN games g ON g.id = e.game_id
            WHERE g.player_name = ?
              AND g.is_test = 0
              AND g.created_ts_utc >= ?
            GROUP BY e.user_move
            ORDER BY COUNT(*) DESC
            """,
            (player_name, METRICS_SINCE_DATE),
        ).fetchall()
        
        move_distribution = []
        for move, count in move_rows:
            pct = round((count / total_rounds) * 100.0, 2) if total_rounds else 0.0
            move_distribution.append({"move": move, "count": count, "percentage": pct})

        return {
            "player_name": player_name,
            "total_games": total_games,
            "total_rounds": total_rounds,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "win_rate": win_rate,
            "policy_performance": policy_performance,
            "move_distribution": move_distribution,
        }
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to get player stats: {exc}") from exc
    finally:
        conn.close()

@router.get("/export")
def export_game_data(
    format: str = Query("json", regex="^(json|csv)$"),
    limit: int = Query(1000, ge=1, le=10000)
):
    """Export game data in JSON or CSV format"""
    conn = connect()
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
            SELECT
                e.game_id,
                e.user_move,
                e.bot_move,
                e.result,
                g.bot_policy,
                g.bot_model_alias,
                g.easy_mode,
                g.session_id,
                g.player_name,
                COALESCE(e.created_ts_utc, e.ts) AS event_ts
            FROM events e
            JOIN games g ON g.id = e.game_id
            WHERE g.is_test = 0
              AND g.created_ts_utc >= ?
            ORDER BY event_ts DESC
            LIMIT ?
            """,
            (METRICS_SINCE_DATE, limit),
        )

        rows = cursor.fetchall()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to export data: {exc}") from exc
    finally:
        conn.close()

    if format == "json":
        data = [
            {
                "game_id": game_id,
                "player_move": user_move,
                "bot_move": bot_move,
                "result": result,
                "bot_policy": bot_policy,
                "model_alias": alias,
                "easy_mode": bool(easy_flag),
                "session_id": session_id,
                "player_name": player_name,
                "event_ts": event_ts,
            }
            for (
                game_id,
                user_move,
                bot_move,
                result,
                bot_policy,
                alias,
                easy_flag,
                session_id,
                player_name,
                event_ts,
            ) in rows
        ]

        return {"data": data, "count": len(data), "format": "json"}

    headers = [
        "game_id",
        "player_move",
        "bot_move",
        "result",
        "bot_policy",
        "model_alias",
        "easy_mode",
        "session_id",
        "player_name",
        "event_ts",
    ]

    csv_data = [headers]
    for row in rows:
        # Convert easy_mode flag to bool/int for CSV readability
        row_list = list(row)
        row_list[6] = bool(row_list[6])
        csv_data.append(row_list)

    return {"data": csv_data, "count": len(rows), "format": "csv"}

# Additional endpoints from original monolith
def now_iso():
    from datetime import datetime
    return datetime.utcnow().isoformat() + "Z"

@router.get("/stats")
def stats(session_id: Optional[str] = None, game_id: Optional[str] = None):
    """Get game or session statistics (excludes test data and respects metrics cutoff date)"""
    conn = connect()
    cur = conn.cursor()
    
    if game_id:
        n = cur.execute("""
            SELECT COUNT(*) FROM events e
            JOIN games g ON g.id = e.game_id
            WHERE e.game_id=? AND g.is_test = 0 AND g.created_ts_utc >= ?
        """, (game_id, METRICS_SINCE_DATE)).fetchone()[0]
        
        wins = cur.execute("""
            SELECT COUNT(*) FROM events e
            JOIN games g ON g.id = e.game_id
            WHERE e.game_id=? AND e.result='win' AND g.is_test = 0 AND g.created_ts_utc >= ?
        """, (game_id, METRICS_SINCE_DATE)).fetchone()[0]
        
        loses = cur.execute("""
            SELECT COUNT(*) FROM events e
            JOIN games g ON g.id = e.game_id
            WHERE e.game_id=? AND e.result='lose' AND g.is_test = 0 AND g.created_ts_utc >= ?
        """, (game_id, METRICS_SINCE_DATE)).fetchone()[0]
        
        draws = n - wins - loses
        g = cur.execute("""
            SELECT user_score, bot_score, winner FROM games 
            WHERE id=? AND is_test = 0 AND created_ts_utc >= ?
        """, (game_id, METRICS_SINCE_DATE)).fetchone()
        us, bs, winner = g if g else (0,0,None)
        conn.close()
        return {"scope":"game","rounds":n,"wins":wins,"losses":loses,"draws":draws,
                "user_score":us,"bot_score":bs,"winner":winner}
    
    # session scope (aggregate across games)
    if not session_id: 
        conn.close()
        raise HTTPException(400, "need session_id or game_id")
    
    n = cur.execute("""SELECT COUNT(*) FROM events e
                       JOIN games g ON g.id=e.game_id
                       WHERE g.session_id=? AND g.is_test = 0 AND g.created_ts_utc >= ?""", 
                    (session_id, METRICS_SINCE_DATE)).fetchone()[0]
    wins = cur.execute("""SELECT COUNT(*) FROM events e
                          JOIN games g ON g.id=e.game_id
                          WHERE g.session_id=? AND e.result='win' AND g.is_test = 0 AND g.created_ts_utc >= ?""", 
                       (session_id, METRICS_SINCE_DATE)).fetchone()[0]
    loses = cur.execute("""SELECT COUNT(*) FROM events e
                           JOIN games g ON g.id=e.game_id
                           WHERE g.session_id=? AND e.result='lose' AND g.is_test = 0 AND g.created_ts_utc >= ?""", 
                        (session_id, METRICS_SINCE_DATE)).fetchone()[0]
    draws = n - wins - loses
    games_played = cur.execute("""
        SELECT COUNT(*) FROM games 
        WHERE session_id=? AND is_test = 0 AND created_ts_utc >= ?
    """, (session_id, METRICS_SINCE_DATE)).fetchone()[0]
    conn.close()
    return {"scope":"session","rounds":n,"wins":wins,"losses":loses,"draws":draws,"games":games_played}

@router.get("/bot_win_rates")
def bot_win_rates(easy_mode: bool = False, human_only: bool = True):
    """Get bot win rates versus human players with shared filtering rules."""
    from app.policies import list_policies

    conn = connect()
    try:
        cursor = conn.cursor()
        rows = cursor.execute(
            """
            SELECT
                g.bot_policy,
                g.player_name,
                g.winner
            FROM games g
            WHERE g.winner IS NOT NULL
              AND g.bot_policy IS NOT NULL
              AND g.easy_mode = ?
              AND g.is_test = 0
              AND g.created_ts_utc >= ?
            """,
            (int(easy_mode), METRICS_SINCE_DATE),
        ).fetchall()

        policies = list_policies()
        bot_rates: Dict[str, Dict[str, Any]] = {
            policy: {
                "bot_win_rate": None,
                "human_win_rate": None,
                "human_games": 0,
                "bot_wins": 0,
                "human_wins": 0,
                "ties": 0,
                "excluded_games": 0,
                "total_games_tracked": 0,
            }
            for policy in policies
        }

        exclusion_samples: Dict[str, set[str]] = {}
        total_excluded = 0

        for bot_policy, player_name, winner in rows:
            if bot_policy not in bot_rates:
                continue

            stats = bot_rates[bot_policy]
            stats["total_games_tracked"] += 1

            reason = player_exclusion_reason(player_name) if human_only else None
            if reason:
                stats["excluded_games"] += 1
                total_excluded += 1
                if player_name:
                    exclusion_samples.setdefault(reason, set()).add(player_name.strip())
                continue

            stats["human_games"] += 1

            if winner == "bot":
                stats["bot_wins"] += 1
            elif winner == "user":
                stats["human_wins"] += 1
            else:
                stats["ties"] += 1

        for stats in bot_rates.values():
            games = stats["human_games"]
            if games:
                stats["bot_win_rate"] = round((stats["bot_wins"] / games) * 100.0, 2)
                stats["human_win_rate"] = round((stats["human_wins"] / games) * 100.0, 2)
            else:
                stats["bot_win_rate"] = None
                stats["human_win_rate"] = None

        exclusions_payload = {
            "total_games": total_excluded,
            "by_reason": {reason: len(names) for reason, names in exclusion_samples.items()},
            "sample_names": {reason: sorted(names)[:5] for reason, names in exclusion_samples.items()},
        }

        return {
            "bot_rates": bot_rates,
            "easy_mode": bool(easy_mode),
            "human_only": bool(human_only),
            "since": METRICS_SINCE_DATE,
            "timestamp": now_iso(),
            "exclusions": exclusions_payload,
        }
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Database error: {exc}") from exc
    finally:
        conn.close()

@router.get("/win_streaks")
def win_streaks(opponent: Optional[str] = None, limit: int = 10, easy_mode: bool = False):
    """Calculate actual consecutive win streaks for players (excludes test data and respects metrics cutoff date)"""
    conn = connect()
    try:
        cur = conn.cursor()

        base_query = """
        SELECT 
            g.session_id,
            g.player_name,
            g.id,
            g.winner,
            g.finished_ts,
            g.bot_policy
        FROM games g
        WHERE g.winner IS NOT NULL
          AND g.finished_ts IS NOT NULL
          AND g.is_test = 0
          AND g.created_ts_utc >= ?
        """

        params = [METRICS_SINCE_DATE]
        if opponent:
            base_query += " AND g.bot_policy = ?"
            params.append(opponent)

        base_query += " AND g.easy_mode = ?"
        params.append(int(easy_mode))

        base_query += " ORDER BY g.session_id, g.finished_ts"

        rows = cur.execute(base_query, params).fetchall()

        player_streaks: Dict[str, Dict[str, Any]] = {}
        filtered_players: Dict[str, str] = {}

        for session_id, raw_name, game_id, winner, finished_ts, bot_policy in rows:
            display_name = (raw_name or "").strip()
            if not display_name or display_name.lower() == "player":
                display_name = _format_session_placeholder(session_id)

            exclusion_reason = player_exclusion_reason(raw_name or display_name)
            if exclusion_reason:
                filtered_players[display_name] = exclusion_reason
                continue

            key = session_id or display_name
            if key not in player_streaks:
                player_streaks[key] = {
                    "display_name": display_name,
                    "current_streak": 0,
                    "max_streak": 0,
                    "last_win_date": None,
                    "games_played": 0
                }
            elif raw_name and raw_name.strip() and raw_name.strip().lower() != "player":
                # Upgrade placeholder names if the player later provides one
                player_streaks[key]["display_name"] = raw_name.strip()

            entry = player_streaks[key]
            entry["games_played"] += 1

            if winner == "user":
                entry["current_streak"] += 1
                entry["last_win_date"] = finished_ts
                entry["max_streak"] = max(entry["max_streak"], entry["current_streak"])
            else:
                entry["current_streak"] = 0

        sorted_players = sorted(
            player_streaks.values(),
            key=lambda data: (data["max_streak"], data["current_streak"]),
            reverse=True
        )[:limit]

        leaderboard = [
            {
                "username": data["display_name"],
                "streak": data["max_streak"],
                "current_streak": data["current_streak"],
                "last_win_date": data["last_win_date"] or now_iso(),
                "total_games": data["games_played"]
            }
            for data in sorted_players
            if data["max_streak"] > 0
        ]

        return {
            "players": leaderboard,
            "filters": {
                "opponent": opponent,
                "limit": limit,
                "easy_mode": bool(easy_mode)
            },
            "since": METRICS_SINCE_DATE,
            "timestamp": now_iso(),
            "filtered_out": [{"player": name, "reason": reason} for name, reason in sorted(filtered_players.items())],
        }
    finally:
        conn.close()