#!/usr/bin/env python3
"""Bot vs Bot Game Simulation Script for the RPS MLOps System.

This CLI exercises the public gameplay API by repeatedly starting games via
`/start_game` and submitting moves through `/play`. It keeps the interface that
lets you pick a "user" policy (local strategy we simulate client-side) and the
server-side opponent policy exposed by FastAPI.

Examples:
    python bot_vs_bot_sim.py --user random --bot brian --games 5
    python bot_vs_bot_sim.py --user random --bot logan --games 10
"""

import argparse
import os
import random
import sys
from collections import Counter
from typing import Any, Dict, Optional, List

import requests

# Import legacy models for proper policy implementation
try:
    from app.legacy_models import (
        choose_legacy_bot_move,
        get_legacy_display_name,
        list_legacy_policies,
    )
    LEGACY_POLICIES_AVAILABLE = True
except ImportError:
    LEGACY_POLICIES_AVAILABLE = False
    print("Warning: Could not import legacy models. Using simplified fallbacks.")

# Configuration
API_BASE_URL = "http://localhost:8080"
HEALTH_ENDPOINTS = ("/healthz", "/health")
MOVES = ["rock", "paper", "scissors"]
REQUEST_TIMEOUT = float(os.getenv("SIM_REQUEST_TIMEOUT", "30"))


def pick_user_move(
    policy: str, 
    *, 
    round_pts: Dict[str, float], 
    game_id: str, 
    round_no: int, 
    api: "APIClient",
    game_history: Optional[List[Dict[str, Any]]] = None
) -> str:
    """Return the next user move for the chosen policy.

    For legacy policies (ace, bob, cal, dan, edd, fox, gus, hal), delegate to app.legacy_models.
    For ML policies (brian, forrest, logan), use simple random strategy.
    Anything unrecognised falls back to random play.
    
    Args:
        policy: User strategy name
        round_pts: Current round point values {rock, paper, scissors}
        game_id: Game identifier
        round_no: Current round number
        api: API client (currently unused)
        game_history: List of previous round results [{user_move, bot_move, result, user_score, bot_score}, ...]
    """

    policy = (policy or "").lower()
    
    # For ML policies, just use random since we're simulating the user side
    if policy in {"brian", "forrest", "logan"}:
        return random.choice(MOVES)
    
    # For legacy policies, use the canonical implementation
    if LEGACY_POLICIES_AVAILABLE and policy in list_legacy_policies():
        try:
            return choose_legacy_bot_move(
                policy=policy,
                game_id=game_id,
                round_no=round_no,
                round_pts=round_pts,
                cur=None,  # No database cursor available
                in_memory_history=game_history  # Pass in-memory history instead
            )
        except Exception as e:
            print(f"Warning: Failed to use legacy policy {policy}: {e}")
            # Fall through to random fallback
    
    # Fallback for unknown policies or if legacy_models not available
    return random.choice(MOVES)

class APIClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()
    
    def start_game(
        self,
        *,
        policy: str,
        difficulty_mode: str = "normal",
        player_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "policy": policy,
            "difficulty_mode": difficulty_mode,
        }
        if player_name:
            payload["player_name"] = player_name

        response = self.session.post(
            f"{self.base_url}/start_game",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()
    
    def get_round_points(self, *, game_id: str) -> Dict[str, Any]:
        response = self.session.get(
            f"{self.base_url}/round_points",
            params={"game_id": game_id},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()
    
    def play_round(self, *, game_id: str, user_move: str, easy_mode: bool = False) -> Dict[str, Any]:
        response = self.session.post(
            f"{self.base_url}/play",
            json={"game_id": game_id, "user_move": user_move, "easy_mode": easy_mode},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()

def _resolve_player_name(user_bot: str) -> Optional[str]:
    if not LEGACY_POLICIES_AVAILABLE:
        return user_bot.title() if user_bot else None
    key = user_bot.lower() if user_bot else ""
    if key in list_legacy_policies():
        return get_legacy_display_name(key)
    return user_bot.title() if user_bot else None


def play_single_game(
    api: APIClient,
    user_bot: str,
    server_bot: str,
    verbose: bool = False,
    *,
    easy_mode: bool = False,
    difficulty_mode: str = "normal",
) -> Dict[str, Any]:
    """Play a single game between two bot policies."""

    try:
        player_name = _resolve_player_name(user_bot)
        game = api.start_game(
            policy=server_bot,
            difficulty_mode=difficulty_mode,
            player_name=player_name,
        )
        game_id = game["game_id"]
        session_id = game.get("session_id")
        target_score = game.get("target_score", 10)
        model_alias = game.get("bot_model_alias", "unknown")
        model_type = game.get("model_type", "unknown")
        last_bot_move: Optional[str] = None

        model_usage_counter: Counter[str] = Counter()
        prediction_source_counter: Counter[str] = Counter()
        probability_source_counter: Counter[str] = Counter()
        alias_counter: Counter[str] = Counter()
        alias_counter[model_alias] += 1
        
        # Track game history for new policies (edd, fox, gus, hal)
        game_history: List[Dict[str, Any]] = []

        if verbose:
            print(f"Started game {game_id} (session {session_id}): {user_bot} vs {server_bot}")

        max_rounds = 60

        for round_num in range(max_rounds):
            # Get the actual round points from the API
            round_info = api.get_round_points(game_id=game_id)
            round_pts = round_info["round_points"]
            round_no = round_info["round_no"]
            
            move = pick_user_move(
                user_bot, 
                round_pts=round_pts,
                game_id=game_id,
                round_no=round_no,
                api=api,
                game_history=game_history  # Pass history for new policies
            )
            response = api.play_round(
                game_id=game_id,
                user_move=move,
                easy_mode=easy_mode,
            )
            last_bot_move = response.get("bot_move")
            
            # Extract model info from response (handles both top-level and nested structures)
            model_info = response.get("model_info", {})
            debug_info = response.get("debug", {})
            
            # Get model type (prefer model_info, fallback to debug)
            model_type = model_info.get("model_type") or debug_info.get("model_type") or "unknown"
            
            # Get probability source
            prob_source = model_info.get("probability_source") or debug_info.get("probability_source") or "unknown"
            
            model_usage_counter[model_type] += 1
            prediction_source_counter[model_type] += 1  # Using model_type as prediction source
            probability_source_counter[prob_source] += 1
            
            # Track this round in history
            game_history.append({
                "user_move": move,
                "bot_move": last_bot_move,
                "result": response.get("result"),  # "win", "lose", "draw" from user perspective
                "user_score": response.get("user_score", 0.0),
                "bot_score": response.get("bot_score", 0.0),
            })

            if verbose:
                print(
                    f"Round {round_num + 1}: user {move} vs bot {response.get('bot_move', '?')}"
                )
                print(
                    f"Score: user {response['user_score']:.1f} - bot {response['bot_score']:.1f}"
                )
                print(
                    f"Model: {model_type} | Probability source: {prob_source}"
                )

            if response.get("finished", False):
                result = {
                    "user_bot": user_bot,
                    "server_bot": server_bot,
                    "difficulty_mode": difficulty_mode,
                    "winner": response.get("winner"),
                    "rounds": round_num + 1,
                    "user_score": response["user_score"],
                    "bot_score": response["bot_score"],
                    "game_id": game_id,
                    "session_id": session_id,
                    "target_score": target_score,
                    "model_usage": dict(model_usage_counter),
                    "prediction_sources": dict(prediction_source_counter),
                    "probability_sources": dict(probability_source_counter),
                    "model_alias": model_alias,
                    "model_type": model_type,
                    "alias_counter": dict(alias_counter),
                    "ab_assignment": response.get("ab_assignment"),
                }

                if verbose:
                    print(
                        f"Game ended after {round_num + 1} rounds. Winner: {response.get('winner')}"
                    )

                return result

        print(f"Game {game_id} timed out after {max_rounds} rounds")
        return {
            "user_bot": user_bot,
            "server_bot": server_bot,
            "winner": "timeout",
            "rounds": max_rounds,
            "user_score": 0,
            "bot_score": 0,
            "game_id": game_id,
            "session_id": session_id,
            "target_score": target_score,
        }

    except Exception as e:
        print(f"Error playing game: {e}")
        return {
            "user_bot": user_bot,
            "server_bot": server_bot,
            "winner": "error",
            "rounds": 0,
            "user_score": 0,
            "bot_score": 0,
            "game_id": "unknown",
            "session_id": None,
            "target_score": None,
            "model_usage": {},
            "prediction_sources": {},
            "probability_sources": {},
        }


def simulate_games(
    api: APIClient,
    user_bot: str,
    server_bot: str,
    num_games: int,
    verbose: bool = False,
    *,
    easy_mode: bool = False,
    difficulty_mode: str = "normal",
):
    """Simulate multiple games and report results"""
    print(f"Simulating {num_games} games: {user_bot} vs {server_bot}")
    print("-" * 50)
    
    results = []
    wins = {"user": 0, "bot": 0, "tie": 0, "timeout": 0, "error": 0}
    aggregate_model_usage: Counter[str] = Counter()
    aggregate_prediction_sources: Counter[str] = Counter()
    aggregate_probability_sources: Counter[str] = Counter()
    aggregate_aliases: Counter[str] = Counter()
    aggregate_model_types: Counter[str] = Counter()
    
    for i in range(num_games):
        if verbose:
            print(f"\nGame {i+1}/{num_games}:")
        
        result = play_single_game(
            api,
            user_bot,
            server_bot,
            verbose,
            easy_mode=easy_mode,
            difficulty_mode=difficulty_mode,
        )
        results.append(result)
        aggregate_model_usage.update(result.get("model_usage", {}))
        aggregate_prediction_sources.update(result.get("prediction_sources", {}))
        aggregate_probability_sources.update(result.get("probability_sources", {}))
        aggregate_aliases.update(result.get("alias_counter", {}))
        if result.get("model_type"):
            aggregate_model_types.update([result["model_type"]])
        
        # Count wins
        winner = result["winner"]
        if winner == "user":
            wins["user"] += 1
        elif winner == "bot":
            wins["bot"] += 1
        elif winner in ("tie", None):
            wins["tie"] += 1
        elif winner == "timeout":
            wins["timeout"] += 1
        else:
            wins["error"] += 1
    
    # Summary
    print("\nSimulation Results:")
    print(f"{user_bot} wins: {wins['user']}")
    print(f"{server_bot} wins: {wins['bot']}")
    print(f"Ties: {wins['tie']}")
    if wins["timeout"]:
        print(f"Timeouts: {wins['timeout']}")
    if wins["error"]:
        print(f"Errors: {wins['error']}")
    
    if num_games > 0:
        print(f"{user_bot} win rate: {wins['user']/num_games:.2%}")

    if aggregate_model_usage:
        print("Model usage counts:", dict(aggregate_model_usage))
    if aggregate_prediction_sources:
        print("Prediction sources:", dict(aggregate_prediction_sources))
    if aggregate_probability_sources:
        print("Probability sources:", dict(aggregate_probability_sources))
    if aggregate_aliases:
        print("Alias exposure:", dict(aggregate_aliases))
    if aggregate_model_types:
        print("Model types:", dict(aggregate_model_types))
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Bot vs Bot Game Simulation")
    parser.add_argument("--user", required=True, help="User bot policy")
    parser.add_argument("--bot", required=True, help="Server bot policy")
    parser.add_argument("--games", type=int, default=5, help="Number of games to simulate")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--url", default=API_BASE_URL, help="API base URL")
    parser.add_argument(
        "--easy-mode",
        action="store_true",
        help="(Retired) Request legacy easy mode handicap",
    )
    
    args = parser.parse_args()
    
    # Determine difficulty mode based on bot name or explicit flag
    difficulty_mode = "easy" if args.bot.endswith("_easy") else "normal"
    server_policy = args.bot[:-5] if difficulty_mode == "easy" else args.bot

    if args.easy_mode or difficulty_mode == "easy":
        print("[warn] Easy difficulty is retired; forcing standard mode for server bot")
        difficulty_mode = "normal"
        server_policy = args.bot[:-5] if args.bot.endswith("_easy") else args.bot

    effective_easy_mode = False

    api = APIClient(args.url)
    
    try:
        # Test API connection with modern and legacy health endpoints
        for endpoint in HEALTH_ENDPOINTS:
            try:
                response = requests.get(f"{args.url}{endpoint}", timeout=5)
                response.raise_for_status()
                break
            except Exception:
                continue
        else:
            raise RuntimeError("No health endpoint responded successfully")
        print(f"✓ Connected to RPS API at {args.url}")
    except Exception as e:
        print(f"✗ Failed to connect to API at {args.url}: {e}")
        sys.exit(1)
    
    simulate_games(
        api,
        args.user,
        server_policy,
        args.games,
        args.verbose,
    easy_mode=effective_easy_mode,
        difficulty_mode=difficulty_mode,
    )

if __name__ == "__main__":
    main()