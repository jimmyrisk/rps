"""Shared helper functions for filtering player names in analytics queries.

This module mirrors the filtering logic used by the frontend so that
leaderboards and win-rate calculations consistently exclude synthetic
players (legacy bots, simulation harnesses, etc.).
"""
from __future__ import annotations

import re
from typing import Optional, Tuple

# Set of deterministic legacy bot usernames that should never appear in
# human-facing leaderboards. These match the policies defined in
# ``app/legacy_models.py`` as well as the JavaScript frontend filter set.
LEGACY_PLAYER_NAMES = {
    "ace",
    "bob",
    "cal",
    "dan",
    "edd",
    "fox",
    "gus",
    "hal",
}

# Pre-compiled regular expressions that capture the simulation naming
# patterns produced by historical load-testing scripts (e.g. ``sim1``,
# ``sim[42]``, ``sim(003)``). The goal is to stay in lock-step with the
# frontend's filtering logic, so we keep the patterns intentionally
# permissive.
_SIM_EXACT_PATTERN = re.compile(r"^sim(?:\[\d+\]|\d+|\(\d+\))$", re.IGNORECASE)
_SIM_NUMERIC_PATTERN = re.compile(r"^sim[\w-]*\d", re.IGNORECASE)


def _normalized(name: Optional[str]) -> str:
    """Return a stripped, lowercase variant of ``name`` for comparisons."""
    if not name:
        return ""
    return name.strip().lower()


def is_legacy_player_name(name: Optional[str]) -> bool:
    """Return ``True`` if the supplied name matches a legacy bot alias."""
    return _normalized(name) in LEGACY_PLAYER_NAMES


def is_simulated_player_name(name: Optional[str]) -> bool:
    """Return ``True`` for simulation/test harness player identifiers."""
    raw = (name or "").strip()
    if not raw:
        return False

    lower = raw.lower()
    if _SIM_EXACT_PATTERN.match(raw):
        return True
    if _SIM_NUMERIC_PATTERN.match(raw):
        return True
    if lower.startswith("simp") and any(ch.isdigit() for ch in lower):
        return True
    return lower.startswith("sim[")


def player_exclusion_reason(name: Optional[str]) -> Optional[str]:
    """Return the exclusion reason for ``name`` or ``None`` if allowed."""
    if is_legacy_player_name(name):
        return "legacy"
    if is_simulated_player_name(name):
        return "simulated"
    return None


def should_exclude_player(name: Optional[str]) -> bool:
    """Return ``True`` when the player should be omitted from analytics."""
    return player_exclusion_reason(name) is not None


def summarize_exclusion(name: Optional[str]) -> Tuple[str, str]:
    """Return a tuple of (reason, canonical_name) for reporting purposes."""
    reason = player_exclusion_reason(name)
    canonical = (name or "").strip()
    return reason or "", canonical
