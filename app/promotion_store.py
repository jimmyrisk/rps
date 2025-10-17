"""Helpers for persisting and deriving promotion ticker state.

This module centralises the bookkeeping required to keep the "ticker" style
metrics in sync with production behaviour.  The tickers track how many games and
promotion cycles have elapsed since the last Production ↔ B swap so both the
FastAPI application and the automation scripts rely on a single source of
truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.db import DB


@dataclass(frozen=True)
class PromotionEventBaseline:
	"""Snapshot of the counters captured at the last Production/B swap."""

	event_id: int
	production_wins: int
	production_total_games: int
	b_wins: int
	b_total_games: int


@dataclass(frozen=True)
class TickerSnapshot:
	"""Derived ticker values for the current promotion cycle."""

	production_games_since_swap: int
	b_games_since_swap: int
	cycles_since_swap: int


def _safe_int(value: Optional[int]) -> int:
	if value is None:
		return 0
	return int(value)


def _fetch_last_swap_event(model_type: str) -> Optional[PromotionEventBaseline]:
	cursor = DB.execute(
		"""
		SELECT id, production_wins, production_total_games, b_wins, b_total_games
		FROM promotion_events
		WHERE model_type = ? AND decision = 'swap_production_b'
		ORDER BY created_ts_utc DESC
		LIMIT 1
		""",
		(model_type,),
	)
	row = cursor.fetchone()
	if row is None:
		return None
	return PromotionEventBaseline(
		event_id=int(row[0]),
		production_wins=_safe_int(row[1]),
		production_total_games=_safe_int(row[2]),
		b_wins=_safe_int(row[3]),
		b_total_games=_safe_int(row[4]),
	)


def _count_events_after_id(model_type: str, last_event_id: Optional[int]) -> int:
	if last_event_id is None:
		cursor = DB.execute(
			"SELECT COUNT(*) FROM promotion_events WHERE model_type = ?",
			(model_type,),
		)
	else:
		cursor = DB.execute(
			"SELECT COUNT(*) FROM promotion_events WHERE model_type = ? AND id > ?",
			(model_type, last_event_id),
		)
	row = cursor.fetchone()
	return int(row[0] or 0)


def _delta(current: Optional[int], baseline: Optional[int]) -> int:
	current_value = _safe_int(current) if current is not None else 0
	if baseline is None:
		return max(current_value, 0)
	delta = current_value - _safe_int(baseline)
	if delta < 0:
		return max(current_value, 0)
	return delta


def calculate_ticker_snapshot(
	model_type: str,
	decision: str,
	*,
	production_wins: Optional[int],
	production_total_games: Optional[int],
	b_wins: Optional[int],
	b_total_games: Optional[int],
) -> TickerSnapshot:
	"""Return games + cycles since the last Production/B promotion.

	The counters advance on every auto-promotion cycle and reset to zero when a
	fresh Production/B swap is recorded.  Challenger-only reorders do not reset
	the tickers; they continue counting until a swap occurs.
	"""

	if decision == "swap_production_b":
		return TickerSnapshot(0, 0, 0)

	baseline = _fetch_last_swap_event(model_type)
	baseline_id = baseline.event_id if baseline else None
	cycles_before_current = _count_events_after_id(model_type, baseline_id)

	production_games_since_swap = _delta(
		production_total_games,
		baseline.production_total_games if baseline else None,
	)
	b_games_since_swap = _delta(
		b_total_games,
		baseline.b_total_games if baseline else None,
	)

	return TickerSnapshot(
		production_games_since_swap=production_games_since_swap,
		b_games_since_swap=b_games_since_swap,
		cycles_since_swap=cycles_before_current + 1,
	)


def fetch_last_swap_baseline(model_type: str) -> Optional[PromotionEventBaseline]:
	"""Expose the last-swap snapshot for callers outside this module."""

	return _fetch_last_swap_event(model_type)


__all__ = [
	"PromotionEventBaseline",
	"TickerSnapshot",
	"calculate_ticker_snapshot",
	"fetch_last_swap_baseline",
]
