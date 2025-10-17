"""Internal endpoints for automated promotion bookkeeping."""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, ValidationError

from app.auto_promotion import normalise_model_type
from app.db import DB
from app.metrics import (
    record_alias_reorder,
    record_promotion_evaluation,
    update_alias_rank,
    update_cycles_since_swap,
    update_games_since_swap,
)
from app.promotion_store import TickerSnapshot, calculate_ticker_snapshot

router = APIRouter(prefix="/internal/promotion", tags=["Promotion"])

_ALLOWED_DECISIONS = {
    "swap_production_b",
    "retain_production",
    "insufficient_data",
    "alias_reorder",
}

_DECISION_ORDER = [
    "swap_production_b",
    "retain_production",
    "insufficient_data",
    "alias_reorder",
]

_KNOWN_MODELS = [
    "xgboost",
    "feedforward_nn",
    "multinomial_logistic",
]

_ALLOWED_ALIASES = {"B", "shadow1", "shadow2"}


class AliasRankingPayload(BaseModel):
    alias: str = Field(..., description="Alias receiving the supplied rank")
    rank: int = Field(..., ge=1, le=3, description="Rank assigned to alias (1=best)")
    source_alias: Optional[str] = Field(
        default=None,
        description="Original alias that supplied the metrics for this rank",
    )
    accuracy: Optional[float] = Field(
        default=None,
        description="Observed action accuracy for the ranked alias (0-1 range)",
    )
    total_predictions: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of action predictions observed for the ranked alias",
    )


class AliasAccuracyPayload(BaseModel):
    alias: str = Field(..., description="Alias whose action accuracy was measured")
    source_alias: Optional[str] = Field(
        default=None,
        description="Original alias that supplied the underlying metrics",
    )
    accuracy: Optional[float] = Field(
        default=None,
        description="Observed action accuracy for the alias (0-1 range)",
    )
    total_predictions: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of action predictions observed while gathering metrics",
    )


class AliasAssignmentPayload(BaseModel):
    alias: str = Field(..., description="Alias being (re)assigned")
    version: Optional[str] = Field(
        default=None,
        description="Target MLflow model version after the promotion cycle",
    )
    source_alias: Optional[str] = Field(
        default=None,
        description="Alias that supplied the artifacts applied to this assignment",
    )
    previous_version: Optional[str] = Field(
        default=None,
        description="Previously active MLflow model version before reassignment",
    )
    changed: bool = Field(
        False,
        description="Whether the alias version changed compared to the previous assignment",
    )


class PromotionReportPayload(BaseModel):
    model_type: str = Field(..., description="Model type identifier (xgboost/feedforward_nn/multinomial_logistic)")
    decision: str = Field(..., description="Outcome of the Production vs B comparison")
    z_statistic: Optional[float] = Field(default=None, description="Computed z statistic for the hypothesis test")
    p_value: Optional[float] = Field(default=None, description="One-sided p-value for the hypothesis test")
    production_wins: Optional[int] = Field(default=None, ge=0)
    production_total_games: Optional[int] = Field(default=None, ge=0)
    b_wins: Optional[int] = Field(default=None, ge=0)
    b_total_games: Optional[int] = Field(default=None, ge=0)
    production_games_since_swap: Optional[int] = Field(
        default=None,
        ge=0,
        description="Ticker value for Production games since the last Production/B swap",
    )
    b_games_since_swap: Optional[int] = Field(
        default=None,
        ge=0,
        description="Ticker value for B games since the last Production/B swap",
    )
    promotion_cycles_since_swap: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of auto-promotion cycles evaluated since the last swap",
    )
    reorder_applied: bool = Field(False, description="Whether challenger aliases were reordered")
    alias_rankings: List[AliasRankingPayload] = Field(
        default_factory=list,
        description="Ranks applied to challenger aliases after reordering",
    )
    alias_accuracies: List[AliasAccuracyPayload] = Field(
        default_factory=list,
        description="Observed action accuracy for each alias during the promotion window",
    )
    alias_assignments: List[AliasAssignmentPayload] = Field(
        default_factory=list,
        description="Final alias-to-version assignments after the promotion cycle",
    )
    reason: Optional[str] = Field(default=None, description="Human-readable explanation of the decision")
    source: str = Field("auto_promotion", description="Origin of the report message")
    created_ts_utc: Optional[str] = Field(
        default=None,
        description="Optional override for the ledger timestamp (ISO 8601, defaults to now)",
    )


class PromotionHistoryRecord(BaseModel):
    id: int
    created_ts_utc: str
    model_type: str
    decision: str
    z_statistic: Optional[float]
    p_value: Optional[float]
    production_wins: Optional[int]
    production_total_games: Optional[int]
    production_win_rate: Optional[float]
    b_wins: Optional[int]
    b_total_games: Optional[int]
    b_win_rate: Optional[float]
    production_games_since_swap: Optional[int]
    b_games_since_swap: Optional[int]
    promotion_cycles_since_swap: Optional[int]
    reorder_applied: bool
    alias_rankings: List[AliasRankingPayload]
    alias_accuracies: List[AliasAccuracyPayload]
    alias_assignments: List[AliasAssignmentPayload]
    reason: Optional[str]
    source: str
    rank1_summary: Optional[str] = Field(default=None, description="Formatted summary for rank 1 alias")
    rank2_summary: Optional[str] = Field(default=None, description="Formatted summary for rank 2 alias")
    rank3_summary: Optional[str] = Field(default=None, description="Formatted summary for rank 3 alias")
    step2_order: Optional[str] = Field(default=None, description="Alias order string produced during challenger reseeding")
    step2_accuracy: Optional[str] = Field(default=None, description="Comma-separated accuracy summary for challenger aliases")
    step2_sources: Optional[str] = Field(default=None, description="Comma-separated source alias summary for challenger aliases")


class PromotionEventTotalsRow(BaseModel):
    model: str = Field(..., description="Model identifier or ALL for fleet aggregate")
    period_start: str = Field(..., description="Inclusive UTC timestamp for the reporting window start")
    period_end: str = Field(..., description="Exclusive UTC timestamp for the reporting window end")
    swap_production_b: int = Field(..., ge=0, description="Number of Production/B swaps performed")
    retain_production: int = Field(..., ge=0, description="Number of cycles that retained the incumbent Production alias")
    insufficient_data: int = Field(..., ge=0, description="Cycles that exited due to insufficient comparison data")
    alias_reorder: int = Field(..., ge=0, description="Cycles that only reordered challenger aliases")
    total_events: int = Field(..., ge=0, description="Total number of promotion cycles observed in the window")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalise_iso_timestamp(value: str) -> str:
    candidate = value.strip()
    if not candidate:
        raise HTTPException(status_code=400, detail="ISO 8601 timestamp cannot be empty")
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:  # noqa: B904
        raise HTTPException(status_code=400, detail="Invalid ISO 8601 timestamp supplied") from exc
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_ratio(numerator: Optional[int], denominator: Optional[int]) -> Optional[float]:
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def persist_promotion_event(
    model_type: str,
    payload: PromotionReportPayload,
    decision: str,
    *,
    production_games_since_swap: Optional[int],
    b_games_since_swap: Optional[int],
    promotion_cycles_since_swap: Optional[int],
) -> int:
    alias_rankings_data = [item.model_dump(mode="python") for item in payload.alias_rankings]
    alias_rankings_json = json.dumps(alias_rankings_data) if alias_rankings_data else None
    alias_accuracies_data = [item.model_dump(mode="python") for item in payload.alias_accuracies]
    alias_accuracies_json = json.dumps(alias_accuracies_data) if alias_accuracies_data else None
    alias_assignments_data = [item.model_dump(mode="python") for item in payload.alias_assignments]
    alias_assignments_json = json.dumps(alias_assignments_data) if alias_assignments_data else None
    raw_payload_json = json.dumps(payload.model_dump(mode="python"))

    created_ts = _utc_now_iso() if payload.created_ts_utc is None else _normalise_iso_timestamp(payload.created_ts_utc)

    cursor = DB.execute(
        """
        INSERT INTO promotion_events (
            created_ts_utc,
            model_type,
            decision,
            z_statistic,
            p_value,
            production_wins,
            production_total_games,
            b_wins,
            b_total_games,
            reorder_applied,
            alias_rankings_json,
            alias_accuracies_json,
            alias_assignments_json,
            production_games_since_swap,
            b_games_since_swap,
            promotion_cycles_since_swap,
            reason,
            source,
            payload_json
        )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            created_ts,
            model_type,
            decision,
            payload.z_statistic,
            payload.p_value,
            payload.production_wins,
            payload.production_total_games,
            payload.b_wins,
            payload.b_total_games,
            1 if payload.reorder_applied else 0,
            alias_rankings_json,
            alias_accuracies_json,
            alias_assignments_json,
            production_games_since_swap,
            b_games_since_swap,
            promotion_cycles_since_swap,
            payload.reason,
            payload.source,
            raw_payload_json,
        ),
    )
    DB.commit()
    return int(cursor.lastrowid)


@router.post("/report")
def submit_promotion_report(payload: PromotionReportPayload) -> dict:
    """Record the outcome of an automated promotion cycle in Prometheus metrics."""

    model_type = normalise_model_type(payload.model_type)
    decision = payload.decision.strip().lower()

    if decision not in _ALLOWED_DECISIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported decision '{payload.decision}'")

    # Persist statistics to Prometheus metrics.
    record_promotion_evaluation(
        model=model_type,
        decision=decision,
        z_statistic=payload.z_statistic,
        p_value=payload.p_value,
    )

    record_alias_reorder(model=model_type, performed=payload.reorder_applied)

    seen_aliases: set[str] = set()
    for item in payload.alias_rankings:
        alias = item.alias.strip()
        if alias not in _ALLOWED_ALIASES:
            raise HTTPException(status_code=400, detail=f"Unsupported alias '{alias}' in rankings")
        if alias in seen_aliases:
            raise HTTPException(status_code=400, detail=f"Alias '{alias}' listed multiple times")
        seen_aliases.add(alias)
        update_alias_rank(model=model_type, alias=alias, rank=item.rank)

    if decision == "swap_production_b":
        ticker_snapshot = TickerSnapshot(0, 0, 0)
    elif (
        payload.production_games_since_swap is not None
        and payload.b_games_since_swap is not None
        and payload.promotion_cycles_since_swap is not None
    ):
        ticker_snapshot = TickerSnapshot(
            production_games_since_swap=int(payload.production_games_since_swap),
            b_games_since_swap=int(payload.b_games_since_swap),
            cycles_since_swap=int(payload.promotion_cycles_since_swap),
        )
    else:
        ticker_snapshot = calculate_ticker_snapshot(
            model_type,
            decision,
            production_wins=payload.production_wins,
            production_total_games=payload.production_total_games,
            b_wins=payload.b_wins,
            b_total_games=payload.b_total_games,
        )

    update_games_since_swap(model_type, "Production", ticker_snapshot.production_games_since_swap)
    update_games_since_swap(model_type, "B", ticker_snapshot.b_games_since_swap)
    update_cycles_since_swap(model_type, ticker_snapshot.cycles_since_swap)

    event_id = persist_promotion_event(
        model_type,
        payload,
        decision,
        production_games_since_swap=ticker_snapshot.production_games_since_swap,
        b_games_since_swap=ticker_snapshot.b_games_since_swap,
        promotion_cycles_since_swap=ticker_snapshot.cycles_since_swap,
    )

    return {
        "success": True,
        "model_type": model_type,
        "decision": decision,
        "reorder_applied": payload.reorder_applied,
        "alias_rankings": [item.model_dump(mode="python") for item in payload.alias_rankings],
        "alias_accuracies": [item.model_dump(mode="python") for item in payload.alias_accuracies],
        "alias_assignments": [item.model_dump(mode="python") for item in payload.alias_assignments],
        "reason": payload.reason,
        "production_games_since_swap": ticker_snapshot.production_games_since_swap,
        "b_games_since_swap": ticker_snapshot.b_games_since_swap,
        "promotion_cycles_since_swap": ticker_snapshot.cycles_since_swap,
        "event_id": event_id,
    }


@router.get("/event_totals", response_model=List[PromotionEventTotalsRow])
def fetch_promotion_event_totals(
    days: int = Query(7, ge=1, le=90, description="Length of the reporting window in days"),
) -> List[PromotionEventTotalsRow]:
    """Summarise promotion cycle outcomes over a recent rolling window."""

    period_end = datetime.now(timezone.utc)
    period_start = period_end - timedelta(days=days)
    period_start_iso = period_start.isoformat().replace("+00:00", "Z")
    period_end_iso = period_end.isoformat().replace("+00:00", "Z")

    def _empty_counts() -> dict[str, int]:
        return {decision: 0 for decision in _DECISION_ORDER}

    totals_by_model: dict[str, dict[str, int]] = defaultdict(_empty_counts)

    cursor = DB.execute(
        """
        SELECT model_type, decision, COUNT(*)
        FROM promotion_events
        WHERE created_ts_utc >= ? AND created_ts_utc < ?
        GROUP BY model_type, decision
        """,
        (period_start_iso, period_end_iso),
    )

    for model_type, decision, count in cursor.fetchall():
        if decision not in _ALLOWED_DECISIONS:
            continue
        totals_by_model[model_type][decision] = int(count)

    for model in _KNOWN_MODELS:
        totals_by_model.setdefault(model, _empty_counts())

    overall_counts = {decision: 0 for decision in _DECISION_ORDER}
    rows: List[PromotionEventTotalsRow] = []

    ordered_models: List[str] = []
    for model in _KNOWN_MODELS:
        if model in totals_by_model:
            ordered_models.append(model)
    for model in sorted(totals_by_model):
        if model not in ordered_models:
            ordered_models.append(model)

    for model in ordered_models:
        counts = totals_by_model[model]
        for decision in _DECISION_ORDER:
            overall_counts[decision] += counts.get(decision, 0)
        total_events = sum(counts.values())
        rows.append(
            PromotionEventTotalsRow(
                model=model,
                period_start=period_start_iso,
                period_end=period_end_iso,
                swap_production_b=counts.get("swap_production_b", 0),
                retain_production=counts.get("retain_production", 0),
                insufficient_data=counts.get("insufficient_data", 0),
                alias_reorder=counts.get("alias_reorder", 0),
                total_events=total_events,
            )
        )

    overall_total = sum(overall_counts.values())
    rows.append(
        PromotionEventTotalsRow(
            model="ALL",
            period_start=period_start_iso,
            period_end=period_end_iso,
            swap_production_b=overall_counts["swap_production_b"],
            retain_production=overall_counts["retain_production"],
            insufficient_data=overall_counts["insufficient_data"],
            alias_reorder=overall_counts["alias_reorder"],
            total_events=overall_total,
        )
    )

    return rows


@router.get("/history", response_model=List[PromotionHistoryRecord])
def fetch_promotion_history(
    limit: int = Query(20, ge=1, le=200, description="Maximum number of records to return"),
    model_type: Optional[str] = Query(None, description="Optional filter for model family"),
    decision: Optional[str] = Query(None, description="Filter by decision outcome"),
    since: Optional[str] = Query(None, description="Restrict to events created at or after this ISO 8601 timestamp"),
) -> List[PromotionHistoryRecord]:
    filters: List[str] = []
    params: List[object] = []

    if model_type:
        filters.append("model_type = ?")
        params.append(normalise_model_type(model_type))

    if decision:
        candidate = decision.strip().lower()
        if candidate not in _ALLOWED_DECISIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported decision '{decision}'")
        filters.append("decision = ?")
        params.append(candidate)

    if since:
        filters.append("created_ts_utc >= ?")
        params.append(_normalise_iso_timestamp(since))

    query = (
        "SELECT id, created_ts_utc, model_type, decision, z_statistic, p_value, "
        "production_wins, production_total_games, b_wins, b_total_games, "
        "production_games_since_swap, b_games_since_swap, promotion_cycles_since_swap, "
        "reorder_applied, alias_rankings_json, alias_accuracies_json, alias_assignments_json, "
        "reason, source "
        "FROM promotion_events"
    )

    if filters:
        query += " WHERE " + " AND ".join(filters)

    query += " ORDER BY created_ts_utc DESC LIMIT ?"
    params.append(limit)

    cursor = DB.execute(query, params)
    rows = cursor.fetchall()

    history: List[PromotionHistoryRecord] = []

    def _format_alias_summary(entry: AliasRankingPayload) -> str:
        label = entry.alias
        if entry.source_alias and entry.source_alias != entry.alias:
            label = f"{entry.alias} <- {entry.source_alias}"

        extras: List[str] = []
        if entry.total_predictions:
            plural = "pred" if entry.total_predictions == 1 else "preds"
            extras.append(f"{entry.total_predictions} {plural}")
        if entry.accuracy is not None:
            extras.append(f"{entry.accuracy * 100:.1f}%")

        if extras:
            return f"{label} ({', '.join(extras)})"
        return label

    def _summarise_rankings(entries: List[AliasRankingPayload]) -> tuple[str, str, str]:
        if not entries:
            return "-", "-", "-"

        order = " > ".join(item.alias for item in entries) or "-"

        accuracy_parts: List[str] = []
        source_parts: List[str] = []
        for item in entries:
            accuracy = "n/a" if item.accuracy is None else f"{item.accuracy * 100:.1f}%"
            accuracy_parts.append(f"{item.alias}: {accuracy}")
            source_parts.append(f"{item.alias} src={item.source_alias or '-'}")

        accuracy_summary = ", ".join(accuracy_parts) or "-"
        source_summary = ", ".join(source_parts) or "-"

        return order, accuracy_summary, source_summary

    for row in rows:
        (
            event_id,
            created_ts_utc,
            model,
            decision_value,
            z_statistic,
            p_value,
            production_wins,
            production_total,
            b_wins,
            b_total,
            production_games_since_swap,
            b_games_since_swap,
            promotion_cycles_since_swap,
            reorder_flag,
            alias_rankings_json,
            alias_accuracies_json,
            alias_assignments_json,
            reason,
            source,
        ) = row

        alias_rankings: List[AliasRankingPayload] = []
        if alias_rankings_json:
            try:
                raw_rankings = json.loads(alias_rankings_json)
                alias_rankings = [AliasRankingPayload.model_validate(item) for item in raw_rankings]
            except (json.JSONDecodeError, ValidationError):  # pragma: no cover - safeguard
                alias_rankings = []

        alias_accuracies: List[AliasAccuracyPayload] = []
        if alias_accuracies_json:
            try:
                raw_accuracies = json.loads(alias_accuracies_json)
                alias_accuracies = [AliasAccuracyPayload.model_validate(item) for item in raw_accuracies]
            except (json.JSONDecodeError, ValidationError):  # pragma: no cover - safeguard
                alias_accuracies = []

        alias_assignments: List[AliasAssignmentPayload] = []
        if alias_assignments_json:
            try:
                raw_assignments = json.loads(alias_assignments_json)
                alias_assignments = [AliasAssignmentPayload.model_validate(item) for item in raw_assignments]
            except (json.JSONDecodeError, ValidationError):  # pragma: no cover - safeguard
                alias_assignments = []

        rank_summaries = [_format_alias_summary(item) for item in alias_rankings]
        step2_order, step2_accuracy, step2_sources = _summarise_rankings(alias_rankings)

        history.append(
            PromotionHistoryRecord(
                id=event_id,
                created_ts_utc=created_ts_utc,
                model_type=model,
                decision=decision_value,
                z_statistic=z_statistic,
                p_value=p_value,
                production_wins=production_wins,
                production_total_games=production_total,
                production_win_rate=_safe_ratio(production_wins, production_total),
                b_wins=b_wins,
                b_total_games=b_total,
                b_win_rate=_safe_ratio(b_wins, b_total),
                production_games_since_swap=production_games_since_swap,
                b_games_since_swap=b_games_since_swap,
                promotion_cycles_since_swap=promotion_cycles_since_swap,
                reorder_applied=bool(reorder_flag),
                alias_rankings=alias_rankings,
                alias_accuracies=alias_accuracies,
                alias_assignments=alias_assignments,
                reason=reason,
                source=source,
                rank1_summary=rank_summaries[0] if len(rank_summaries) > 0 else None,
                rank2_summary=rank_summaries[1] if len(rank_summaries) > 1 else None,
                rank3_summary=rank_summaries[2] if len(rank_summaries) > 2 else None,
                step2_order=step2_order,
                step2_accuracy=step2_accuracy,
                step2_sources=step2_sources,
            )
        )

    return history
