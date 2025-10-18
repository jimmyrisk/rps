#!/usr/bin/env python3
"""Automated alias promotion orchestrator.

This script runs immediately after a training sweep to evaluate challenger
performance and reshuffle aliases when the data supports a promotion.

Steps per model family (xgboost, feedforward_nn, multinomial_logistic):
1. Pull the latest Prometheus counters from the FastAPI service.
2. Run a one-sided 2-sample z test on Production vs B win rates.
3. Swap Production/B aliases in MLflow when B is statistically superior.
4. Re-rank challenger slots (B, shadow1, shadow2) so B always represents the
   most accurate contender based on action accuracy since the last pod restart.
5. Report the outcome back to the application so Grafana can visualise the
   hypothesis tests, decisions, and rank ordering.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import requests
from mlflow.tracking import MlflowClient  # type: ignore
from prometheus_client.parser import text_string_to_metric_families  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.auto_promotion import (  # noqa: E402
    AliasAccuracyStats,
    AliasGameStats,
    PromotionTestResult,
    evaluate_promotion_decision,
    normalise_model_type,
)
from app.config import set_mlflow_tracking_uri_if_needed  # noqa: E402

# Mapping from public model types to MLflow registered model names.
MODEL_REGISTRY_NAMES: Dict[str, str] = {
    "xgboost": "rps_bot_xgboost",
    "feedforward_nn": "rps_bot_feedforward",
    "multinomial_logistic": "rps_bot_mnlogit",
}

ALL_ALIASES: Sequence[str] = ("Production", "B", "shadow1", "shadow2")
CHALLENGER_ALIASES: Sequence[str] = ("B", "shadow1", "shadow2")

DEFAULT_METRICS_ENDPOINTS: Sequence[str] = (
    "http://rps-app.mlops-poc.svc.cluster.local:8080/metrics",
    "http://localhost:8080/metrics",
    "https://mlops-rps.uk/metrics",
)

DEFAULT_PROMOTION_ENDPOINTS: Sequence[str] = (
    "http://rps-app.mlops-poc.svc.cluster.local:8080/internal/promotion/report",
    "http://localhost:8080/internal/promotion/report",
    "https://mlops-rps.uk/internal/promotion/report",
)

DEFAULT_RELOAD_ENDPOINTS: Sequence[str] = (
    "http://rps-app.mlops-poc.svc.cluster.local:8080/models/reload",
    "http://localhost:8080/models/reload",
    "https://mlops-rps.uk/models/reload",
)

DEFAULT_HISTORY_ENDPOINTS: Sequence[str] = (
    "http://rps-app.mlops-poc.svc.cluster.local:8080/internal/promotion/history",
    "http://localhost:8080/internal/promotion/history",
    "https://mlops-rps.uk/internal/promotion/history",
)

logger = logging.getLogger("auto_promotion")


@dataclass
class AliasMetrics:
    """Container for all metrics needed to evaluate promotions."""

    game_stats: Dict[str, AliasGameStats]
    accuracy_stats: Dict[str, AliasAccuracyStats]


@dataclass
class AliasAssignment:
    """Tracks current and original provenance for each alias."""

    version: str
    source_alias: str


@dataclass(frozen=True)
class PromotionBaseline:
    """Counters captured at the time of the last Production/B swap."""

    production_wins: int
    production_total_games: int
    b_wins: int
    b_total_games: int


@dataclass(frozen=True)
class PromotionTickerState:
    """Ticker values observed prior to evaluating the current cycle."""

    baseline: Optional[PromotionBaseline]
    cycles_since_swap: int


@dataclass
class PromotionOutcome:
    model_type: str
    test_result: PromotionTestResult
    alias_rankings: List[Dict[str, object]]
    alias_accuracies: List[Dict[str, object]]
    alias_assignments: List[Dict[str, object]]
    swap_performed: bool
    reorder_applied: bool
    production_games_since_swap: int
    b_games_since_swap: int
    cycles_since_swap: int
    message: str


@dataclass(frozen=True)
class ChallengerCandidate:
    """Represents a challenger alias evaluated for reseeding."""

    source_alias: str
    accuracy: float
    total_predictions: int
    current_alias: str
    position: int


def _build_endpoint_list(env_var: str, defaults: Sequence[str]) -> List[str]:
    custom = os.getenv(env_var)
    if custom:
        endpoints = [entry.strip() for entry in custom.split(",") if entry.strip()]
        if endpoints:
            return endpoints
    return list(defaults)


def fetch_metrics_text(endpoints: Iterable[str], timeout: float = 10.0) -> str:
    last_error: Optional[Exception] = None
    for url in endpoints:
        try:
            logger.debug("Fetching metrics from %s", url)
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                logger.info("Pulled metrics from %s", url)
                return response.text
            logger.warning("Metrics endpoint %s returned %s", url, response.status_code)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.debug("Metrics fetch failed for %s: %s", url, exc)
    raise RuntimeError(f"Unable to fetch Prometheus metrics: {last_error}")


def parse_metrics(metrics_text: str) -> Dict[str, list]:
    parsed: Dict[str, list] = defaultdict(list)
    for family in text_string_to_metric_families(metrics_text):
        parsed[family.name].extend(family.samples)
    return parsed


def _sum_metric(samples: Iterable, **label_filters: str) -> float:
    total = 0.0
    for sample in samples:
        if all(sample.labels.get(k) == v for k, v in label_filters.items()):
            total += float(sample.value)
    return total


def extract_alias_metrics(parsed: Dict[str, list], model_type: str) -> AliasMetrics:
    game_stats: Dict[str, AliasGameStats] = {}
    accuracy_stats: Dict[str, AliasAccuracyStats] = {}

    def _samples(*candidate_names: str) -> list:
        for name in candidate_names:
            if name in parsed:
                return parsed[name]
        return []

    game_wins_samples = _samples("rps_model_game_wins_by_alias_total", "rps_model_game_wins_by_alias")
    game_total_samples = _samples("rps_model_games_by_alias_total", "rps_model_games_by_alias")
    correct_samples = _samples("rps_model_correct_predictions_by_alias_total", "rps_model_correct_predictions_by_alias")
    pred_samples = _samples("rps_model_predictions_by_alias_total", "rps_model_predictions_by_alias")

    for alias in ALL_ALIASES:
        wins = _sum_metric(game_wins_samples, model=model_type, alias=alias)
        total_games = _sum_metric(game_total_samples, model=model_type, alias=alias)
        game_stats[alias] = AliasGameStats(wins=int(round(wins)), total=int(round(total_games)))

        correct = _sum_metric(correct_samples, model=model_type, alias=alias)
        total_preds = _sum_metric(pred_samples, model=model_type, alias=alias)
        accuracy_stats[alias] = AliasAccuracyStats(
            correct=int(round(correct)),
            total=int(round(total_preds)),
        )

    return AliasMetrics(game_stats=game_stats, accuracy_stats=accuracy_stats)


def _coerce_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _delta_counter(current: int, baseline: Optional[int]) -> int:
    if baseline is None:
        return max(current, 0)
    delta = current - baseline
    if delta < 0:
        return max(current, 0)
    return delta


def _adjust_game_stats(
    alias: str,
    current: AliasGameStats,
    baseline: Optional[PromotionBaseline],
) -> AliasGameStats:
    if baseline is None:
        return AliasGameStats(wins=current.wins, total=current.total)

    if alias == "Production":
        base_wins = baseline.production_wins
        base_total = baseline.production_total_games
    else:
        base_wins = baseline.b_wins
        base_total = baseline.b_total_games

    return AliasGameStats(
        wins=_delta_counter(current.wins, base_wins),
        total=_delta_counter(current.total, base_total),
    )


def fetch_ticker_state(
    endpoints: Iterable[str],
    model_type: str,
    *,
    history_limit: int = 50,
) -> PromotionTickerState:
    params = {"limit": history_limit, "model_type": model_type}
    last_error: Optional[Exception] = None

    for url in endpoints:
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code != 200:
                last_error = RuntimeError(f"{response.status_code}: {response.text}")
                continue

            try:
                records = response.json()
            except ValueError as exc:  # noqa: B904
                last_error = exc
                continue

            baseline: Optional[PromotionBaseline] = None
            cycles_since_swap = 0

            if records:
                cycles_since_swap = _coerce_int(records[0].get("promotion_cycles_since_swap"))

            for record in records:
                if record.get("decision") == "swap_production_b":
                    baseline = PromotionBaseline(
                        production_wins=_coerce_int(record.get("production_wins")),
                        production_total_games=_coerce_int(record.get("production_total_games")),
                        b_wins=_coerce_int(record.get("b_wins")),
                        b_total_games=_coerce_int(record.get("b_total_games")),
                    )
                    break

            return PromotionTickerState(baseline=baseline, cycles_since_swap=cycles_since_swap)

        except Exception as exc:  # noqa: BLE001
            last_error = exc

    logger.warning("Unable to load promotion history for %s: %s", model_type, last_error)
    return PromotionTickerState(baseline=None, cycles_since_swap=0)


def sort_challenger_candidates(
    provenance_map: Dict[str, str],
    accuracy_stats: Dict[str, AliasAccuracyStats],
    *,
    challenger_aliases: Sequence[str] = CHALLENGER_ALIASES,
) -> List[ChallengerCandidate]:
    """Return challenger aliases ordered by observed accuracy.

    The result keeps deterministic tie-breaking based on the current slot order so
    repeated promotions produce stable alias assignments.
    """

    candidates: List[ChallengerCandidate] = []
    for index, alias in enumerate(challenger_aliases):
        source_alias = provenance_map[alias]
        stats = accuracy_stats.get(source_alias, AliasAccuracyStats(correct=0, total=0))
        candidates.append(
            ChallengerCandidate(
                source_alias=source_alias,
                accuracy=stats.accuracy,
                total_predictions=stats.total,
                current_alias=alias,
                position=index,
            )
        )

    candidates.sort(
        key=lambda item: (
            item.accuracy,
            item.total_predictions,
            -item.position,
        ),
        reverse=True,
    )
    return candidates


def determine_alias_assignments(client: MlflowClient, model_name: str) -> Dict[str, AliasAssignment]:
    assignments: Dict[str, AliasAssignment] = {}
    for alias in ALL_ALIASES:
        mv = client.get_model_version_by_alias(model_name, alias)
        assignments[alias] = AliasAssignment(version=str(mv.version), source_alias=alias)
    return assignments


def apply_alias_updates(
    client: MlflowClient,
    model_name: str,
    desired: Dict[str, AliasAssignment],
    original: Dict[str, AliasAssignment],
    *,
    dry_run: bool,
) -> bool:
    """Apply alias changes in MLflow. Returns True if any updates were made."""

    updates_applied = False
    for alias, assignment in desired.items():
        previous_version = original[alias].version
        if assignment.version != previous_version:
            updates_applied = True
            prefix = "DRY-RUN" if dry_run else "Applying"
            logger.info(
                "%s alias update %s@%s -> version %s (was %s)",
                prefix,
                model_name,
                alias,
                assignment.version,
                previous_version,
            )
            if not dry_run:
                # mlflow client expects (model_name, alias, version)
                client.set_registered_model_alias(model_name, alias, assignment.version)
    return updates_applied


def post_promotion_report(payload: dict, endpoints: Iterable[str]) -> None:
    for endpoint in endpoints:
        try:
            response = requests.post(endpoint, json=payload, timeout=10)
            if response.status_code == 200:
                logger.info("Reported promotion metrics via %s", endpoint)
                return
            logger.debug("Promotion report failed (%s): %s", response.status_code, response.text)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Promotion report exception for %s: %s", endpoint, exc)
    logger.warning("Unable to notify API about promotion results")


def trigger_reload(endpoints: Iterable[str]) -> None:
    for endpoint in endpoints:
        try:
            response = requests.post(endpoint, timeout=30)
            if response.status_code == 200:
                logger.info("Triggered model reload via %s", endpoint)
                return
            logger.debug("Reload endpoint %s responded %s: %s", endpoint, response.status_code, response.text)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Reload attempt failed for %s: %s", endpoint, exc)
    logger.warning("Failed to trigger application reload after promotions")


def evaluate_model_family(
    model_type: str,
    *,
    client: MlflowClient,
    metrics: AliasMetrics,
    ticker_state: PromotionTickerState,
    alpha: float,
    dry_run: bool,
) -> PromotionOutcome:
    model_name = MODEL_REGISTRY_NAMES[model_type]
    original_assignments = determine_alias_assignments(client, model_name)

    desired_assignments: Dict[str, AliasAssignment] = {
        alias: AliasAssignment(version=value.version, source_alias=value.source_alias)
        for alias, value in original_assignments.items()
    }

    # CHANGED 2025-10-18: Use raw counters instead of ticker-adjusted windows
    # The ticker/baseline system was causing issues after swaps because it tracked
    # the old model identities. Now we compare current performance directly.
    baseline = ticker_state.baseline
    production_stats_raw = metrics.game_stats["Production"]
    b_stats_raw = metrics.game_stats["B"]

    test_result = evaluate_promotion_decision(
        b_stats_raw,
        production_stats_raw,
        alpha=alpha,
    )
    
    # Still track the delta for telemetry/display purposes
    production_stats_window = _adjust_game_stats("Production", production_stats_raw, baseline)
    b_stats_window = _adjust_game_stats("B", b_stats_raw, baseline)
    swap_performed = False

    if test_result.should_swap():
        logger.info("%s: swapping Production/B aliases", model_type)
        swap_performed = True
        desired_assignments["Production"] = AliasAssignment(
            version=original_assignments["B"].version,
            source_alias=original_assignments["B"].source_alias,
        )
        desired_assignments["B"] = AliasAssignment(
            version=original_assignments["Production"].version,
            source_alias=original_assignments["Production"].source_alias,
        )
    else:
        logger.info("%s: retaining Production alias (decision=%s)", model_type, test_result.decision)

    # Prepare challenger rankings using post-swap provenance.
    # After the potential swap the provenance for alias B is whatever source
    # supplied the version currently assigned to B.
    provenance_map: Dict[str, str] = {
        alias: desired_assignments[alias].source_alias for alias in ALL_ALIASES
    }

    challenger_candidates = sort_challenger_candidates(provenance_map, metrics.accuracy_stats)

    reorder_applied = False
    alias_rankings: List[Dict[str, object]] = []
    rank_summary_parts: List[str] = []

    for index, candidate in enumerate(challenger_candidates):
        target_alias = CHALLENGER_ALIASES[index]
        source_alias = candidate.source_alias
        accuracy = candidate.accuracy
        total_preds = candidate.total_predictions

        current_source = provenance_map[target_alias]
        alias_rankings.append(
            {
                "alias": target_alias,
                "rank": index + 1,
                "source_alias": source_alias,
                "accuracy": accuracy,
                "total_predictions": total_preds,
            }
        )

        if current_source != source_alias:
            reorder_applied = True

        desired_assignments[target_alias] = AliasAssignment(
            version=original_assignments[source_alias].version,
            source_alias=source_alias,
        )

        alias_display = target_alias if source_alias == target_alias else f"{target_alias} <- {source_alias}"
        if total_preds > 0:
            accuracy_pct = f"{accuracy * 100:.1f}%"
            preds_text = f"{total_preds} preds"
        else:
            accuracy_pct = "n/a"
            preds_text = "0 preds"
        if candidate.current_alias not in {target_alias, source_alias}:
            alias_display += f" (was {candidate.current_alias})"
        rank_summary_parts.append(f"{index + 1}. {alias_display} ({accuracy_pct}, {preds_text})")

    alias_accuracy_items: List[Dict[str, object]] = []
    for alias_name, stats in metrics.accuracy_stats.items():
        alias_accuracy_items.append(
            {
                "alias": alias_name,
                "source_alias": alias_name,
                "accuracy": stats.accuracy,
                "total_predictions": stats.total,
            }
        )

    if test_result.should_swap():
        cycles_since_swap = 0
        production_games_since_swap = 0
        b_games_since_swap = 0
        ticker_note = "Ticker reset after Production/B swap"
    else:
        cycles_since_swap = ticker_state.cycles_since_swap + 1
        production_games_since_swap = production_stats_window.total
        b_games_since_swap = b_stats_window.total
        ticker_note = (
            "Ticker since swap → Production"
            f" {production_games_since_swap} games, B {b_games_since_swap} games, cycles {cycles_since_swap}"
        )

    message = test_result.reason
    if reorder_applied:
        message += " | Challenger aliases reordered"

    if rank_summary_parts:
        stage_label = "reordered" if reorder_applied else "checked"
        message += " | Step 2 challenger order (" + stage_label + "): " + "; ".join(rank_summary_parts)

    message += " | " + ticker_note

    updates_applied = apply_alias_updates(
        client,
        model_name,
        desired_assignments,
        original_assignments,
        dry_run=dry_run,
    )

    if updates_applied:
        logger.info("%s: alias updates applied", model_type)
    else:
        logger.info("%s: no alias changes required", model_type)

    # CHANGED 2025-10-18: Don't change decision field for alias reordering
    # This way Grafana filters for swap_production_b/retain_production/insufficient_data
    # will work cleanly without seeing "alias_reorder" noise
    decision = test_result.decision
    # Note: reorder_applied boolean field still shows if challengers were reshuffled

    alias_assignments_items: List[Dict[str, object]] = []
    for alias_name in ALL_ALIASES:
        desired_assignment = desired_assignments[alias_name]
        original_assignment = original_assignments[alias_name]
        alias_assignments_items.append(
            {
                "alias": alias_name,
                "version": desired_assignment.version,
                "source_alias": desired_assignment.source_alias,
                "previous_version": original_assignment.version,
                "changed": desired_assignment.version != original_assignment.version,
            }
        )

    return PromotionOutcome(
        model_type=model_type,
        test_result=PromotionTestResult(
            z_statistic=test_result.z_statistic,
            p_value=test_result.p_value,
            decision=decision,
            reason=test_result.reason,
        ),
        alias_rankings=alias_rankings,
        alias_accuracies=alias_accuracy_items,
        alias_assignments=alias_assignments_items,
        swap_performed=swap_performed,
        reorder_applied=reorder_applied,
        production_games_since_swap=production_games_since_swap,
        b_games_since_swap=b_games_since_swap,
        cycles_since_swap=cycles_since_swap,
        message=message,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Automated alias promotion helper")
    parser.add_argument(
        "--metrics-timeout",
        type=float,
        default=10.0,
        help="Timeout (seconds) for the metrics HTTP call",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.2,
        help="Significance level for the Production vs B hypothesis test",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Calculate decisions without mutating MLflow aliases",
    )
    parser.add_argument(
        "--model-types",
        nargs="+",
        choices=list(MODEL_REGISTRY_NAMES.keys()),
        help="Restrict processing to specific model families",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    metrics_endpoints = _build_endpoint_list("PROMOTION_METRICS_ENDPOINTS", DEFAULT_METRICS_ENDPOINTS)
    report_endpoints = _build_endpoint_list("PROMOTION_REPORT_ENDPOINTS", DEFAULT_PROMOTION_ENDPOINTS)
    reload_endpoints = _build_endpoint_list("PROMOTION_RELOAD_ENDPOINTS", DEFAULT_RELOAD_ENDPOINTS)
    history_endpoints = _build_endpoint_list("PROMOTION_HISTORY_ENDPOINTS", DEFAULT_HISTORY_ENDPOINTS)

    metrics_text = fetch_metrics_text(metrics_endpoints, timeout=args.metrics_timeout)
    parsed_metrics = parse_metrics(metrics_text)

    set_mlflow_tracking_uri_if_needed()
    client = MlflowClient()

    model_types = args.model_types or list(MODEL_REGISTRY_NAMES.keys())

    outcomes: List[PromotionOutcome] = []
    any_alias_updated = False

    for model_type in model_types:
        norm_type = normalise_model_type(model_type)
        alias_metrics = extract_alias_metrics(parsed_metrics, norm_type)
        ticker_state = fetch_ticker_state(history_endpoints, norm_type)
        outcome = evaluate_model_family(
            norm_type,
            client=client,
            metrics=alias_metrics,
            ticker_state=ticker_state,
            alpha=args.alpha,
            dry_run=args.dry_run,
        )
        outcomes.append(outcome)
        if outcome.swap_performed or outcome.reorder_applied:
            any_alias_updated = True

        payload = {
            "model_type": norm_type,
            "decision": outcome.test_result.decision,
            "z_statistic": outcome.test_result.z_statistic,
            "p_value": outcome.test_result.p_value,
            "production_wins": alias_metrics.game_stats["Production"].wins,
            "production_total_games": alias_metrics.game_stats["Production"].total,
            "b_wins": alias_metrics.game_stats["B"].wins,
            "b_total_games": alias_metrics.game_stats["B"].total,
            "reorder_applied": outcome.reorder_applied,
            "alias_rankings": [
                {
                    "alias": item["alias"],
                    "rank": item["rank"],
                    "source_alias": item["source_alias"],
                    "accuracy": item.get("accuracy"),
                    "total_predictions": item.get("total_predictions"),
                }
                for item in outcome.alias_rankings
            ],
            "alias_accuracies": [
                {
                    "alias": item["alias"],
                    "source_alias": item.get("source_alias"),
                    "accuracy": item.get("accuracy"),
                    "total_predictions": item.get("total_predictions"),
                }
                for item in outcome.alias_accuracies
            ],
            "alias_assignments": outcome.alias_assignments,
            "production_games_since_swap": outcome.production_games_since_swap,
            "b_games_since_swap": outcome.b_games_since_swap,
            "promotion_cycles_since_swap": outcome.cycles_since_swap,
            "reason": outcome.message,
        }
        logger.debug("Promotion payload for %s: %s", norm_type, json.dumps(payload))
        post_promotion_report(payload, report_endpoints)

    if any_alias_updated and not args.dry_run:
        trigger_reload(reload_endpoints)

    # Emit a short summary for logs / CronJob output.
    for outcome in outcomes:
        logger.info(
            "%s: decision=%s swap=%s reorder=%s -- %s",
            outcome.model_type,
            outcome.test_result.decision,
            outcome.swap_performed,
            outcome.reorder_applied,
            outcome.message,
        )


if __name__ == "__main__":
    main()
