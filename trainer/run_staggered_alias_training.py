#!/usr/bin/env python3
"""Utility to run alias-preserving training in staggered fashion.

This orchestrator checks data-driven gating once, then invokes
``train_all_aliases.py`` separately for each alias. Running each alias in
its own subprocess keeps memory usage lower and provides more granular
failure reporting while still ensuring all aliases are retrained on the
same schedule.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, List

# ``train_all_aliases`` lives alongside this file; import shared helpers.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_all_aliases import ALIASES, MODEL_TRAINERS, check_training_needed  # type: ignore

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

DEFAULT_MODEL_TYPES: List[str] = list(MODEL_TRAINERS.keys())


def _determine_db_path() -> Path:
    data_path_env = os.getenv("DATA_PATH")
    if data_path_env:
        return Path(data_path_env) / "rps.db"
    return Path("local") / "rps.db"


def _run_for_alias(alias: str, model_types: Iterable[str], force_child: bool, skip_reload: bool) -> int:
    script_path = Path(__file__).resolve().with_name("train_all_aliases.py")
    cmd: List[str] = [sys.executable, str(script_path), "--aliases", alias]

    if model_types:
        cmd.extend(["--model-types", *model_types])

    # Parent has already performed gating; force children so they always run.
    if force_child:
        cmd.append("--force")

    if skip_reload:
        cmd.append("--skip-app-reload")

    logger.info("➡️  Launching training for alias %s", alias)
    logger.debug("Command: %s", " ".join(cmd))

    result = subprocess.run(cmd, check=False)
    logger.info("⬅️  Alias %s finished with exit code %s", alias, result.returncode)
    return result.returncode


def _run_auto_promotion(verbose: bool = False) -> int:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "auto_promote_models.py"
    if not script_path.exists():
        logger.warning("Auto-promotion script not found at %s", script_path)
        return 0

    cmd: List[str] = [sys.executable, str(script_path)]
    if verbose:
        cmd.append("--verbose")

    logger.info("🚀 Running automated promotion checks...")
    result = subprocess.run(cmd, check=False)
    if result.returncode == 0:
        logger.info("✅ Auto-promotion cycle completed")
    else:
        logger.error("⚠️  Auto-promotion script exited with %s", result.returncode)
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run staggered alias training")
    parser.add_argument(
        "--aliases",
        nargs="+",
        choices=ALIASES,
        help="Aliases to train (default: all)",
    )
    parser.add_argument(
        "--model-types",
        nargs="+",
        choices=DEFAULT_MODEL_TYPES,
        help="Model types to train (default: all)",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=20.0,
        help="Delay between alias runs (seconds)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force training without data gating",
    )

    args = parser.parse_args()

    aliases: List[str] = args.aliases if args.aliases else list(ALIASES)
    model_types: List[str] = args.model_types if args.model_types else list(DEFAULT_MODEL_TYPES)

    logger.info("Aliases to train: %s", ", ".join(aliases))
    logger.info("Model types to train: %s", ", ".join(model_types))

    auto_promotion_disabled = os.getenv("DISABLE_AUTO_PROMOTION", "false").lower() in {"1", "true", "yes"}
    verbose_auto_promotion = os.getenv("AUTO_PROMOTION_VERBOSE", "false").lower() in {"1", "true", "yes"}

    # Honor data-driven gating once up front unless forced.
    if not args.force:
        db_path = _determine_db_path()
        min_new_rows = int(os.getenv("MIN_NEW_ROWS_FOR_TRAINING", "50"))
        min_total_rows = int(os.getenv("MIN_TOTAL_ROWS", "300"))

        logger.info("🔍 Checking training eligibility across all aliases...")
        should_train, stats = check_training_needed(str(db_path), min_new_rows, min_total_rows)

        logger.info(
            "📊 Data snapshot: total=%s new(last %sm)=%s thresholds=(%s new, %s total)",
                    stats.get("total_rows", "?"),
            stats.get("lookback_minutes", "?"),
                    stats.get("new_rows_since_cutoff", "?"),
                    min_new_rows,
                    min_total_rows)

        if not should_train:
            logger.info("⏭️  Skipping staggered training - insufficient fresh data")

            if auto_promotion_disabled:
                logger.info("Auto-promotion skipped due to DISABLE_AUTO_PROMOTION flag")
                sys.exit(0)

            logger.info("📈 Refreshing promotion telemetry despite skip")
            promotion_exit = _run_auto_promotion(verbose=verbose_auto_promotion)
            if promotion_exit != 0:
                logger.warning("Auto-promotion encountered an error (exit code %s)", promotion_exit)
            sys.exit(0)
        logger.info("✅ Thresholds satisfied, proceeding with staggered training")

    else:
        logger.info("⚠️  Force flag supplied; skipping data gating")

    failures: List[str] = []

    for idx, alias in enumerate(aliases):
        skip_reload = idx < len(aliases) - 1
        exit_code = _run_for_alias(alias, model_types, force_child=True, skip_reload=skip_reload)
        if exit_code != 0:
            failures.append(alias)

        # Optional delay between alias runs for resource breathing room.
        if idx < len(aliases) - 1 and args.delay_seconds > 0:
            logger.info("⏳ Waiting %.1f seconds before next alias...", args.delay_seconds)
            time.sleep(args.delay_seconds)

    if failures:
        logger.error("❌ Staggered training completed with failures: %s", ", ".join(failures))
        sys.exit(1)

    logger.info("✅ Staggered training completed successfully for all aliases")

    if auto_promotion_disabled:
        logger.info("Auto-promotion skipped due to DISABLE_AUTO_PROMOTION flag")
        return

    promotion_exit = _run_auto_promotion(verbose=verbose_auto_promotion)
    if promotion_exit != 0:
        logger.warning("Auto-promotion encountered an error (exit code %s)", promotion_exit)


if __name__ == "__main__":
    main()
