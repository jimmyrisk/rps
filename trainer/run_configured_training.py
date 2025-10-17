"""Train and promote RPS models using declarative hyperparameter configs.

Usage:
    python trainer/run_configured_training.py \
        --config configs/model_hyperparameters.json \
        --model-types feedforward multinomial_logistic xgboost

The script loads the JSON configuration, applies per-model hyperparameters,
trains sequentially, and promotes the resulting runs to the configured aliases.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MODEL_REGISTRY: Dict[str, Tuple[str, str]] = {
    "feedforward": ("trainer.train_feedforward", "FeedforwardRPSModel"),
    "feedforward_nn": ("trainer.train_feedforward", "FeedforwardRPSModel"),
    "multinomial_logistic": ("trainer.train_mnlogit_torch", "MNLogitRPSModel"),
    "mnlogit": ("trainer.train_mnlogit_torch", "MNLogitRPSModel"),
    "xgboost": ("trainer.train_xgboost", "XGBoostRPSModel"),
}


def _load_model_class(model_type: str):
    key = model_type.lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model type '{model_type}'. Known: {sorted(MODEL_REGISTRY)}")
    module_name, class_name = MODEL_REGISTRY[key]
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _stringify_tags(raw_tags: Dict[str, Any]) -> Dict[str, str]:
    return {k: ("" if v is None else str(v)) for k, v in raw_tags.items()}


def _apply_training(model_entry: Dict[str, Any], config: Dict[str, Any], dry_run: bool = False) -> Dict[str, Any]:
    model_type = model_entry["model_type"]
    ModelClass = _load_model_class(model_type)
    alias = config.get("promote_alias")
    stage = config.get("promote_stage")
    hyperparams = config.get("hyperparameters", {})
    config_id = config.get("id")
    tags = config.get("mlflow_tags", {})

    if dry_run:
        return {
            "model_type": model_type,
            "config_id": config_id,
            "alias": alias,
            "stage": stage,
            "dry_run": True,
        }

    model = ModelClass()
    if alias is not None:
        model.promote_alias = alias
    else:
        model.promote_alias = None

    if stage is not None:
        model.promote_stage = stage
    else:
        # Explicitly disable stage promotion when stage omitted
        model.promote_stage = None

    # Augment MLflow tags for richer traceability
    augmented_tags = {
        "config_family": f"{model_type}_manual_configs",
        "target_alias": alias,
        "requested_stage": stage,
    }
    augmented_tags.update(tags or {})
    tags_str = _stringify_tags(augmented_tags)

    print("\n" + "=" * 80)
    alias_label = alias if alias not in (None, "", "none") else "disabled"
    print(f"Training {model_type} with config '{config_id}' → alias '{alias_label}'")
    print("=" * 80)
    print(f"Hyperparameters: {json.dumps(hyperparams, indent=2)}")

    success = model.train(
        hyperparam_overrides=hyperparams,
        config_id=config_id,
        extra_tags=tags_str,
    )

    result = {
        "model_type": model_type,
        "config_id": config_id,
        "alias": alias,
        "stage": stage,
        "success": bool(success),
    }
    return result


def _filter_configs(
    model_entries: Iterable[Dict[str, Any]],
    model_types: Optional[List[str]],
    config_ids: Optional[List[str]],
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    normalized_types = {m.lower() for m in model_types} if model_types else None
    normalized_ids = set(config_ids) if config_ids else None

    for entry in model_entries:
        entry_type = entry.get("model_type", "").lower()
        if normalized_types and entry_type not in normalized_types:
            continue
        for config in entry.get("configs", []):
            config_id = config.get("id")
            if normalized_ids and config_id not in normalized_ids:
                continue
            pairs.append((entry, config))
    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RPS models from JSON hyperparameter configs")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "model_hyperparameters.json"),
        help="Path to the hyperparameter configuration JSON file.",
    )
    parser.add_argument(
        "--model-types",
        nargs="*",
        help="Optional list of model types to train (e.g., feedforward xgboost).",
    )
    parser.add_argument(
        "--config-ids",
        nargs="*",
        help="Optional list of specific configuration IDs to run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned runs without executing training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    model_entries = payload.get("models", [])
    plan = _filter_configs(model_entries, args.model_types, args.config_ids)
    if not plan:
        print("No matching configurations found. Nothing to do.")
        return

    results = []
    for model_entry, config in plan:
        try:
            outcome = _apply_training(model_entry, config, dry_run=args.dry_run)
            results.append(outcome)
        except Exception as exc:  # noqa: BLE001 - surface exact failure per config
            results.append(
                {
                    "model_type": model_entry.get("model_type"),
                    "config_id": config.get("id"),
                    "alias": config.get("promote_alias"),
                    "stage": config.get("promote_stage"),
                    "success": False,
                    "error": str(exc),
                }
            )
            print(f"❌ Training failed for {model_entry.get('model_type')}@{config.get('id')}: {exc}")
            raise

    print("\n" + "-" * 80)
    print("Training summary:")
    for row in results:
        status = "SUCCESS" if row.get("success", False) else "FAILED"
        alias = row.get("alias")
        config_id = row.get("config_id")
        model_type = row.get("model_type")
        print(f"  [{status}] {model_type} :: {config_id} → {alias}")
    print("-" * 80 + "\n")


if __name__ == "__main__":
    main()
