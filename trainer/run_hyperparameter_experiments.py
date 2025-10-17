#!/usr/bin/env python3
"""Deprecated shim for the legacy hyperparameter experiments entry point."""

import sys

DEPRECATION_MESSAGE = (
    "trainer/run_hyperparameter_experiments.py has been removed. "
    "Use trainer/train_all_models.py with configs/model_hyperparameters.json "
    "or trainer/run_configured_training.py instead."
)


def main() -> None:
    """Exit immediately with a helpful migration message."""
    sys.stderr.write(DEPRECATION_MESSAGE + "\n")
    sys.exit(1)


if __name__ == "__main__":
    main()
