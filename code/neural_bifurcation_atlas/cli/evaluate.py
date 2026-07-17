from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from neural_bifurcation_atlas.evaluation.bootstrap import bootstrap_auc, paired_bootstrap_auc


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(prog="nam-evaluate")
    value.add_argument("--predictions", type=Path, required=True)
    value.add_argument("--resamples", type=int, default=10000)
    value.add_argument("--seed", type=int, default=1729)
    return value


def main() -> None:
    arguments = parser().parse_args()
    archive = np.load(arguments.predictions)
    target = archive["target"]
    probability = archive["probability"]
    interval = bootstrap_auc(target, probability, arguments.resamples, arguments.seed)
    result: dict[str, object] = {
        "auroc": interval.estimate,
        "lower": interval.lower,
        "upper": interval.upper,
        "standard_error": interval.standard_error,
    }
    if "baseline_probability" in archive:
        comparison = paired_bootstrap_auc(
            target,
            probability,
            archive["baseline_probability"],
            arguments.resamples,
            arguments.seed,
        )
        result["comparison"] = {
            "difference": comparison.difference,
            "lower": comparison.lower,
            "upper": comparison.upper,
            "p_value": comparison.p_value,
        }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
