from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from neural_bifurcation_atlas.data.signal import preprocess_eeg, window_signal


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(prog="nam-prepare")
    value.add_argument("--input", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--sample-rate", type=float, required=True)
    value.add_argument("--window-seconds", type=float, default=30.0)
    value.add_argument("--stride-seconds", type=float, default=2.0)
    return value


def main() -> None:
    arguments = parser().parse_args()
    signal = torch.from_numpy(np.load(arguments.input)).float()
    cleaned = preprocess_eeg(signal, arguments.sample_rate)
    window = round(arguments.window_seconds * arguments.sample_rate)
    stride = round(arguments.stride_seconds * arguments.sample_rate)
    windows = window_signal(cleaned, window, stride)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(arguments.output, windows.numpy())


if __name__ == "__main__":
    main()
