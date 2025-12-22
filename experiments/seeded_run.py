#!/usr/bin/env python3
from __future__ import annotations

import argparse
import runpy
import sys


def _seed_all(seed: int) -> None:
    try:
        import random

        random.seed(seed)
    except Exception:
        pass
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed wrapper for experiment runs")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("script")
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    _seed_all(args.seed)
    sys.argv = [args.script] + args.script_args
    runpy.run_path(args.script, run_name="__main__")


if __name__ == "__main__":
    main()
