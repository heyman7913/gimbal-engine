"""Stage-4 driver: train the IHN (two phases) and run the head-to-head benchmark on NUS.

Usage:
    python scripts/train_and_benchmark.py train      [--quick]
    python scripts/train_and_benchmark.py phaseb     [--quick]
    python scripts/train_and_benchmark.py benchmark
    python scripts/train_and_benchmark.py all         [--quick]
"""

from __future__ import annotations

import argparse
import json

from cuda_motion_flow import runner


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["train", "phaseb", "benchmark", "all"])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.phase in ("train", "all"):
        summary = runner.train(args.quick)
        print(json.dumps({k: summary[k] for k in (
            "use_fused", "ihn_best_mace", "regression_best_mace", "phase_b")}, indent=2))
    if args.phase in ("phaseb", "all"):
        print("phase B:", json.dumps(runner.phaseb(args.quick), indent=2))
    if args.phase in ("benchmark", "all"):
        report = runner.benchmark()
        print("microbench:", json.dumps(report.correlation_microbench, indent=2))
        print(f"wrote {runner.OUT}/benchmark.json, benchmark.csv, stability_by_category.png")


if __name__ == "__main__":
    main()
