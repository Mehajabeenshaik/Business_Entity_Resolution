"""
Amazon ML Challenge 2026 - Pipeline Orchestrator

Usage:
    python -m src.main --stage all
    python -m src.main --stage load_data
    python -m src.main --stage blocking --split train
    python -m src.main --stage blocking --split test
    python -m src.main --stage train
    python -m src.main --stage evaluate
    python -m src.main --stage predict
    python -m src.main --stage validate
"""

from __future__ import annotations
import argparse
import subprocess
import sys


def run_stage(cmd: list[str]) -> None:
    print("\n" + "=" * 70)
    print(">>> Running stage:", " ".join(cmd))
    print("=" * 70 + "\n")
    subprocess.check_call(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description="Amazon ML Challenge 2026 Pipeline Orchestrator")
    parser.add_argument(
        "--stage",
        type=str,
        choices=["all", "load_data", "blocking", "train", "evaluate", "predict", "validate"],
        default="all",
        help="Pipeline stage to execute. 'all' runs: load_data -> blocking(train) -> train -> evaluate.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "test"],
        default="train",
        help="Split for blocking stage (train or test). Default: train",
    )

    args = parser.parse_args()

    if args.stage == "all":
        steps = [
            [sys.executable, "-m", "src.load_data"],
            [sys.executable, "-m", "src.blocking", "--split", "train"],
            [sys.executable, "-m", "src.train"],
            [sys.executable, "-m", "src.evaluate"],
        ]
        for cmd in steps:
            run_stage(cmd)
    elif args.stage == "load_data":
        run_stage([sys.executable, "-m", "src.load_data"])
    elif args.stage == "blocking":
        run_stage([sys.executable, "-m", "src.blocking", "--split", args.split])
    elif args.stage == "train":
        run_stage([sys.executable, "-m", "src.train"])
    elif args.stage == "evaluate":
        run_stage([sys.executable, "-m", "src.evaluate"])
    elif args.stage == "predict":
        run_stage([sys.executable, "-m", "src.predict"])
    elif args.stage == "validate":
        run_stage([sys.executable, "-m", "src.validate_output"])


if __name__ == "__main__":
    main()
