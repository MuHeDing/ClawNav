#!/usr/bin/env python
import argparse
import json
from pathlib import Path


def build_manifest(memory_source: str, split: str) -> dict:
    return {
        "memory_source": memory_source,
        "entries": [
            {
                "split": split,
                "uses_target_instruction": False,
                "uses_oracle_path": False,
                "uses_future_observations": False,
            }
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--memory_source",
        choices=["scene-prior", "train-scene-only"],
        required=True,
    )
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    split = "exploration" if args.memory_source == "scene-prior" else "train"
    Path(args.output_path).write_text(
        json.dumps(build_manifest(args.memory_source, split), indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
