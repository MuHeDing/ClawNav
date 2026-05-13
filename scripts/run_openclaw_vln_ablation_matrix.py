#!/usr/bin/env python
import argparse
import subprocess
from pathlib import Path


LOWMEM_ARGS = [
    "--num_history 8",
    "--max_pixels 401408",
    "--kv_start_size 8",
    "--kv_recent_size 24",
]


def build_matrix():
    common = [
        "src/evaluation_harness.py",
        "--model_path ${MODEL_PATH}",
        "--habitat_config_path config/vln_r2r.yaml",
        *LOWMEM_ARGS,
    ]
    return [
        {
            "name": "baseline_lowmem",
            "args": common + ["--harness_runtime phase2", "--harness_mode act_only"],
        },
        {
            "name": "phase2_memory_recall",
            "args": common + ["--harness_runtime phase2", "--harness_mode memory_recall"],
        },
        {
            "name": "phase3_openclaw_bridge",
            "args": common + ["--harness_runtime openclaw_bridge"],
        },
        {
            "name": "scene_prior_memory",
            "args": common
            + ["--harness_runtime openclaw_bridge", "--harness_memory_source scene-prior"],
        },
        {
            "name": "train_scene_memory",
            "args": common
            + [
                "--harness_runtime openclaw_bridge",
                "--harness_memory_source train-scene-only",
            ],
        },
        {
            "name": "subagent_planner",
            "args": common
            + ["--harness_runtime openclaw_bridge", "--openclaw_enable_subagent_planner"],
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--output_root", default="results/openclaw_vln_ablation")
    args = parser.parse_args()

    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    for item in build_matrix():
        command = ["bash", "-lc", " ".join(item["args"])]
        print(item["name"], command)
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
