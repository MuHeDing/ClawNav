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

GATEWAY_ARGS = [
    "--harness_runtime openclaw_bridge",
    "--openclaw_planner_backend gateway",
    "--openclaw_gateway_url ${OPENCLAW_GATEWAY_URL}",
]

VISUAL_GATEWAY_ENV = [
    "OPENCLAW_VISUAL_MODE=describe",
    "OPENCLAW_VISUAL_MODEL=${OPENCLAW_VISUAL_MODEL:-qwen/qwen3.5-flash}",
    "OPENCLAW_VISUAL_MAX_IMAGES=2",
    "OPENCLAW_VISUAL_TIMEOUT_MS=30000",
]

CURATOR_CRITIC_ENV = [
    "OPENCLAW_ENABLE_SUBAGENT_CRITIC=1",
    "OPENCLAW_ENABLE_SUBAGENT_MEMORY_CURATOR=1",
]


def build_matrix():
    common = [
        "PYTHONPATH=.:src",
        "/ssd/dingmuhe/anaconda3/envs/janusvln/bin/python",
        "src/evaluation_harness.py",
        "--model_path ${MODEL_PATH}",
        "--habitat_config_path config/vln_r2r.yaml",
        *LOWMEM_ARGS,
    ]
    return [
        {
            "ladder_id": "A0",
            "name": "baseline_lowmem",
            "args": common + ["--harness_runtime phase2", "--harness_mode act_only"],
        },
        {
            "name": "phase2_memory_recall",
            "args": common + ["--harness_runtime phase2", "--harness_mode memory_recall"],
        },
        {
            "ladder_id": "A1",
            "name": "phase3_openclaw_bridge",
            "args": common + ["--harness_runtime openclaw_bridge"],
        },
        {
            "ladder_id": "A2",
            "name": "openclaw_visual_observations",
            "args": VISUAL_GATEWAY_ENV + common + GATEWAY_ARGS,
        },
        {
            "ladder_id": "A3",
            "name": "openclaw_visual_memory_write",
            "args": VISUAL_GATEWAY_ENV
            + common
            + GATEWAY_ARGS
            + ["--harness_memory_source episode-local"],
        },
        {
            "ladder_id": "A4",
            "name": "openclaw_visual_memory_recall",
            "args": VISUAL_GATEWAY_ENV
            + common
            + GATEWAY_ARGS
            + ["--harness_mode memory_recall", "--harness_memory_source episode-local"],
        },
        {
            "ladder_id": "A5",
            "name": "openclaw_visual_memory_recall_replan",
            "args": VISUAL_GATEWAY_ENV
            + ["OPENCLAW_ENABLE_SUBAGENT_CRITIC=1"]
            + common
            + GATEWAY_ARGS
            + [
                "--harness_mode memory_recall",
                "--harness_memory_source episode-local",
                "--openclaw_enable_subagent_critic",
            ],
        },
        {
            "ladder_id": "A6",
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
        {
            "ladder_id": "A7",
            "name": "openclaw_full_visual_memory_system",
            "args": VISUAL_GATEWAY_ENV
            + CURATOR_CRITIC_ENV
            + common
            + GATEWAY_ARGS
            + [
                "--harness_mode memory_recall",
                "--harness_memory_source train-scene-only",
                "--openclaw_enable_subagent_critic",
                "--openclaw_enable_subagent_memory_curator",
            ],
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--output_root", default="results/openclaw_vln_ablation")
    args = parser.parse_args()

    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    for item in build_matrix():
        run_args = item["args"] + [
            "--output_path",
            str(Path(args.output_root) / item["name"]),
        ]
        command = ["bash", "-lc", " ".join(run_args)]
        print(item["name"], command)
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
