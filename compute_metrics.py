#!/usr/bin/env python3
"""
Simple one-liner metrics computation similar to evaluation.py style.
"""

import json
import argparse

def main():
    parser = argparse.ArgumentParser(description='Compute metrics from result.json with optional line limit')
    parser.add_argument('--result_file', type=str,
                        default="/ssd/dingmuhe/Embodied-task/JanusVLN_Memory/results/janusvln_qformeronly_memory_1605632_start8_recent24_history8_mbl50_ckpt8000/result.json",
                        help='Path to result.json file')
    parser.add_argument('--max_lines', type=int, default=1816,
                        help='Maximum number of lines to read (default: all)')
    parser.add_argument('--skip_lines', type=int, default=0,
                        help='Number of lines to skip from the beginning (default: 0)')
    args = parser.parse_args()

    success_all = []
    spl_all = []
    os_all = []
    ne_all = []

    with open(args.result_file, 'r') as f:
        for line_idx, line in enumerate(f):
            # Skip initial lines if specified
            if line_idx < args.skip_lines:
                continue

            # Stop if max_lines reached
            if args.max_lines is not None and len(success_all) >= args.max_lines:
                break

            if line.strip():
                data = json.loads(line)
                success_all.append(data['success'])
                spl_all.append(data['spl'])
                os_all.append(data['os'])
                ne_all.append(data['ne'])

    # Compute aggregated metrics (similar to evaluation.py style)
    result_all = {
        "success_all": sum(success_all) / len(success_all),
        "spl_all": sum(spl_all) / len(spl_all),
        "os_all": sum(os_all) / len(os_all),
        "ne_all": sum(ne_all) / len(ne_all),
        'length': len(success_all)
    }

    print(f"\nProcessed {len(success_all)} episodes from: {args.result_file}")
    if args.skip_lines > 0:
        print(f"Skipped first {args.skip_lines} lines")
    if args.max_lines is not None:
        print(f"Limited to {args.max_lines} lines")
    print("-" * 60)
    print(result_all)
    print()

if __name__ == "__main__":
    main()
