#!/usr/bin/env python
import argparse
import json
from pathlib import Path
from typing import List, Optional

from harness.visual_readback.metrics import (
    format_visual_readback_markdown,
    summarize_visual_readback_run,
)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--output", default="")
    args = parser.parse_args(argv)

    summary = summarize_visual_readback_run(Path(args.run_dir))
    if args.format == "json":
        text = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
    else:
        text = format_visual_readback_markdown(summary)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
