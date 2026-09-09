"""Isolated schema-only process: the only v2 reader authorized to inspect test."""
import json
import sys
from pathlib import Path
from ..schema import validate_episode_benchmark

if __name__ == "__main__":
    report = validate_episode_benchmark(Path(sys.argv[1]))
    Path(sys.argv[2]).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    raise SystemExit(0 if report.get("ok") else 1)
