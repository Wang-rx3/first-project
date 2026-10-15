"""Small command-line adapter for the v5 cooperative-EKF diagnostic.

Keeping this module thin makes the experiment implementation importable for
pure contract tests while preserving the requested Stage-3 command name.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))
    from src.experiments.three_uav_cooperative_ekf_reliability_v5 import CONFIG, STEM, build_parser, run
else:
    from .three_uav_cooperative_ekf_reliability_v5 import CONFIG, STEM, build_parser, run


def main() -> None:
    args = build_parser().parse_args()
    report = run(args.run_id, args.config)
    print(json.dumps({
        "run_id": args.run_id,
        "report_path": str(ROOT / "data/synthetic" / STEM / args.run_id / "report.json"),
        "trajectory_transition_count": report.get("trajectory_transition_count"),
        "branch_transition_count": report.get("branch_transition_count"),
        "total_transition_count": report.get("total_transition_count"),
        "integrity_ok": report.get("integrity_ok"),
        "scientific_training_ready": report.get("scientific_training_ready"),
    }, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()


__all__ = ["CONFIG", "STEM", "build_parser", "main", "run"]
