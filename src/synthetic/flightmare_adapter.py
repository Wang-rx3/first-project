"""Import contract for production Flightmare episode exports.

Flightmare is intentionally not imported here. This keeps the repository
usable without the simulator while defining the exact file contract that a
Flightmare exporter must satisfy before data enters the benchmark.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from ..schema import EPISODE_FILE_SPECS, validate_episode_dir


def import_flightmare_episode(
    source_episode: str | Path,
    output_root: str | Path,
    split: str,
    episode_id: str,
    allow_overwrite: bool = False,
) -> dict[str, Any]:
    """Copy a validated Flightmare export into the canonical episode layout.

    The source directory must already contain the canonical CSV filenames and
    ``scenario.json``. A future simulator-specific exporter can write these
    files directly or convert native Flightmare logs before calling this
    function.
    """

    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    source = Path(source_episode)
    target = Path(output_root) / split / episode_id
    required = [source / filename for filename in EPISODE_FILE_SPECS]
    required.append(source / "scenario.json")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Flightmare export is missing: " + ", ".join(missing))

    source_report = validate_episode_dir(source)
    if not source_report["ok"]:
        raise ValueError(f"Flightmare export failed schema validation: {source_report}")
    if target.exists() and any(target.iterdir()) and not allow_overwrite:
        raise FileExistsError(f"target episode is not empty: {target}")
    target.mkdir(parents=True, exist_ok=True)

    for filename in EPISODE_FILE_SPECS:
        shutil.copy2(source / filename, target / filename)
    scenario = json.loads((source / "scenario.json").read_text(encoding="utf-8"))
    scenario.update(
        {
            "episode_id": episode_id,
            "split": split,
            "engine_status": "imported_flightmare_export",
            "source_episode": str(source.resolve()),
        }
    )
    (target / "scenario.json").write_text(
        json.dumps(scenario, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "source": str(source),
        "target": str(target),
        "split": split,
        "episode_id": episode_id,
        "files_copied": len(EPISODE_FILE_SPECS) + 1,
        "engine_status": "imported_flightmare_export",
    }

