"""Direct PyCharm/CLI entry point for the v4 diagnostic calibration."""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.experiments.three_uav_measurement_selection_v4 import main
else:
    from .three_uav_measurement_selection_v4 import main


if __name__ == "__main__":
    main("calibrate")

