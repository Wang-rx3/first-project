"""PyCharm-compatible independent v6 entry point, never a training entry."""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.experiments.three_uav_fusion_validation_v6 import main
else:
    from .three_uav_fusion_validation_v6 import main

if __name__ == "__main__":
    main()
