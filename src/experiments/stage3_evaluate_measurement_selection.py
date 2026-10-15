"""Explicit human-run Stage-3 evaluate entry, also runnable directly in PyCharm."""
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.experiments.three_uav_measurement_selection import main
else:
    from .three_uav_measurement_selection import main

if __name__ == '__main__':
    main('evaluate')
