"""Research project lightweight entry point.

The actual implementation lives under ``src/``. This file only forwards to
the command-line interface so that the project can be run as ``main.py``.
"""

from src.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
