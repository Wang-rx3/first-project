"""Audit the single-source-of-truth notation table.

The audit focuses on the registry itself. A duplicate registry entry is an
unambiguous notation defect; scanning free-form prose would create many false
positives. Researchers must still register a new symbol before using it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SymbolAudit:
    path: str
    canonical_symbols: int = 0
    legacy_symbols: int = 0
    duplicate_symbols: dict[str, list[int]] = field(default_factory=dict)
    malformed_rows: list[int] = field(default_factory=list)
    missing_status_rows: list[int] = field(default_factory=list)
    missing_meaning_rows: list[int] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (
            self.duplicate_symbols
            or self.malformed_rows
            or self.missing_status_rows
            or self.missing_meaning_rows
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "ok": self.ok,
            "canonical_symbols": self.canonical_symbols,
            "legacy_symbols": self.legacy_symbols,
            "duplicate_symbols": self.duplicate_symbols,
            "malformed_rows": self.malformed_rows,
            "missing_status_rows": self.missing_status_rows,
            "missing_meaning_rows": self.missing_meaning_rows,
        }


def _cells(line: str) -> list[str]:
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return []
    return [cell.strip() for cell in stripped[1:-1].split("|")]


def _normalize_symbol(symbol: str) -> str:
    value = symbol.strip()
    value = re.sub(r"^\\\(|\\\)$", "", value)
    value = re.sub(r"^\$|\$$", "", value)
    value = re.sub(r"\s+", "", value)
    value = value.replace("\\!", "")
    return value


def audit_symbol_table(path: str | Path) -> SymbolAudit:
    """Check canonical and legacy notation tables in one Markdown file."""

    source = Path(path)
    result = SymbolAudit(str(source))
    if not source.exists():
        result.malformed_rows.append(0)
        return result

    lines = source.read_text(encoding="utf-8").splitlines()
    in_canonical = False
    in_legacy = False
    seen: dict[str, list[int]] = {}

    for line_number, line in enumerate(lines, start=1):
        if line.startswith("### Table I. Notation"):
            in_canonical = True
            in_legacy = False
            continue
        if re.match(r"^### 2\.\d+\.1\b", line):
            in_canonical = False
            in_legacy = True
            continue
        if re.match(r"^### 2\.\d+\.[23]\b", line):
            in_legacy = False
            continue
        if not (in_canonical or in_legacy):
            continue

        cells = _cells(line)
        if not cells or all(set(cell) <= {"-", ":", " "} for cell in cells):
            continue
        expected_cells = 4 if in_canonical else 3
        if len(cells) != expected_cells:
            result.malformed_rows.append(line_number)
            continue

        symbol = _normalize_symbol(cells[1])
        meaning = cells[2]
        if not symbol:
            result.malformed_rows.append(line_number)
            continue
        if in_canonical:
            result.canonical_symbols += 1
            if not meaning:
                result.missing_meaning_rows.append(line_number)
            if not cells[3]:
                result.missing_status_rows.append(line_number)
            seen.setdefault(symbol, []).append(line_number)
        else:
            result.legacy_symbols += 1

    result.duplicate_symbols = {
        symbol: rows for symbol, rows in seen.items() if len(rows) > 1
    }
    return result
