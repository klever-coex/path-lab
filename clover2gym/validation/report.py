"""Validation reports: violations and machine-readable output."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Violation:
    """A single constraint violation."""

    check: str                 # machine id, e.g. "speed_limit"
    message: str               # human-readable description
    t: float | None = None     # time of occurrence, s
    value: float | None = None # measured value
    limit: float | None = None # limit that was exceeded

    def to_dict(self) -> dict:
        return {
            "check": self.check,
            "message": self.message,
            "t": self.t,
            "value": self.value,
            "limit": self.limit,
        }


@dataclass
class CheckReport:
    """Result of one validation stage (static or simulation)."""

    stage: str
    ok: bool
    violations: list[Violation] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "ok": self.ok,
            "violations": [v.to_dict() for v in self.violations],
            "stats": self.stats,
        }

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))

    def to_markdown(self) -> str:
        lines = [f"## {self.stage} check: {'PASS' if self.ok else 'FAIL'}", ""]
        if self.violations:
            lines.append("| check | message | t, s | value | limit |")
            lines.append("|---|---|---|---|---|")
            for v in self.violations:
                lines.append(
                    f"| {v.check} | {v.message} | "
                    f"{v.t if v.t is not None else ''} | "
                    f"{v.value if v.value is not None else ''} | "
                    f"{v.limit if v.limit is not None else ''} |"
                )
            lines.append("")
        for key, value in self.stats.items():
            lines.append(f"- **{key}**: {value}")
        return "\n".join(lines) + "\n"
