from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json


@dataclass(frozen=True)
class Event:
    id: str
    t: float
    from_node: str
    to_node: str
    state_delta: dict[str, Any] = field(default_factory=dict)

    def to_projection_row(self) -> dict[str, Any]:
        return {
            "t": self.t,
            "node": self.to_node,
            "type": "state_transition",
            "trigger": "internal",
            "state_delta": self.state_delta,
            "observability": "log",
        }


class Orchestrator:
    def __init__(self, manifest: dict[str, Any]):
        self.manifest = manifest
        self.state: dict[str, Any] = {}
        self.event_log: list[Event] = []
        self._edges = {
            (edge["from"], edge["to"]): edge
            for edge in self.manifest.get("graph", {}).get("edges", [])
        }

    def now(self) -> float:
        return datetime.now(tz=timezone.utc).timestamp()

    def apply_state(self, delta: dict[str, Any]) -> None:
        for key, value in delta.items():
            self.state[key] = value

    def emit(self, event: Event) -> None:
        self.event_log.append(event)
        self.apply_state(event.state_delta)

    def transition(self, from_node: str, to_node: str) -> Event:
        edge = self._edges.get((from_node, to_node))
        if edge is None:
            raise ValueError("Invalid transition (not in orchestration graph)")

        event = Event(
            id=f"{from_node}->{to_node}",
            t=self.now(),
            from_node=from_node,
            to_node=to_node,
            state_delta=edge.get("state_delta", {}),
        )
        self.emit(event)
        return event


def load_manifest(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        return yaml.safe_load(raw)
    except Exception:
        return _parse_manifest_fallback(raw)


def _parse_manifest_fallback(raw: str) -> dict[str, Any]:
    manifest: dict[str, Any] = {"graph": {"edges": []}}
    in_edges = False
    current: dict[str, str] | None = None
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("entry:"):
            manifest["graph"]["entry"] = stripped.split(":", 1)[1].strip()
        if stripped == "edges:":
            in_edges = True
            continue
        if in_edges and stripped.startswith("- from:"):
            if current:
                manifest["graph"]["edges"].append(current)
            current = {"from": stripped.split(":", 1)[1].strip()}
            continue
        if in_edges and stripped.startswith("to:") and current is not None:
            current["to"] = stripped.split(":", 1)[1].strip()
    if current:
        manifest["graph"]["edges"].append(current)
    return manifest


def validate_manifest(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not manifest.get("graph", {}).get("entry"):
        errors.append("graph.entry is required")

    edges = manifest.get("graph", {}).get("edges", [])
    if not edges:
        errors.append("graph.edges must be non-empty")

    for i, edge in enumerate(edges):
        if "from" not in edge or "to" not in edge:
            errors.append(f"graph.edges[{i}] must include from and to")
    return errors


def generate_github_actions(manifest: dict[str, Any]) -> str:
    entry = manifest["graph"]["entry"]
    return """name: DREDGE Runtime\non:\n  workflow_dispatch:\n  push:\n    branches: [main]\n\njobs:\n  dredge_boot:\n    runs-on: ubuntu-latest\n    steps:\n      - uses: actions/checkout@v4\n      - uses: actions/setup-python@v5\n        with:\n          python-version: '3.11'\n      - run: pip install -e .\n      - run: python -m dredge.runtime {entry}\n""".format(entry=entry)


def generate_dockerfile() -> str:
    return """FROM python:3.11-slim\n\nWORKDIR /app\n\nCOPY . /app\nRUN pip install -r requirements.txt\n\nCMD [\"python\", \"-m\", \"dredge.runtime\"]\n"""


def render_524_projection(events: list[Event], seconds: int = 324) -> list[dict[str, Any]]:
    rows = [e.to_projection_row() for e in events if e.t <= seconds]
    return sorted(rows, key=lambda row: row["t"])


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
