from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import re
import shlex


NODE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
MAX_GRAPH_EDGES = 512


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
        errors = validate_manifest(manifest)
        if errors:
            raise ValueError("Manifest validation failed:\n- " + "\n- ".join(errors))
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
        if not isinstance(delta, dict):
            raise ValueError("state_delta must be an object")
        for key, value in delta.items():
            if not isinstance(key, str) or not key:
                raise ValueError("state_delta keys must be non-empty strings")
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

        manifest = yaml.safe_load(raw)
        if not isinstance(manifest, dict):
            return {}
        # Extract the manifest root element if present
        if "manifest" in manifest and isinstance(manifest["manifest"], dict):
            return manifest["manifest"]
        return manifest
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
    if not isinstance(manifest, dict):
        return ["manifest must be an object"]

    graph = manifest.get("graph")
    if not isinstance(graph, dict):
        return ["graph must be an object"]

    entry = graph.get("entry")
    if not isinstance(entry, str) or not entry:
        errors.append("graph.entry is required")
    elif not NODE_ID_RE.match(entry):
        errors.append("graph.entry contains unsafe characters or is too long")

    edges = graph.get("edges")
    if not isinstance(edges, list) or not edges:
        errors.append("graph.edges must be non-empty")
        return errors
    if len(edges) > MAX_GRAPH_EDGES:
        errors.append(f"graph.edges must contain at most {MAX_GRAPH_EDGES} edges")

    nodes: set[str] = set()
    seen_edges: set[tuple[str, str]] = set()
    adjacency: dict[str, list[str]] = {}

    for i, edge in enumerate(edges):
        if not isinstance(edge, dict):
            errors.append(f"graph.edges[{i}] must be an object")
            continue

        from_node = edge.get("from")
        to_node = edge.get("to")
        if not isinstance(from_node, str) or not isinstance(to_node, str):
            errors.append(f"graph.edges[{i}] must include from and to")
            continue

        if not NODE_ID_RE.match(from_node):
            errors.append(f"graph.edges[{i}].from contains unsafe characters or is too long")
        if not NODE_ID_RE.match(to_node):
            errors.append(f"graph.edges[{i}].to contains unsafe characters or is too long")
        if from_node == to_node:
            errors.append(f"graph.edges[{i}] must not be a self-loop")

        state_delta = edge.get("state_delta", {})
        if not isinstance(state_delta, dict):
            errors.append(f"graph.edges[{i}].state_delta must be an object")

        edge_key = (from_node, to_node)
        if edge_key in seen_edges:
            errors.append(f"graph.edges[{i}] duplicates {from_node}->{to_node}")
        seen_edges.add(edge_key)
        nodes.update(edge_key)
        adjacency.setdefault(from_node, []).append(to_node)

    if isinstance(entry, str) and entry and entry not in nodes:
        errors.append("graph.entry must reference a node in graph.edges")

    unreachable = nodes - _reachable_nodes(entry, adjacency) if isinstance(entry, str) else set()
    for node in sorted(unreachable):
        errors.append(f"graph node {node!r} is unreachable from graph.entry")

    cycle = _find_cycle(adjacency)
    if cycle:
        errors.append("graph must be acyclic; cycle detected: " + " -> ".join(cycle))
    return errors


def generate_github_actions(manifest: dict[str, Any]) -> str:
    errors = validate_manifest(manifest)
    if errors:
        raise ValueError("Manifest validation failed:\n- " + "\n- ".join(errors))
    entry = shlex.quote(manifest["graph"]["entry"])
    return """name: DREDGE Runtime\non:\n  workflow_dispatch:\n  push:\n    branches: [main]\n\njobs:\n  dredge_boot:\n    runs-on: ubuntu-latest\n    steps:\n      - uses: actions/checkout@v4\n      - uses: actions/setup-python@v5\n        with:\n          python-version: '3.11'\n      - run: pip install -e .\n      - run: python -m dredge.runtime {entry}\n""".format(entry=entry)


def generate_dockerfile() -> str:
    return """FROM python:3.11-slim\n\nWORKDIR /app\n\nCOPY . /app\nRUN pip install -r requirements.txt\n\nCMD [\"python\", \"-m\", \"dredge.runtime\"]\n"""


def render_524_projection(events: list[Event], seconds: int = 324) -> list[dict[str, Any]]:
    rows = [e.to_projection_row() for e in events if e.t <= seconds]
    return sorted(rows, key=lambda row: row["t"])


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _reachable_nodes(entry: str, adjacency: dict[str, list[str]]) -> set[str]:
    if not entry:
        return set()

    reachable: set[str] = set()
    stack = [entry]
    while stack:
        node = stack.pop()
        if node in reachable:
            continue
        reachable.add(node)
        stack.extend(adjacency.get(node, []))
    return reachable


def _find_cycle(adjacency: dict[str, list[str]]) -> list[str]:
    visiting: set[str] = set()
    visited: set[str] = set()
    path: list[str] = []

    def visit(node: str) -> list[str]:
        if node in visiting:
            start = path.index(node)
            return path[start:] + [node]
        if node in visited:
            return []

        visiting.add(node)
        path.append(node)
        for next_node in adjacency.get(node, []):
            cycle = visit(next_node)
            if cycle:
                return cycle
        path.pop()
        visiting.remove(node)
        visited.add(node)
        return []

    for node in sorted(adjacency):
        cycle = visit(node)
        if cycle:
            return cycle
    return []