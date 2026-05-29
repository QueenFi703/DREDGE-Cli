from __future__ import annotations

import argparse
import json
from pathlib import Path

from .orchestration import NODE_ID_RE, Orchestrator, dump_json, load_manifest, validate_manifest


MAX_RUNTIME_TRANSITIONS = 512


def run(entry: str, manifest_path: Path = Path("dredge.manifest.yaml")) -> int:
    manifest = load_manifest(manifest_path)
    errors = validate_manifest(manifest)
    if errors:
        raise ValueError("Manifest validation failed:\n- " + "\n- ".join(errors))
    if not NODE_ID_RE.match(entry):
        raise ValueError("Runtime entry contains unsafe characters or is too long")

    orchestrator = Orchestrator(manifest)
    edges_by_source: dict[str, list[dict[str, object]]] = {}
    for edge in manifest["graph"]["edges"]:
        edges_by_source.setdefault(edge["from"], []).append(edge)
    graph_nodes = {
        node
        for edge in manifest["graph"]["edges"]
        for node in (edge["from"], edge["to"])
    }
    if entry not in graph_nodes:
        raise ValueError(f"Runtime entry {entry!r} is not present in graph.edges")

    current = entry
    for _ in range(MAX_RUNTIME_TRANSITIONS):
        outgoing = edges_by_source.get(current, [])
        if not outgoing:
            break
        if len(outgoing) > 1:
            targets = ", ".join(str(edge["to"]) for edge in outgoing)
            raise ValueError(f"Ambiguous transition from {current!r}: {targets}")

        edge = outgoing[0]
        orchestrator.transition(str(edge["from"]), str(edge["to"]))
        current = str(edge["to"])
    else:
        raise ValueError(f"Runtime exceeded {MAX_RUNTIME_TRANSITIONS} transitions")

    dump_json(Path("build/dredge.event_log.json"), [e.__dict__ for e in orchestrator.event_log])
    print(json.dumps({"events": len(orchestrator.event_log), "final_node": current}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dredge.runtime")
    parser.add_argument("entry", nargs="?", default="entry.initialize")
    parser.add_argument("--manifest", default="dredge.manifest.yaml")
    args = parser.parse_args(argv)
    return run(entry=args.entry, manifest_path=Path(args.manifest))


if __name__ == "__main__":
    raise SystemExit(main())
