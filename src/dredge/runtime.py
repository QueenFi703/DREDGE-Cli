from __future__ import annotations

import argparse
import json
from pathlib import Path

from .orchestration import Orchestrator, dump_json, load_manifest


def run(entry: str, manifest_path: Path = Path("dredge.manifest.yaml")) -> int:
    manifest = load_manifest(manifest_path)
    orchestrator = Orchestrator(manifest)

    current = entry
    for edge in manifest.get("graph", {}).get("edges", []):
        if edge["from"] == current:
            orchestrator.transition(edge["from"], edge["to"])
            current = edge["to"]

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
