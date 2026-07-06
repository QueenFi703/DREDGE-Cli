from __future__ import annotations

from pathlib import Path

from .orchestration import (
    generate_dockerfile,
    generate_github_actions,
    load_manifest,
    render_524_projection,
    validate_manifest,
    dump_json,
)
from .runtime import run


def sync(manifest_path: Path = Path("dredge.manifest.yaml")) -> int:
    manifest = load_manifest(manifest_path)
    errors = validate_manifest(manifest)
    if errors:
        raise ValueError("Manifest validation failed:\n- " + "\n- ".join(errors))

    Path(".github/workflows").mkdir(parents=True, exist_ok=True)
    Path(".github/workflows/dredge-runtime.yml").write_text(
        generate_github_actions(manifest), encoding="utf-8"
    )
    Path("Dockerfile.dredge.generated").write_text(generate_dockerfile(), encoding="utf-8")

    run(entry=manifest["graph"]["entry"], manifest_path=manifest_path)

    from json import loads
    events = loads(Path("build/dredge.event_log.json").read_text(encoding="utf-8"))
    projection = [
        {
            "t": e["t"],
            "node": e["to_node"],
            "type": "state_transition",
            "trigger": "internal",
            "state_delta": e.get("state_delta", {}),
            "observability": "log",
        }
        for e in events
        if e["t"] <= 324
    ]
    dump_json(Path("build/dredge.524.json"), projection)
    return 0
