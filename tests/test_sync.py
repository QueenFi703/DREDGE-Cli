from pathlib import Path

from dredge.sync import sync


def test_sync_generates_artifacts(tmp_path):
    manifest = Path("dredge.manifest.yaml")
    target_manifest = tmp_path / "dredge.manifest.yaml"
    target_manifest.write_text(manifest.read_text(encoding="utf-8"), encoding="utf-8")

    cwd = Path.cwd()
    try:
        import os

        os.chdir(tmp_path)
        rc = sync(target_manifest)
        assert rc == 0
        assert (tmp_path / ".github/workflows/dredge-runtime.yml").exists()
        assert (tmp_path / "Dockerfile.dredge.generated").exists()
        assert (tmp_path / "build/dredge.event_log.json").exists()
        assert (tmp_path / "build/dredge.524.json").exists()
    finally:
        os.chdir(cwd)
