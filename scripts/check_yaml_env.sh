#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import sys

try:
    import yaml
    print(f"PyYAML available: {yaml.__version__}")
except Exception as exc:
    print("PyYAML not importable in this environment.")
    print(f"Reason: {exc}")
    print("Hint: if pip installs fail with 403 while fetching index pages, use a vendored wheel flow:")
    print("  pip install --no-index --find-links=vendor PyYAML")
    sys.exit(1)
PY
