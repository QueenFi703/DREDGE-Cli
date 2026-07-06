#!/usr/bin/env python
"""
DREDGE CLI Entry Point

Usage:
  dredge --help
  dredge pipeline -q "query"
  dredge translate "text" -t es
  dredge analyze "question"
  dredge status
"""

import sys
from dredge.cli import cli


if __name__ == "__main__":
    try:
        cli()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
