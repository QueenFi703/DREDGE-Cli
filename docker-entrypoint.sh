#!/bin/sh
set -eu

if [ "$(id -u)" = "0" ]; then
  if [ -n "${STUDIO_DB_PATH:-}" ]; then
    studio_directory=$(dirname -- "$STUDIO_DB_PATH")
    case "$studio_directory" in
      /data/studio)
        install -d -m 700 -o appuser -g appuser "$studio_directory"
        ;;
      *)
        echo "Studio volume database must be inside /data/studio" >&2
        exit 1
        ;;
    esac
  fi
  exec gosu appuser "$@"
fi

exec "$@"
