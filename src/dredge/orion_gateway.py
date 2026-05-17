"""Orion Gateway server for routing model invocation requests."""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass

from flask import Flask, jsonify, request

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ApiKeyPolicy:
    name: str
    quota_per_hour: int


API_KEYS = {
    "demo-pro-key": ApiKeyPolicy(name="demo-pro", quota_per_hour=100),
}

_USAGE = defaultdict(list)


def _check_quota(api_key: str, quota_per_hour: int) -> tuple[bool, int]:
    now = time.time()
    window_start = now - 3600
    hits = [ts for ts in _USAGE[api_key] if ts >= window_start]
    _USAGE[api_key] = hits
    if len(hits) >= quota_per_hour:
        return False, 0
    _USAGE[api_key].append(now)
    return True, quota_per_hour - len(_USAGE[api_key])


def _route_to_reasoning_engine(text: str, mode: str) -> dict:
    model = "quasimoto" if mode == "deep" else "dolly"
    return {
        "model": model,
        "mode": mode,
        "output": f"[{model}] Analysis complete for: {text}",
    }


def create_app() -> Flask:
    app = Flask(__name__)

    @app.post("/invoke")
    def invoke() -> tuple:
        api_key = request.headers.get("x-api-key", "")
        policy = API_KEYS.get(api_key)
        if not policy:
            return jsonify({"error": "Unauthorized API key"}), 401

        allowed, remaining = _check_quota(api_key, policy.quota_per_hour)
        if not allowed:
            return jsonify({"error": "Quota exceeded"}), 429

        payload = request.get_json(silent=True) or {}
        user_input = payload.get("input")
        mode = payload.get("mode", "fast")
        if not user_input:
            return jsonify({"error": "Missing required field: input"}), 400

        logger.info("invoke request key=%s mode=%s", policy.name, mode)
        result = _route_to_reasoning_engine(str(user_input), str(mode))

        response = {
            "gateway": "orion",
            "status": "ok",
            "result": result,
            "usage": {
                "requests_remaining": remaining,
                "quota_per_hour": policy.quota_per_hour,
            },
        }
        return jsonify(response), 200

    @app.get("/health")
    def health() -> tuple:
        return jsonify({"status": "healthy", "service": "orion-gateway"}), 200

    return app


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    app = create_app()
    app.run(host="0.0.0.0", port=3001)


if __name__ == "__main__":
    main()
