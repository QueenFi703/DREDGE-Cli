"""
DREDGE Architecture API Routes

Provides REST API endpoints for the new architecture:
- Pipeline execution
- Provider management
- Telemetry
"""

import asyncio
import json
import logging
from flask import Blueprint, jsonify, request
from flask_login import login_required, current_user
from .architecture import dredge_run_pipeline
from .providers import (
    execute_translation_chain,
    execute_analysis_chain,
    get_provider_status
)

logger = logging.getLogger(__name__)

arch_bp = Blueprint("architecture", __name__, url_prefix="/api/architecture")


@arch_bp.route("/pipeline/execute", methods=["POST"])
@login_required
def execute_pipeline():
    """Execute DREDGE pipeline

    POST /api/architecture/pipeline/execute
    {
        "input_data": {...},
        "pipeline_type": "standard" | "ios_swift"
    }
    """
    try:
        data = request.get_json() or {}
        input_data = data.get("input_data", {})
        pipeline_type = data.get("pipeline_type", "standard")

        # Run async pipeline
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            dredge_run_pipeline(input_data, pipeline_type=pipeline_type)
        )
        loop.close()

        return jsonify({
            "status": "success",
            "result": result,
            "user": current_user.name
        })

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


@arch_bp.route("/translate", methods=["POST"])
@login_required
def translate_text():
    """Execute translation chain with failover

    POST /api/architecture/translate
    {
        "text": "Hello",
        "source_language": "en",
        "target_language": "es"
    }
    """
    try:
        data = request.get_json() or {}

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(execute_translation_chain(data))
        loop.close()

        return jsonify({
            "status": "success",
            "result": result
        })

    except Exception as e:
        logger.error(f"Translation failed: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


@arch_bp.route("/analyze", methods=["POST"])
@login_required
def analyze_text():
    """Execute analysis chain with failover

    POST /api/architecture/analyze
    {
        "query": "What is...",
        "context": {...}
    }
    """
    try:
        data = request.get_json() or {}

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(execute_analysis_chain(data))
        loop.close()

        return jsonify({
            "status": "success",
            "result": result
        })

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


@arch_bp.route("/providers/status", methods=["GET"])
@login_required
def provider_status():
    """Get provider system status

    GET /api/architecture/providers/status
    """
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        status = loop.run_until_complete(get_provider_status())
        loop.close()

        return jsonify({
            "status": "success",
            "providers": status
        })

    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


@arch_bp.route("/health", methods=["GET"])
def architecture_health():
    """Architecture system health check (public)

    GET /api/architecture/health
    """
    return jsonify({
        "status": "healthy",
        "components": {
            "pipeline_engine": "operational",
            "provider_chain": "operational",
            "cache_layer": "operational",
            "telemetry": "operational"
        }
    })


def register_architecture_routes(app):
    """Register architecture routes"""
    app.register_blueprint(arch_bp)
    logger.info("Architecture routes registered")
