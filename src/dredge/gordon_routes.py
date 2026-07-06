"""
Gordon Integration Routes

REST API endpoints for Gordon-DREDGE communication
"""

import asyncio
import json
import logging
from flask import Blueprint, jsonify, request
from flask_login import login_required
from .gordon_integration import (
    GordonDREDGEBridge,
    GordonTask,
    get_bridge_status,
    start_gordon_bridge,
    stop_gordon_bridge
)

logger = logging.getLogger(__name__)

gordon_bp = Blueprint("gordon", __name__, url_prefix="/api/gordon")

# Bridge instance
_bridge = None


def get_or_create_bridge():
    """Get or create Gordon bridge"""
    global _bridge
    if _bridge is None:
        _bridge = GordonDREDGEBridge()
    return _bridge


@gordon_bp.route("/health", methods=["GET"])
def gordon_health():
    """Get Gordon bridge health

    GET /api/gordon/health
    """
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        status = loop.run_until_complete(get_bridge_status())
        loop.close()

        return jsonify({
            "status": "success",
            "bridge": status
        })

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


@gordon_bp.route("/start", methods=["POST"])
@login_required
def gordon_start():
    """Start Gordon bridge

    POST /api/gordon/start
    {
        "gordon_url": "http://gordon:8000",
        "dredge_url": "http://dredge:3001"
    }
    """
    try:
        data = request.get_json() or {}
        gordon_url = data.get("gordon_url", "http://localhost:8000")
        dredge_url = data.get("dredge_url", "http://localhost:3001")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        bridge = get_or_create_bridge()
        loop.run_until_complete(bridge.start())

        loop.close()

        return jsonify({
            "status": "success",
            "message": "Gordon bridge started",
            "gordon_url": gordon_url,
            "dredge_url": dredge_url
        })

    except Exception as e:
        logger.error(f"Failed to start bridge: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


@gordon_bp.route("/stop", methods=["POST"])
@login_required
def gordon_stop():
    """Stop Gordon bridge

    POST /api/gordon/stop
    """
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(stop_gordon_bridge())
        loop.close()

        return jsonify({
            "status": "success",
            "message": "Gordon bridge stopped"
        })

    except Exception as e:
        logger.error(f"Failed to stop bridge: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


@gordon_bp.route("/task/execute", methods=["POST"])
@login_required
def execute_task():
    """Execute a task from Gordon

    POST /api/gordon/task/execute
    {
        "task_id": "task_123",
        "title": "Translate text",
        "type": "translate",
        "input_data": {
            "text": "Hello",
            "source_language": "en",
            "target_language": "es"
        }
    }
    """
    try:
        data = request.get_json() or {}

        # Create task
        task = GordonTask(
            task_id=data.get("task_id", "task_" + str(time.time())),
            title=data.get("title", ""),
            description=data.get("description", ""),
            type=data.get("type", "pipeline"),
            input_data=data.get("input_data", {}),
            priority=data.get("priority", 5)
        )

        # Execute with bridge
        bridge = get_or_create_bridge()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(bridge._execute_task(task))
        loop.close()

        return jsonify({
            "status": "success",
            "task_id": task.task_id,
            "execution_status": result.status,
            "result": result.result,
            "duration": result.duration,
            "error": result.error
        })

    except Exception as e:
        logger.error(f"Task execution failed: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


@gordon_bp.route("/capabilities", methods=["GET"])
def capabilities():
    """Get DREDGE capabilities for Gordon

    GET /api/gordon/capabilities
    """
    return jsonify({
        "status": "success",
        "agent": "DREDGE",
        "version": "1.0.0",
        "capabilities": [
            {
                "name": "pipeline_execution",
                "description": "Execute DREDGE DAG pipelines",
                "endpoint": "/api/architecture/pipeline/execute",
                "method": "POST"
            },
            {
                "name": "text_translation",
                "description": "Translate text with multi-provider support",
                "endpoint": "/api/architecture/translate",
                "method": "POST"
            },
            {
                "name": "semantic_analysis",
                "description": "Perform semantic analysis on text",
                "endpoint": "/api/architecture/analyze",
                "method": "POST"
            },
            {
                "name": "provider_management",
                "description": "Monitor and manage provider health",
                "endpoint": "/api/architecture/providers/status",
                "method": "GET"
            }
        ],
        "max_concurrent_tasks": 10,
        "timeout": 30,
        "supported_task_types": [
            "pipeline",
            "translate",
            "analyze"
        ]
    })


@gordon_bp.route("/status", methods=["GET"])
def status():
    """Get detailed DREDGE-Gordon bridge status

    GET /api/gordon/status
    """
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        bridge_status = loop.run_until_complete(get_bridge_status())
        loop.close()

        return jsonify({
            "status": "success",
            "bridge": bridge_status,
            "timestamp": datetime.utcnow().isoformat()
        })

    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


def register_gordon_routes(app):
    """Register Gordon routes"""
    app.register_blueprint(gordon_bp)
    logger.info("Gordon integration routes registered")


# Import at module level
import time
from datetime import datetime
