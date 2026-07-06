"""
Gordon Integration for DREDGE

Connects DREDGE architecture with Docker's Gordon AI assistant.
Enables:
- Gordon as a task router
- DREDGE as the execution engine
- Seamless Docker ecosystem integration
- Multi-agent collaboration
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import httpx

logger = logging.getLogger(__name__)


@dataclass
class GordonTask:
    """Task from Gordon"""
    task_id: str
    title: str
    description: str
    type: str  # pipeline, translate, analyze, etc.
    input_data: Dict[str, Any]
    priority: int = 5  # 1-10, higher = more important
    created_at: str = None
    status: str = "pending"  # pending, running, completed, failed

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow().isoformat()


@dataclass
class GordonResult:
    """Result for Gordon"""
    task_id: str
    status: str  # success, error, partial
    result: Dict[str, Any]
    duration: float
    completed_at: str = None
    error: Optional[str] = None

    def __post_init__(self):
        if self.completed_at is None:
            self.completed_at = datetime.utcnow().isoformat()


class GordonClient:
    """Client for communicating with Gordon"""

    def __init__(self, gordon_url: str = "http://localhost:8000"):
        self.gordon_url = gordon_url
        self.client = httpx.AsyncClient(timeout=30.0)
        self.session_id = None

    async def register_with_gordon(self, agent_name: str = "DREDGE") -> bool:
        """Register DREDGE as a sub-agent with Gordon"""
        try:
            response = await self.client.post(
                f"{self.gordon_url}/api/agents/register",
                json={
                    "agent_name": agent_name,
                    "capabilities": [
                        "pipeline_execution",
                        "text_translation",
                        "semantic_analysis",
                        "provider_management"
                    ],
                    "version": "1.0.0",
                    "endpoints": {
                        "execute": "/api/architecture/pipeline/execute",
                        "translate": "/api/architecture/translate",
                        "analyze": "/api/architecture/analyze",
                        "status": "/api/architecture/providers/status"
                    }
                }
            )

            if response.status_code == 200:
                data = response.json()
                self.session_id = data.get("session_id")
                logger.info(f"Registered with Gordon: {self.session_id}")
                return True
            else:
                logger.error(f"Registration failed: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Failed to register with Gordon: {e}")
            return False

    async def listen_for_tasks(self) -> Optional[GordonTask]:
        """Listen for incoming tasks from Gordon"""
        try:
            response = await self.client.get(
                f"{self.gordon_url}/api/tasks/next",
                params={"session_id": self.session_id}
            )

            if response.status_code == 200:
                data = response.json()
                if data and "task" in data:
                    task_data = data["task"]
                    return GordonTask(
                        task_id=task_data.get("id"),
                        title=task_data.get("title"),
                        description=task_data.get("description"),
                        type=task_data.get("type"),
                        input_data=task_data.get("input_data", {}),
                        priority=task_data.get("priority", 5)
                    )

            return None

        except Exception as e:
            logger.error(f"Failed to listen for tasks: {e}")
            return None

    async def send_result(self, result: GordonResult) -> bool:
        """Send result back to Gordon"""
        try:
            response = await self.client.post(
                f"{self.gordon_url}/api/tasks/complete",
                json={
                    "session_id": self.session_id,
                    "task_id": result.task_id,
                    "status": result.status,
                    "result": result.result,
                    "duration": result.duration,
                    "error": result.error
                }
            )

            return response.status_code == 200

        except Exception as e:
            logger.error(f"Failed to send result to Gordon: {e}")
            return False

    async def close(self):
        """Close client connection"""
        await self.client.aclose()


class GordonDREDGEBridge:
    """Bridge between Gordon and DREDGE"""

    def __init__(self, gordon_url: str = "http://localhost:8000", dredge_url: str = "http://localhost:3001"):
        self.gordon = GordonClient(gordon_url)
        self.dredge_url = dredge_url
        self.http_client = httpx.AsyncClient(timeout=30.0)
        self.is_running = False

    async def start(self):
        """Start the bridge"""
        logger.info("Starting Gordon-DREDGE bridge")

        # Register with Gordon
        registered = await self.gordon.register_with_gordon("DREDGE")
        if not registered:
            logger.error("Failed to register with Gordon")
            return False

        self.is_running = True

        # Start listening for tasks
        await self._listen_loop()

        return True

    async def _listen_loop(self):
        """Main listen loop"""
        while self.is_running:
            try:
                # Get next task from Gordon
                task = await self.gordon.listen_for_tasks()

                if task:
                    logger.info(f"Received task from Gordon: {task.task_id}")

                    # Execute task with DREDGE
                    result = await self._execute_task(task)

                    # Send result back to Gordon
                    await self.gordon.send_result(result)

                    logger.info(f"Task completed: {task.task_id}")

                else:
                    # No task available, wait before retry
                    await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Error in listen loop: {e}")
                await asyncio.sleep(2)

    async def _execute_task(self, task: GordonTask) -> GordonResult:
        """Execute a task using DREDGE"""
        start_time = asyncio.get_event_loop().time()

        try:
            if task.type == "pipeline":
                result = await self._execute_pipeline(task)
            elif task.type == "translate":
                result = await self._execute_translate(task)
            elif task.type == "analyze":
                result = await self._execute_analyze(task)
            else:
                return GordonResult(
                    task_id=task.task_id,
                    status="error",
                    result={},
                    duration=asyncio.get_event_loop().time() - start_time,
                    error=f"Unknown task type: {task.type}"
                )

            duration = asyncio.get_event_loop().time() - start_time

            return GordonResult(
                task_id=task.task_id,
                status="success",
                result=result,
                duration=duration
            )

        except Exception as e:
            duration = asyncio.get_event_loop().time() - start_time
            logger.error(f"Task execution failed: {e}")

            return GordonResult(
                task_id=task.task_id,
                status="error",
                result={},
                duration=duration,
                error=str(e)
            )

    async def _execute_pipeline(self, task: GordonTask) -> Dict[str, Any]:
        """Execute pipeline task"""
        response = await self.http_client.post(
            f"{self.dredge_url}/api/architecture/pipeline/execute",
            json=task.input_data
        )
        response.raise_for_status()
        return response.json()

    async def _execute_translate(self, task: GordonTask) -> Dict[str, Any]:
        """Execute translate task"""
        response = await self.http_client.post(
            f"{self.dredge_url}/api/architecture/translate",
            json=task.input_data
        )
        response.raise_for_status()
        return response.json()

    async def _execute_analyze(self, task: GordonTask) -> Dict[str, Any]:
        """Execute analyze task"""
        response = await self.http_client.post(
            f"{self.dredge_url}/api/architecture/analyze",
            json=task.input_data
        )
        response.raise_for_status()
        return response.json()

    async def stop(self):
        """Stop the bridge"""
        self.is_running = False
        await self.gordon.close()
        await self.http_client.aclose()

    async def health_check(self) -> Dict[str, Any]:
        """Get bridge health status"""
        try:
            # Check Gordon
            gordon_status = "unknown"
            try:
                response = await self.http_client.get(f"{self.gordon_url}/health")
                gordon_status = "healthy" if response.status_code == 200 else "unhealthy"
            except:
                gordon_status = "unreachable"

            # Check DREDGE
            dredge_status = "unknown"
            try:
                response = await self.http_client.get(f"{self.dredge_url}/health")
                dredge_status = "healthy" if response.status_code == 200 else "unhealthy"
            except:
                dredge_status = "unreachable"

            return {
                "status": "operational" if self.is_running else "stopped",
                "gordon": gordon_status,
                "dredge": dredge_status,
                "bridge_running": self.is_running
            }

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "error", "error": str(e)}


# Global bridge instance
_bridge: Optional[GordonDREDGEBridge] = None


async def start_gordon_bridge(gordon_url: str = "http://localhost:8000",
                             dredge_url: str = "http://localhost:3001"):
    """Start Gordon-DREDGE bridge"""
    global _bridge

    _bridge = GordonDREDGEBridge(gordon_url, dredge_url)
    await _bridge.start()


async def stop_gordon_bridge():
    """Stop Gordon-DREDGE bridge"""
    global _bridge

    if _bridge:
        await _bridge.stop()


async def get_bridge_status() -> Dict[str, Any]:
    """Get bridge status"""
    global _bridge

    if _bridge:
        return await _bridge.health_check()

    return {"status": "not_initialized"}


if __name__ == "__main__":
    import sys

    async def main():
        """Demo Gordon integration"""
        bridge = GordonDREDGEBridge()

        try:
            # Start bridge
            await bridge.start()
        except KeyboardInterrupt:
            print("Stopping bridge...")
            await bridge.stop()


    asyncio.run(main())
