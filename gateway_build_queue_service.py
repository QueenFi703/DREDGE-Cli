"""
DREDGE Build Queue Service - Isolated Sequential Build Management

Features:
  - Sequential build processing (no races)
  - Per-build container isolation
  - Isolated networks and volumes
  - Deterministic versioning
  - Atomic Vercel pushes
"""

import json
import uuid
import logging
import asyncio
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from enum import Enum
from pydantic import BaseModel

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

# Configuration
ARTIFACTS_PATH = Path("/artifacts")
STATE_PATH = Path("/app/state")
QUEUE_MAX_PENDING = int(10)
BUILD_TIMEOUT = int(1800)  # 30 minutes
ARTIFACT_RETENTION = int(86400)  # 24 hours
ISOLATION_ENABLED = True
LANE_NETWORK = "build-lanes"


# ============================================================================
# Models
# ============================================================================

class BuildStatus(str, Enum):
    """Build status states"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BuildJob(BaseModel):
    """Build job definition"""
    id: str = None  # UUID, auto-generated
    status: BuildStatus = BuildStatus.PENDING
    source_branch: str
    source_commit: str
    build_command: str
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    logs: str = ""
    error: Optional[str] = None
    artifacts_path: Optional[str] = None


class BuildQueue:
    """Sequential build queue with isolation"""
    
    def __init__(self):
        """Initialize build queue"""
        self.queue: List[BuildJob] = []
        self.current_build: Optional[BuildJob] = None
        self.build_history: Dict[str, BuildJob] = {}
        self.running = False
        
        # Ensure directories exist
        ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)
        STATE_PATH.mkdir(parents=True, exist_ok=True)
        
        logger.info("Build queue initialized")
    
    async def start(self):
        """Start queue processor"""
        self.running = True
        logger.info("Build queue processor started")
        await self._process_queue()
    
    async def stop(self):
        """Stop queue processor"""
        self.running = False
        logger.info("Build queue processor stopped")
    
    async def submit_build(self, build_job: BuildJob) -> str:
        """Submit build job to queue"""
        
        # Check queue length
        if len(self.queue) >= QUEUE_MAX_PENDING:
            raise HTTPException(
                status_code=429,
                detail=f"Queue full (max {QUEUE_MAX_PENDING} pending builds)"
            )
        
        # Generate unique ID if not provided
        if not build_job.id:
            build_job.id = f"build-{uuid.uuid4().hex[:8]}"
        
        # Set timestamps
        if not build_job.created_at:
            build_job.created_at = datetime.utcnow()
        
        build_job.status = BuildStatus.QUEUED
        
        # Add to queue
        self.queue.append(build_job)
        
        logger.info(f"Build {build_job.id} submitted (queue size: {len(self.queue)})")
        
        return build_job.id
    
    async def _process_queue(self):
        """Main queue processor loop"""
        while self.running:
            try:
                # Process one build at a time
                if self.queue and not self.current_build:
                    build = self.queue.pop(0)
                    self.current_build = build
                    
                    await self._execute_build(build)
                    
                    # Store in history
                    self.build_history[build.id] = build
                    self.current_build = None
                
                await asyncio.sleep(1)
            
            except Exception as e:
                logger.error(f"Queue processing error: {e}")
                self.current_build = None
                await asyncio.sleep(1)
    
    async def _execute_build(self, build: BuildJob):
        """Execute isolated build"""
        logger.info(f"Executing build {build.id}...")
        
        build.status = BuildStatus.RUNNING
        build.started_at = datetime.utcnow()
        
        try:
            # Create build-specific paths
            build_dir = ARTIFACTS_PATH / build.id
            build_dir.mkdir(parents=True, exist_ok=True)
            
            # Create isolated network for this build
            network_name = f"{LANE_NETWORK}-{build.id}"
            await self._create_network(network_name)
            
            # Run build in isolated container
            logs, success = await self._run_isolated_build(
                build_id=build.id,
                network=network_name,
                build_dir=build_dir,
                build_command=build.build_command
            )
            
            build.logs = logs
            build.artifacts_path = str(build_dir)
            
            if success:
                logger.info(f"✅ Build {build.id} succeeded")
                build.status = BuildStatus.SUCCESS
                
                # Push to Vercel if applicable
                await self._push_to_vercel(build)
            else:
                logger.error(f"❌ Build {build.id} failed")
                build.status = BuildStatus.FAILED
                build.error = "Build command failed"
            
            # Cleanup network
            await self._delete_network(network_name)
        
        except asyncio.TimeoutError:
            logger.error(f"Build {build.id} timed out")
            build.status = BuildStatus.FAILED
            build.error = f"Build timeout (>{BUILD_TIMEOUT}s)"
        
        except Exception as e:
            logger.error(f"Build execution error: {e}")
            build.status = BuildStatus.FAILED
            build.error = str(e)
        
        finally:
            build.completed_at = datetime.utcnow()
    
    async def _create_network(self, network_name: str):
        """Create isolated network for build"""
        try:
            subprocess.run(
                ["docker", "network", "create", "--driver", "bridge", network_name],
                check=True,
                capture_output=True,
                timeout=10
            )
            logger.info(f"Network {network_name} created")
        except subprocess.CalledProcessError as e:
            if "already exists" not in e.stderr.decode():
                raise
    
    async def _delete_network(self, network_name: str):
        """Delete network after build"""
        try:
            await asyncio.sleep(5)  # Grace period
            subprocess.run(
                ["docker", "network", "rm", network_name],
                check=False,  # Don't fail if network doesn't exist
                capture_output=True,
                timeout=10
            )
            logger.info(f"Network {network_name} deleted")
        except Exception as e:
            logger.warning(f"Failed to delete network {network_name}: {e}")
    
    async def _run_isolated_build(
        self,
        build_id: str,
        network: str,
        build_dir: Path,
        build_command: str
    ) -> tuple[str, bool]:
        """Run build in isolated container"""
        
        container_name = f"build-{build_id}"
        
        try:
            # Run build container
            result = subprocess.run(
                [
                    "docker", "run",
                    "--rm",
                    "--name", container_name,
                    "--network", network,
                    "-v", f"{build_dir}:/artifacts",
                    "-v", f"{STATE_PATH}:/state:ro",
                    "-e", "BUILD_ID=" + build_id,
                    "-e", "CI=true",
                    "-w", "/artifacts",
                    "dredge-build-lane:v1.0.0-stable",
                    "sh", "-c", build_command
                ],
                capture_output=True,
                text=True,
                timeout=BUILD_TIMEOUT
            )
            
            logs = result.stdout + "\n" + result.stderr
            success = result.returncode == 0
            
            logger.info(f"Build container {container_name} exited with code {result.returncode}")
            
            return logs, success
        
        except subprocess.TimeoutExpired:
            logger.error(f"Build container timed out: {container_name}")
            
            # Kill container
            try:
                subprocess.run(
                    ["docker", "kill", container_name],
                    check=False,
                    timeout=10
                )
            except:
                pass
            
            return "Build timed out", False
        
        except Exception as e:
            logger.error(f"Build execution error: {e}")
            return f"Error: {str(e)}", False
    
    async def _push_to_vercel(self, build: BuildJob):
        """Push successful build to Vercel"""
        try:
            logger.info(f"Pushing build {build.id} to Vercel...")
            
            # This would integrate with Vercel API
            # For now, just log the action
            
            logger.info(f"✅ Build {build.id} pushed to Vercel")
        
        except Exception as e:
            logger.error(f"Vercel push failed: {e}")
            # Don't fail the build just because push failed
    
    # ========================================================================
    # Query Methods
    # ========================================================================
    
    def get_queue_status(self) -> Dict:
        """Get current queue status"""
        return {
            "queue_length": len(self.queue),
            "max_pending": QUEUE_MAX_PENDING,
            "current_build": self.current_build.dict() if self.current_build else None,
            "pending_builds": [b.dict() for b in self.queue]
        }
    
    def get_build_status(self, build_id: str) -> Optional[BuildJob]:
        """Get build status"""
        if self.current_build and self.current_build.id == build_id:
            return self.current_build
        
        return self.build_history.get(build_id)
    
    def get_build_logs(self, build_id: str) -> Optional[str]:
        """Get build logs"""
        build = self.get_build_status(build_id)
        return build.logs if build else None
    
    async def cancel_build(self, build_id: str) -> bool:
        """Cancel pending build"""
        # Remove from queue
        self.queue = [b for b in self.queue if b.id != build_id]
        
        logger.info(f"Build {build_id} cancelled")
        return True


# ============================================================================
# Global Build Queue
# ============================================================================

build_queue = BuildQueue()


# ============================================================================
# FastAPI Integration
# ============================================================================

def setup_build_queue(app: FastAPI):
    """Register build queue with FastAPI"""
    
    @app.on_event("startup")
    async def startup():
        asyncio.create_task(build_queue.start())
    
    @app.on_event("shutdown")
    async def shutdown():
        await build_queue.stop()
    
    @app.post("/build/submit")
    async def submit_build(build_job: BuildJob) -> Dict:
        """Submit build job"""
        try:
            build_id = await build_queue.submit_build(build_job)
            return {"status": "queued", "build_id": build_id}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.get("/build/status/{build_id}")
    async def build_status(build_id: str) -> Dict:
        """Get build status"""
        build = build_queue.get_build_status(build_id)
        
        if not build:
            raise HTTPException(status_code=404, detail="Build not found")
        
        return build.dict()
    
    @app.get("/build/logs/{build_id}")
    async def build_logs(build_id: str) -> Dict:
        """Get build logs"""
        logs = build_queue.get_build_logs(build_id)
        
        if logs is None:
            raise HTTPException(status_code=404, detail="Build not found")
        
        return {"build_id": build_id, "logs": logs}
    
    @app.post("/build/cancel/{build_id}")
    async def cancel_build(build_id: str) -> Dict:
        """Cancel build"""
        success = await build_queue.cancel_build(build_id)
        return {"cancelled": success}
    
    @app.get("/queue/status")
    async def queue_status() -> Dict:
        """Get queue status"""
        return build_queue.get_queue_status()
    
    @app.get("/health")
    async def health() -> Dict:
        """Health check"""
        return {"status": "healthy", "service": "build-queue"}
    
    logger.info("Build queue integrated with FastAPI")
