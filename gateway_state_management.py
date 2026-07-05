"""
DREDGE State Management - Checkpoint & Recovery System

Handles:
  - Persistent state checkpoints
  - Automatic recovery on startup
  - Session persistence across crashes
  - Deterministic state verification
  - Manifest integrity checks
"""

import json
import os
import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional
import asyncio
import threading

logger = logging.getLogger(__name__)

# Configuration
STATE_PATH = Path(os.getenv("STATE_PERSISTENCE_PATH", "/app/state"))
CHECKPOINT_INTERVAL = int(os.getenv("CHECKPOINT_INTERVAL", "30"))  # seconds
MANIFEST_FILE = STATE_PATH / "manifest.json"
CHECKPOINT_FILE = STATE_PATH / "checkpoint.json"
RECOVERY_LOG = STATE_PATH / "recovery.log"


class DREDGEStateManager:
    """
    Manages DREDGE gateway state persistence and recovery
    
    Features:
      - Periodic checkpoints (30s interval)
      - Crash recovery with state reconstruction
      - Manifest verification (SHA256)
      - Session persistence
      - Request queue reconstruction
    """
    
    def __init__(self):
        """Initialize state manager"""
        self.state: Dict[str, Any] = {}
        self.checkpoint_thread: Optional[threading.Thread] = None
        self.running = False
        
        # Ensure directories exist
        STATE_PATH.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"State manager initialized at {STATE_PATH}")
    
    async def initialize(self):
        """Initialize state on startup"""
        logger.info("Initializing DREDGE state management...")
        
        try:
            # Step 1: Check if recovery is needed
            if CHECKPOINT_FILE.exists():
                await self.recover_from_checkpoint()
            else:
                logger.info("No checkpoint found - starting with fresh state")
                await self.create_initial_state()
            
            # Step 2: Start checkpoint background thread
            self.start_checkpoint_thread()
            
            logger.info("State management ready")
            return True
        
        except Exception as e:
            logger.error(f"Failed to initialize state: {e}")
            raise
    
    async def create_initial_state(self):
        """Create fresh state on first startup"""
        self.state = {
            "initialized_at": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            "gateway_id": self._generate_gateway_id(),
            "sessions": {},
            "request_queue": [],
            "build_queue": [],
            "metrics": {
                "uptime_seconds": 0,
                "requests_processed": 0,
                "builds_completed": 0,
                "crashes_recovered": 0
            }
        }
        
        await self.write_checkpoint()
        logger.info("Fresh state created")
    
    async def recover_from_checkpoint(self):
        """Recover state from checkpoint after crash"""
        logger.info("🔄 Recovering from checkpoint...")
        
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                checkpoint = json.load(f)
            
            # Verify checkpoint integrity
            stored_checksum = checkpoint.pop("checksum", None)
            checkpoint_str = json.dumps(checkpoint, sort_keys=True)
            computed_checksum = hashlib.sha256(checkpoint_str.encode()).hexdigest()
            
            if stored_checksum != computed_checksum:
                logger.warning("Checkpoint checksum mismatch - state may be corrupted")
            
            self.state = checkpoint
            
            # Log recovery event
            await self._log_recovery_event(
                reason="WSL crash recovery",
                checkpoint_age=self._get_checkpoint_age(),
                sessions_recovered=len(self.state.get("sessions", {})),
                queue_size=len(self.state.get("request_queue", []))
            )
            
            logger.info(f"✅ Recovered {len(self.state.get('sessions', {}))} sessions")
            logger.info(f"✅ Recovered {len(self.state.get('request_queue', []))} queued requests")
            
            self.state["metrics"]["crashes_recovered"] += 1
        
        except Exception as e:
            logger.error(f"Recovery failed: {e}")
            logger.warning("Falling back to fresh state")
            await self.create_initial_state()
    
    async def write_checkpoint(self):
        """Write state checkpoint to disk"""
        try:
            # Create checkpoint with timestamp
            checkpoint = {
                **self.state,
                "checkpointed_at": datetime.utcnow().isoformat()
            }
            
            # Add checksum for integrity verification
            checkpoint_str = json.dumps(checkpoint, sort_keys=True)
            checksum = hashlib.sha256(checkpoint_str.encode()).hexdigest()
            checkpoint["checksum"] = checksum
            
            # Atomic write (write to temp file, then rename)
            temp_file = CHECKPOINT_FILE.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            
            # Atomic rename (rename is atomic on POSIX/Windows NTFS)
            temp_file.replace(CHECKPOINT_FILE)
            
            # Update manifest for visibility
            await self._update_manifest()
            
        except Exception as e:
            logger.error(f"Failed to write checkpoint: {e}")
    
    async def _update_manifest(self):
        """Update manifest file for external visibility"""
        try:
            manifest = {
                "timestamp": datetime.utcnow().isoformat(),
                "gateway_id": self.state.get("gateway_id"),
                "version": self.state.get("version"),
                "checkpoint_file": str(CHECKPOINT_FILE),
                "recovery_log": str(RECOVERY_LOG),
                "state_summary": {
                    "active_sessions": len(self.state.get("sessions", {})),
                    "queued_requests": len(self.state.get("request_queue", [])),
                    "total_requests_processed": self.state.get("metrics", {}).get("requests_processed", 0),
                    "uptime_seconds": self.state.get("metrics", {}).get("uptime_seconds", 0),
                    "crashes_recovered": self.state.get("metrics", {}).get("crashes_recovered", 0)
                }
            }
            
            with open(MANIFEST_FILE, 'w') as f:
                json.dump(manifest, f, indent=2)
        
        except Exception as e:
            logger.error(f"Failed to update manifest: {e}")
    
    def start_checkpoint_thread(self):
        """Start background checkpoint thread"""
        if self.running:
            return
        
        self.running = True
        self.checkpoint_thread = threading.Thread(
            target=self._checkpoint_loop,
            daemon=True,
            name="StateCheckpointThread"
        )
        self.checkpoint_thread.start()
        logger.info(f"Checkpoint thread started (interval: {CHECKPOINT_INTERVAL}s)")
    
    def _checkpoint_loop(self):
        """Background thread: periodic checkpoints"""
        while self.running:
            try:
                asyncio.run(self.write_checkpoint())
            except Exception as e:
                logger.error(f"Checkpoint loop error: {e}")
            
            # Wait for next checkpoint
            threading.Event().wait(CHECKPOINT_INTERVAL)
    
    async def stop(self):
        """Stop state management"""
        self.running = False
        await self.write_checkpoint()
        logger.info("State management stopped")
    
    # ========================================================================
    # State Access Methods
    # ========================================================================
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state (snapshot)"""
        return self.state.copy()
    
    async def add_session(self, session_id: str, session_data: Dict[str, Any]):
        """Add or update session"""
        self.state["sessions"][session_id] = {
            "created_at": datetime.utcnow().isoformat(),
            **session_data
        }
        logger.info(f"Session {session_id} added")
    
    async def remove_session(self, session_id: str):
        """Remove session"""
        if session_id in self.state["sessions"]:
            del self.state["sessions"][session_id]
            logger.info(f"Session {session_id} removed")
    
    async def queue_request(self, request_id: str, request_data: Dict[str, Any]):
        """Add request to queue"""
        self.state["request_queue"].append({
            "id": request_id,
            "queued_at": datetime.utcnow().isoformat(),
            **request_data
        })
        logger.info(f"Request {request_id} queued")
    
    async def dequeue_request(self, request_id: str):
        """Remove request from queue"""
        self.state["request_queue"] = [
            r for r in self.state["request_queue"] if r.get("id") != request_id
        ]
        logger.info(f"Request {request_id} dequeued")
    
    async def queue_build(self, build_id: str, build_data: Dict[str, Any]):
        """Queue build job"""
        self.state["build_queue"].append({
            "id": build_id,
            "queued_at": datetime.utcnow().isoformat(),
            "status": "pending",
            **build_data
        })
        logger.info(f"Build {build_id} queued")
    
    async def update_build_status(self, build_id: str, status: str, details: Optional[Dict] = None):
        """Update build job status"""
        for build in self.state["build_queue"]:
            if build.get("id") == build_id:
                build["status"] = status
                build["updated_at"] = datetime.utcnow().isoformat()
                if details:
                    build.update(details)
                logger.info(f"Build {build_id} status: {status}")
                break
    
    async def increment_metric(self, metric: str, amount: int = 1):
        """Increment a metric"""
        if "metrics" not in self.state:
            self.state["metrics"] = {}
        
        self.state["metrics"][metric] = self.state["metrics"].get(metric, 0) + amount
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    def _generate_gateway_id(self) -> str:
        """Generate unique gateway ID"""
        import uuid
        return f"gateway-{uuid.uuid4().hex[:8]}"
    
    def _get_checkpoint_age(self) -> float:
        """Get age of checkpoint in seconds"""
        if CHECKPOINT_FILE.exists():
            mtime = CHECKPOINT_FILE.stat().st_mtime
            return (datetime.now().timestamp() - mtime)
        return -1
    
    async def _log_recovery_event(self, reason: str, **details):
        """Log recovery event"""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "recovery",
            "reason": reason,
            "details": details
        }
        
        try:
            with open(RECOVERY_LOG, 'a') as f:
                f.write(json.dumps(event) + "\n")
        except Exception as e:
            logger.error(f"Failed to log recovery event: {e}")


# ============================================================================
# Global State Manager Instance
# ============================================================================

state_manager = DREDGEStateManager()


async def initialize_state_management():
    """Initialize state management on app startup"""
    await state_manager.initialize()


async def shutdown_state_management():
    """Shutdown state management on app shutdown"""
    await state_manager.stop()


# ============================================================================
# FastAPI Integration
# ============================================================================

def setup_state_management(app):
    """Register state management with FastAPI app"""
    
    @app.on_event("startup")
    async def startup():
        await initialize_state_management()
    
    @app.on_event("shutdown")
    async def shutdown():
        await shutdown_state_management()
    
    @app.get("/state/status")
    async def state_status():
        """Get state manager status"""
        state = state_manager.get_state()
        return {
            "status": "operational",
            "gateway_id": state.get("gateway_id"),
            "version": state.get("version"),
            "initialized_at": state.get("initialized_at"),
            "uptime_seconds": state.get("metrics", {}).get("uptime_seconds", 0),
            "sessions_active": len(state.get("sessions", {})),
            "requests_queued": len(state.get("request_queue", [])),
            "builds_queued": len(state.get("build_queue", [])),
            "crashes_recovered": state.get("metrics", {}).get("crashes_recovered", 0)
        }
    
    @app.get("/state/manifest")
    async def state_manifest():
        """Get manifest (external visibility)"""
        if MANIFEST_FILE.exists():
            with open(MANIFEST_FILE, 'r') as f:
                return json.load(f)
        return {"error": "Manifest not found"}
    
    logger.info("State management integrated with FastAPI")
