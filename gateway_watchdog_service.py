"""
DREDGE Watchdog Service - Health Monitoring & Auto-Recovery

Monitors:
  - Gateway health
  - Container status
  - Resource usage
  - Auto-restart on failure
  - Alert on critical issues
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import subprocess
import httpx

logger = logging.getLogger(__name__)

# Configuration
GATEWAY_CONTAINER = os.getenv("GATEWAY_CONTAINER", "dredge-gateway")
WATCHDOG_INTERVAL = int(os.getenv("WATCHDOG_INTERVAL", "5"))  # seconds
ALERT_THRESHOLD = int(os.getenv("ALERT_THRESHOLD", "3"))  # failures before alert
RESTART_GRACE_PERIOD = int(os.getenv("RESTART_GRACE_PERIOD", "10"))  # seconds
STATE_PATH = Path(os.getenv("STATE_PERSISTENCE_PATH", "/app/state"))


class WatchdogService:
    """
    Monitors DREDGE gateway health and auto-recovery
    
    Features:
      - Health check every 5 seconds
      - Auto-restart on failure
      - Container monitoring
      - Resource alerts
      - Event logging
    """
    
    def __init__(self):
        """Initialize watchdog"""
        self.running = False
        self.failure_count = 0
        self.last_health_status: Optional[Dict[str, Any]] = None
        self.alert_log = STATE_PATH / "alerts.log"
        self.health_log = STATE_PATH / "health.log"
        
        # Ensure directories exist
        STATE_PATH.mkdir(parents=True, exist_ok=True)
        
        logger.info("Watchdog service initialized")
    
    async def start(self):
        """Start watchdog monitoring"""
        self.running = True
        logger.info(f"Watchdog started (interval: {WATCHDOG_INTERVAL}s)")
        
        # Run monitoring loop
        await self._monitoring_loop()
    
    async def stop(self):
        """Stop watchdog"""
        self.running = False
        logger.info("Watchdog stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                await self._health_check()
                await asyncio.sleep(WATCHDOG_INTERVAL)
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(WATCHDOG_INTERVAL)
    
    async def _health_check(self):
        """Perform health check on gateway"""
        health_status = {
            "timestamp": datetime.utcnow().isoformat(),
            "healthy": False,
            "details": {}
        }
        
        try:
            # Check 1: Container running
            container_status = await self._check_container_status()
            health_status["details"]["container"] = container_status
            
            # Check 2: API responds
            api_healthy = await self._check_api_health()
            health_status["details"]["api"] = api_healthy
            
            # Check 3: State readable
            state_healthy = await self._check_state_health()
            health_status["details"]["state"] = state_healthy
            
            # Check 4: Resources OK
            resources_healthy = await self._check_resources()
            health_status["details"]["resources"] = resources_healthy
            
            # Overall health
            health_status["healthy"] = all([
                container_status.get("running", False),
                api_healthy.get("responding", False),
                state_healthy.get("accessible", False),
                resources_healthy.get("ok", False)
            ])
            
            # Log result
            await self._log_health(health_status)
            self.last_health_status = health_status
            
            # Handle unhealthy state
            if not health_status["healthy"]:
                await self._handle_unhealthy(health_status)
            else:
                # Reset failure count on success
                self.failure_count = 0
        
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            health_status["error"] = str(e)
            health_status["healthy"] = False
            self.failure_count += 1
    
    async def _check_container_status(self) -> Dict[str, Any]:
        """Check if gateway container is running"""
        try:
            result = subprocess.run(
                ["docker", "ps", "--filter", f"name={GATEWAY_CONTAINER}", "--quiet"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            running = bool(result.stdout.strip())
            
            # Also check health status
            inspect_result = subprocess.run(
                ["docker", "inspect", GATEWAY_CONTAINER, "--format={{.State.Health.Status}}"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            health = inspect_result.stdout.strip()
            
            return {
                "running": running,
                "health": health,
                "status": "OK" if running and health != "unhealthy" else "FAILED"
            }
        
        except Exception as e:
            logger.error(f"Container check failed: {e}")
            return {"running": False, "error": str(e)}
    
    async def _check_api_health(self) -> Dict[str, Any]:
        """Check if gateway API is responding"""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(
                    "http://dredge-gateway:9000/health",
                    headers={"User-Agent": "DREDGE-Watchdog"}
                )
                
                return {
                    "responding": response.status_code == 200,
                    "status_code": response.status_code,
                    "response_time_ms": response.elapsed.total_seconds() * 1000
                }
        
        except Exception as e:
            logger.warning(f"API health check failed: {e}")
            return {"responding": False, "error": str(e)}
    
    async def _check_state_health(self) -> Dict[str, Any]:
        """Check if state is readable"""
        try:
            checkpoint_file = STATE_PATH / "checkpoint.json"
            
            if not checkpoint_file.exists():
                return {
                    "accessible": False,
                    "reason": "checkpoint file missing"
                }
            
            with open(checkpoint_file, 'r') as f:
                state = json.load(f)
            
            return {
                "accessible": True,
                "checkpoint_age_seconds": (
                    datetime.utcnow().timestamp() - checkpoint_file.stat().st_mtime
                ),
                "sessions_count": len(state.get("sessions", {})),
                "queue_depth": len(state.get("request_queue", []))
            }
        
        except Exception as e:
            logger.error(f"State health check failed: {e}")
            return {"accessible": False, "error": str(e)}
    
    async def _check_resources(self) -> Dict[str, Any]:
        """Check container resource usage"""
        try:
            result = subprocess.run(
                ["docker", "stats", "--no-stream", "--format={{.MemUsage}}", GATEWAY_CONTAINER],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            mem_usage = result.stdout.strip()
            
            # Simple check: if we can get stats, resources are OK
            return {
                "ok": bool(mem_usage),
                "memory_usage": mem_usage
            }
        
        except Exception as e:
            logger.warning(f"Resource check failed: {e}")
            return {"ok": False, "error": str(e)}
    
    async def _handle_unhealthy(self, status: Dict[str, Any]):
        """Handle unhealthy gateway"""
        self.failure_count += 1
        
        logger.warning(f"Gateway unhealthy (failure {self.failure_count}/{ALERT_THRESHOLD})")
        logger.warning(f"Details: {status['details']}")
        
        # Check if we should alert
        if self.failure_count >= ALERT_THRESHOLD:
            await self._trigger_alert(status)
            
            # Auto-restart after grace period
            if self.failure_count >= ALERT_THRESHOLD + 1:
                await self._auto_restart()
    
    async def _trigger_alert(self, status: Dict[str, Any]):
        """Trigger alert for critical failures"""
        alert = {
            "timestamp": datetime.utcnow().isoformat(),
            "alert_type": "gateway_unhealthy",
            "failure_count": self.failure_count,
            "status": status
        }
        
        logger.error(f"🚨 ALERT: Gateway unhealthy - {json.dumps(alert)}")
        
        # Log to alert file
        try:
            with open(self.alert_log, 'a') as f:
                f.write(json.dumps(alert) + "\n")
        except Exception as e:
            logger.error(f"Failed to log alert: {e}")
    
    async def _auto_restart(self):
        """Auto-restart gateway container"""
        logger.info(f"⚙️  Auto-restarting gateway (grace period: {RESTART_GRACE_PERIOD}s)")
        
        try:
            # Wait grace period before restart
            await asyncio.sleep(RESTART_GRACE_PERIOD)
            
            # Restart container
            subprocess.run(
                ["docker", "restart", GATEWAY_CONTAINER],
                check=True,
                timeout=30
            )
            
            logger.info("✅ Gateway restarted successfully")
            self.failure_count = 0  # Reset counter
            
            # Log restart event
            event = {
                "timestamp": datetime.utcnow().isoformat(),
                "event": "auto_restart",
                "container": GATEWAY_CONTAINER,
                "reason": "health_check_failures"
            }
            
            with open(self.alert_log, 'a') as f:
                f.write(json.dumps(event) + "\n")
        
        except Exception as e:
            logger.error(f"Auto-restart failed: {e}")
    
    async def _log_health(self, status: Dict[str, Any]):
        """Log health check result"""
        try:
            with open(self.health_log, 'a') as f:
                f.write(json.dumps(status) + "\n")
        except Exception as e:
            logger.error(f"Failed to log health: {e}")
    
    # ========================================================================
    # API Endpoints (FastAPI)
    # ========================================================================
    
    async def get_status(self) -> Dict[str, Any]:
        """Get watchdog status"""
        return {
            "status": "operational",
            "running": self.running,
            "failure_count": self.failure_count,
            "last_health_check": self.last_health_status,
            "alert_threshold": ALERT_THRESHOLD,
            "interval_seconds": WATCHDOG_INTERVAL
        }
    
    async def get_health_history(self, limit: int = 100) -> list:
        """Get recent health check history"""
        try:
            history = []
            with open(self.health_log, 'r') as f:
                for line in f.readlines()[-limit:]:
                    history.append(json.loads(line))
            return history
        except Exception as e:
            logger.error(f"Failed to read health history: {e}")
            return []
    
    async def get_alerts(self, limit: int = 100) -> list:
        """Get recent alerts"""
        try:
            alerts = []
            with open(self.alert_log, 'r') as f:
                for line in f.readlines()[-limit:]:
                    alerts.append(json.loads(line))
            return alerts
        except Exception as e:
            logger.error(f"Failed to read alerts: {e}")
            return []


# ============================================================================
# Global Watchdog Instance
# ============================================================================

watchdog_service = WatchdogService()


# ============================================================================
# FastAPI Integration
# ============================================================================

def setup_watchdog(app):
    """Register watchdog with FastAPI app"""
    
    @app.on_event("startup")
    async def startup():
        # Start watchdog in background
        asyncio.create_task(watchdog_service.start())
    
    @app.on_event("shutdown")
    async def shutdown():
        await watchdog_service.stop()
    
    @app.get("/watchdog/status")
    async def watchdog_status():
        """Get watchdog status"""
        return await watchdog_service.get_status()
    
    @app.get("/watchdog/health-history")
    async def watchdog_health_history(limit: int = 100):
        """Get health check history"""
        return {
            "history": await watchdog_service.get_health_history(limit)
        }
    
    @app.get("/watchdog/alerts")
    async def watchdog_alerts(limit: int = 100):
        """Get recent alerts"""
        return {
            "alerts": await watchdog_service.get_alerts(limit)
        }
    
    logger.info("Watchdog integrated with FastAPI")
