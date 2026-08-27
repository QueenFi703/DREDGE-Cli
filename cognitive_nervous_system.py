"""
DREDGE Cognitive Nervous System - Neural Integration Layer

Three-Layer Cognitive Architecture:

┌─────────────────────────────────────────────────────┐
│  GPT Sol (Reasoning Layer)                          │
│  - Deep analytical thinking                         │
│  - Multi-modal reasoning (deductive, inductive...)  │
│  - Ethical analysis                                 │
│  - Strategic forecasting                            │
└──────────────┬──────────────────────────────────────┘
               │ (analysis & insights)
               ↓
┌─────────────────────────────────────────────────────┐
│  Tresh (Decision/Orchestration Layer)               │
│  - Strategic decision-making                        │
│  - Agent orchestration                              │
│  - Task sequencing and adaptation                   │
│  - Learning feedback loops                          │
└──────────────┬──────────────────────────────────────┘
               │ (decisions & execution plans)
               ↓
┌─────────────────────────────────────────────────────┐
│  DREDGE (Application/Execution Layer)               │
│  - Execute decisions                                │
│  - Manage resources and workflows                   │
│  - Collect performance feedback                     │
│  - Provide real-time telemetry                      │
└─────────────────────────────────────────────────────┘

Integration: Synchronous request-response with async callbacks
"""

import logging
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
import asyncio
import json

logger = logging.getLogger(__name__)


# ============================================================================
# COMMUNICATION PROTOCOL
# ============================================================================

class NeuroMessage:
    """Inter-layer communication message"""
    
    def __init__(self, source: str, target: str, message_type: str, payload: Dict[str, Any]):
        self.source = source
        self.target = target
        self.message_type = message_type
        self.payload = payload
        self.timestamp = datetime.utcnow().isoformat()
        self.message_id = f"{source}_{target}_{datetime.utcnow().timestamp()}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "source": self.source,
            "target": self.target,
            "type": self.message_type,
            "timestamp": self.timestamp,
            "payload": self.payload
        }


# ============================================================================
# NERVOUS SYSTEM BUS
# ============================================================================

class CognitiveNervousSystem:
    """
    Central integration hub for three-layer cognitive architecture
    
    Responsibilities:
    - Route messages between layers
    - Manage request/response flow
    - Collect and distribute telemetry
    - Handle asynchronous callbacks
    - Maintain system state
    """
    
    def __init__(self):
        self.gpt_sol_engine = None
        self.tresh_engine = None
        self.dredge_engine = None
        self.message_log: List[Dict[str, Any]] = []
        self.telemetry_data: Dict[str, Any] = {
            "requests_processed": 0,
            "avg_response_time": 0.0,
            "layer_latencies": {"gpt_sol": 0.0, "tresh": 0.0, "dredge": 0.0},
            "system_health": "operational"
        }
        logger.info("Cognitive Nervous System initialized")
    
    # ========================================================================
    # LAYER REGISTRATION
    # ========================================================================
    
    def register_layers(self, gpt_sol=None, tresh=None, dredge=None):
        """Register the three cognitive layers"""
        self.gpt_sol_engine = gpt_sol
        self.tresh_engine = tresh
        self.dredge_engine = dredge
        logger.info("All cognitive layers registered")
    
    # ========================================================================
    # REQUEST FLOW CONTROL
    # ========================================================================
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process request through cognitive layers
        
        Flow:
        1. Receive request at nervous system
        2. Route to Tresh (decision layer)
        3. Tresh consults GPT Sol (reasoning layer)
        4. Tresh creates plan and routes to DREDGE (execution layer)
        5. DREDGE executes and provides feedback
        6. Tresh analyzes feedback and adapts if needed
        7. Return integrated result
        """
        request_id = request.get("id", self._generate_id())
        start_time = datetime.utcnow()
        
        try:
            logger.info(f"[Nervous System] Processing request {request_id}")
            
            # Step 1: Entry point
            self._log_message(NeuroMessage("client", "nervous_system", "request", request))
            
            # Step 2: Route to Tresh with GPT Sol context
            logger.info(f"[Nervous System] Routing to Tresh decision layer")
            tresh_request = {
                **request,
                "gpt_sol_context": True  # Enable reasoning consultation
            }
            
            tresh_result = await self._call_tresh(tresh_request)
            if tresh_result.get("status") != "success":
                return tresh_result
            
            # Step 3: Extract execution plan
            execution_plan = tresh_result.get("execution_plan", [])
            logger.info(f"[Nervous System] Got execution plan with {len(execution_plan)} steps")
            
            # Step 4: Route to DREDGE for execution
            logger.info(f"[Nervous System] Routing to DREDGE execution layer")
            dredge_result = await self._call_dredge({
                "request_id": request_id,
                "plan": execution_plan,
                "context": request
            })
            
            # Step 5: Collect performance feedback
            performance_data = dredge_result.get("performance", {})
            
            # Step 6: Optional: Trigger adaptation if significant learning needed
            if dredge_result.get("adaptation_suggested", False):
                logger.info(f"[Nervous System] DREDGE suggests strategy adaptation")
                await self._trigger_adaptation(performance_data)
            
            # Step 7: Assemble final result
            final_result = {
                "request_id": request_id,
                "status": "success",
                "decision": tresh_result.get("decision"),
                "execution": dredge_result,
                "reasoning_chain": tresh_result.get("reasoning_chain", []),
                "confidence": tresh_result.get("confidence", 0.0),
                "telemetry": {
                    "total_time_ms": (datetime.utcnow() - start_time).total_seconds() * 1000,
                    "layers_used": ["tresh", "dredge", "gpt_sol"],
                    "performance_data": performance_data
                }
            }
            
            # Log result
            self._log_message(NeuroMessage("nervous_system", "client", "response", final_result))
            self._update_telemetry(final_result)
            
            return final_result
        
        except Exception as e:
            logger.error(f"[Nervous System] Request processing failed: {e}")
            return {
                "request_id": request_id,
                "status": "error",
                "error": str(e),
                "telemetry": {
                    "total_time_ms": (datetime.utcnow() - start_time).total_seconds() * 1000
                }
            }
    
    # ========================================================================
    # LAYER COMMUNICATION
    # ========================================================================
    
    async def _call_tresh(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Call Tresh decision layer"""
        if not self.tresh_engine:
            logger.error("Tresh engine not registered")
            return {"status": "error", "error": "Tresh not available"}
        
        try:
            result = await self.tresh_engine.make_decision(request, self.gpt_sol_engine)
            return result
        except Exception as e:
            logger.error(f"Tresh call failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _call_dredge(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Call DREDGE execution layer"""
        if not self.dredge_engine:
            logger.warning("DREDGE engine not registered, simulating")
            # Simulate DREDGE response for demo
            return {
                "status": "success",
                "execution_id": request.get("request_id"),
                "steps_executed": len(request.get("plan", [])),
                "performance": {
                    "execution_time_ms": 100.0,
                    "resource_usage": "normal",
                    "success_rate": 0.95
                }
            }
        
        try:
            # Call DREDGE's execute method
            result = await self.dredge_engine.execute_plan(request)
            return result
        except Exception as e:
            logger.error(f"DREDGE call failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _trigger_adaptation(self, performance_data: Dict[str, Any]):
        """Trigger strategy adaptation in Tresh"""
        if not self.tresh_engine:
            return
        
        try:
            result = await self.tresh_engine.adapt_strategy(performance_data, self.gpt_sol_engine)
            logger.info(f"[Adaptation] Strategy updated: {result.get('status')}")
        except Exception as e:
            logger.error(f"Adaptation failed: {e}")
    
    # ========================================================================
    # TELEMETRY & MONITORING
    # ========================================================================
    
    def _log_message(self, message: NeuroMessage):
        """Log inter-layer message"""
        self.message_log.append(message.to_dict())
        # Keep only recent messages
        if len(self.message_log) > 1000:
            self.message_log = self.message_log[-1000:]
    
    def _update_telemetry(self, result: Dict[str, Any]):
        """Update system telemetry"""
        self.telemetry_data["requests_processed"] += 1
        
        if "telemetry" in result:
            total_time = result["telemetry"].get("total_time_ms", 0)
            # Update average
            prev_avg = self.telemetry_data["avg_response_time"]
            count = self.telemetry_data["requests_processed"]
            self.telemetry_data["avg_response_time"] = (prev_avg * (count - 1) + total_time) / count
    
    def get_telemetry(self) -> Dict[str, Any]:
        """Get system telemetry"""
        return {
            **self.telemetry_data,
            "message_log_size": len(self.message_log),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def get_message_log(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent messages"""
        return self.message_log[-limit:]
    
    def get_layer_status(self) -> Dict[str, Any]:
        """Get status of all layers"""
        return {
            "gpt_sol": {"registered": self.gpt_sol_engine is not None, "status": "active"},
            "tresh": {"registered": self.tresh_engine is not None, "status": "active"},
            "dredge": {"registered": self.dredge_engine is not None, "status": "active"},
            "nervous_system": {"status": "operational"}
        }
    
    # ========================================================================
    # UTILITIES
    # ========================================================================
    
    def _generate_id(self) -> str:
        """Generate unique ID"""
        import uuid
        return str(uuid.uuid4())[:8]
    
    def health_check(self) -> Dict[str, Any]:
        """Perform system health check"""
        health = {
            "status": "healthy",
            "layers": self.get_layer_status(),
            "telemetry": self.get_telemetry(),
            "timestamp": datetime.utcnow().isoformat()
        }
        return health


# ============================================================================
# GLOBAL NERVOUS SYSTEM INSTANCE
# ============================================================================

cognitive_nervous_system = CognitiveNervousSystem()


# ============================================================================
# FASTAPI INTEGRATION
# ============================================================================

def setup_cognitive_nervous_system(app, gpt_sol_engine=None, tresh_engine=None):
    """Setup cognitive nervous system routes"""
    from fastapi import APIRouter, HTTPException
    
    # Register layers
    cognitive_nervous_system.register_layers(gpt_sol_engine, tresh_engine)
    
    router = APIRouter(prefix="/nervous-system", tags=["Cognitive Nervous System"])
    
    @router.post("/request")
    async def process_request(request: Dict[str, Any]):
        """Process request through all cognitive layers"""
        try:
            result = await cognitive_nervous_system.process_request(request)
            return result
        except Exception as e:
            logger.error(f"Nervous system error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/status")
    async def get_status():
        """Get nervous system status"""
        return cognitive_nervous_system.get_layer_status()
    
    @router.get("/telemetry")
    async def get_telemetry():
        """Get system telemetry"""
        return cognitive_nervous_system.get_telemetry()
    
    @router.get("/health")
    async def health_check():
        """Perform health check"""
        return cognitive_nervous_system.health_check()
    
    @router.get("/messages")
    async def get_messages(limit: int = 50):
        """Get recent inter-layer messages"""
        return {"messages": cognitive_nervous_system.get_message_log(limit)}
    
    app.include_router(router)


__all__ = [
    'CognitiveNervousSystem',
    'NeuroMessage',
    'cognitive_nervous_system',
    'setup_cognitive_nervous_system'
]
