"""
Tresh Decision & Reasoning Layer - Strategic Orchestration

Tresh Architecture:
  - Strategic decision-making and planning
  - Agent orchestration and task sequencing
  - Real-time reasoning and adaptation
  - Multi-agent coordination
  - Conflict resolution and negotiation
  - Learning and improvement feedback loops

Position in Stack:
  GPT Sol (Reasoning/Analysis) ← provides deep insights
       ↑
       ↓
  Tresh (Decision/Orchestration) ← YOUR LAYER
       ↑
       ↓
  DREDGE (Application/Execution) ← executes decisions
"""

import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
import uuid
import asyncio

logger = logging.getLogger(__name__)


# ============================================================================
# DECISION TYPES & STATES
# ============================================================================

class DecisionType(str, Enum):
    """Types of decisions Tresh can make"""
    TACTICAL = "tactical"              # Immediate, short-term
    STRATEGIC = "strategic"            # Long-term planning
    ADAPTIVE = "adaptive"              # Learning-based adjustment
    EMERGENCY = "emergency"            # Crisis response
    COLLABORATIVE = "collaborative"    # Multi-agent coordination


class DecisionState(str, Enum):
    """State of a decision"""
    PENDING = "pending"
    ANALYZING = "analyzing"
    DECIDED = "decided"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    REVERSED = "reversed"


# ============================================================================
# DECISION RECORD
# ============================================================================

class DecisionRecord:
    """Records a decision made by Tresh"""
    
    def __init__(self, decision_id: str, decision_type: DecisionType):
        self.decision_id = decision_id
        self.decision_type = decision_type
        self.state = DecisionState.PENDING
        self.created_at = datetime.utcnow()
        self.analysis: Optional[Dict] = None
        self.recommendation: Optional[Dict] = None
        self.execution_plan: Optional[List[Dict]] = None
        self.reasoning_chain: List[str] = []
        self.outcome: Optional[Dict] = None
        self.lessons_learned: List[str] = []
        self.confidence: float = 0.0
        self.risk_level: str = "medium"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "decision_id": self.decision_id,
            "decision_type": self.decision_type.value,
            "state": self.state.value,
            "created_at": self.created_at.isoformat(),
            "confidence": self.confidence,
            "risk_level": self.risk_level,
            "reasoning_chain": self.reasoning_chain,
            "lessons_learned": self.lessons_learned
        }


# ============================================================================
# TRESH ORCHESTRATION ENGINE
# ============================================================================

class TreshOrchestrationEngine:
    """
    Strategic orchestration and decision-making engine
    
    Responsibilities:
    - Request routing and sequencing
    - Agent coordination and task delegation
    - Decision analysis and recommendation
    - Conflict resolution
    - Performance monitoring
    - Learning and adaptation
    """
    
    def __init__(self):
        self.decisions: Dict[str, DecisionRecord] = {}
        self.agents: Dict[str, Dict[str, Any]] = {}
        self.task_queue: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, Any] = {
            "total_decisions": 0,
            "successful_decisions": 0,
            "failed_decisions": 0,
            "avg_decision_time": 0.0,
            "learning_loops": 0
        }
        logger.info("Tresh Orchestration Engine initialized")
    
    # ========================================================================
    # CORE DECISION METHODS
    # ========================================================================
    
    async def make_decision(self, 
                           request: Dict[str, Any],
                           gpt_sol_engine=None) -> Dict[str, Any]:
        """
        Make a strategic decision
        
        Process:
        1. Analyze request
        2. Consult GPT Sol for reasoning
        3. Formulate decision
        4. Create execution plan
        5. Return decision with reasoning
        """
        decision_id = str(uuid.uuid4())
        decision_type = DecisionType(request.get("type", "tactical"))
        
        record = DecisionRecord(decision_id, decision_type)
        record.state = DecisionState.ANALYZING
        
        try:
            # Step 1: Request analysis
            self._log_reasoning(record, "Analyzing request and context")
            analysis = await self._analyze_request(request)
            record.analysis = analysis
            
            # Step 2: Consult GPT Sol for reasoning
            if gpt_sol_engine:
                self._log_reasoning(record, "Consulting GPT Sol reasoning engine")
                sol_analysis = await gpt_sol_engine.analyze_request({
                    "problem": analysis.get("problem", ""),
                    "context": analysis.get("context", {}),
                    "constraints": request.get("constraints", [])
                })
                record.reasoning_chain.extend(sol_analysis.get("reasoning_chain", []))
                record.confidence = sol_analysis.get("confidence", 0.5)
            
            # Step 3: Formulate decision
            self._log_reasoning(record, "Formulating decision")
            decision = await self._formulate_decision(analysis)
            record.recommendation = decision
            
            # Step 4: Risk assessment
            self._log_reasoning(record, "Assessing risks")
            risk_assessment = await self._assess_decision_risks(decision)
            record.risk_level = risk_assessment.get("level", "medium")
            
            # Step 5: Create execution plan
            self._log_reasoning(record, "Creating execution plan")
            execution_plan = await self._create_execution_plan(decision, analysis)
            record.execution_plan = execution_plan
            
            record.state = DecisionState.DECIDED
            
            result = {
                "decision_id": decision_id,
                "status": "success",
                "decision": decision,
                "execution_plan": execution_plan,
                "confidence": record.confidence,
                "risk_level": record.risk_level,
                "reasoning_chain": record.reasoning_chain
            }
            
            self.decisions[decision_id] = record
            self._update_metrics("successful_decisions")
            
            return result
        
        except Exception as e:
            logger.error(f"Decision making failed: {e}")
            record.state = DecisionState.FAILED
            self._update_metrics("failed_decisions")
            return {
                "decision_id": decision_id,
                "status": "error",
                "error": str(e)
            }
    
    async def orchestrate_agents(self,
                                agents: List[str],
                                task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Orchestrate multiple agents for a task
        
        Responsibilities:
        - Agent capability matching
        - Task sequencing
        - Parallel vs sequential execution
        - Result aggregation
        - Conflict resolution
        """
        orchestration_id = str(uuid.uuid4())
        
        self._log_reasoning(None, f"Orchestrating {len(agents)} agents for task")
        
        try:
            # Step 1: Task decomposition
            sub_tasks = await self._decompose_task(task)
            self._log_reasoning(None, f"Decomposed into {len(sub_tasks)} sub-tasks")
            
            # Step 2: Agent capability matching
            assignments = await self._match_agents_to_tasks(agents, sub_tasks)
            self._log_reasoning(None, f"Assigned {len(assignments)} tasks to agents")
            
            # Step 3: Execution sequencing
            execution_order = await self._determine_execution_order(assignments)
            self._log_reasoning(None, f"Determined execution order ({len(execution_order)} steps)")
            
            # Step 4: Execute and monitor
            results = await self._execute_orchestrated_tasks(execution_order, assignments)
            
            # Step 5: Aggregate and analyze results
            aggregated = await self._aggregate_results(results)
            
            return {
                "orchestration_id": orchestration_id,
                "status": "success",
                "agents_used": len(agents),
                "tasks_completed": len(results),
                "aggregated_results": aggregated
            }
        
        except Exception as e:
            logger.error(f"Orchestration failed: {e}")
            return {
                "orchestration_id": orchestration_id,
                "status": "error",
                "error": str(e)
            }
    
    async def adapt_strategy(self, 
                            performance_data: Dict[str, Any],
                            gpt_sol_engine=None) -> Dict[str, Any]:
        """
        Adapt strategy based on performance feedback
        
        Learning loop:
        1. Analyze performance
        2. Identify patterns
        3. Consult GPT Sol for insights
        4. Formulate improvements
        5. Update strategy
        """
        adaptation_id = str(uuid.uuid4())
        
        try:
            # Step 1: Performance analysis
            performance_analysis = await self._analyze_performance(performance_data)
            
            # Step 2: Pattern identification
            patterns = await self._identify_patterns(performance_analysis)
            
            # Step 3: GPT Sol consultation
            if gpt_sol_engine:
                insights = await gpt_sol_engine.analyze_request({
                    "problem": "Improve strategy based on performance",
                    "context": {
                        "patterns": patterns,
                        "metrics": self.performance_metrics
                    }
                })
            else:
                insights = {}
            
            # Step 4: Improvement formulation
            improvements = await self._formulate_improvements(patterns)
            
            # Step 5: Strategy update
            updated_strategy = await self._update_strategy(improvements)
            
            # Step 6: Record learning
            self.performance_metrics["learning_loops"] += 1
            
            return {
                "adaptation_id": adaptation_id,
                "status": "success",
                "improvements": improvements,
                "updated_strategy": updated_strategy,
                "insights": insights
            }
        
        except Exception as e:
            logger.error(f"Strategy adaptation failed: {e}")
            return {
                "adaptation_id": adaptation_id,
                "status": "error",
                "error": str(e)
            }
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    async def _analyze_request(self, request: Dict) -> Dict:
        """Analyze incoming request"""
        return {
            "problem": request.get("problem", ""),
            "context": request.get("context", {}),
            "constraints": request.get("constraints", []),
            "priority": request.get("priority", "normal"),
            "urgency": request.get("urgency", "normal")
        }
    
    async def _formulate_decision(self, analysis: Dict) -> Dict:
        """Formulate strategic decision"""
        return {
            "decision": "Recommended action based on analysis",
            "rationale": "Clear reasoning for decision",
            "alternatives": ["Alternative 1", "Alternative 2"],
            "expected_outcome": "Positive result from decision"
        }
    
    async def _assess_decision_risks(self, decision: Dict) -> Dict:
        """Assess risks of decision"""
        return {
            "level": "medium",
            "key_risks": ["Risk 1", "Risk 2"],
            "mitigation": "Mitigation strategies"
        }
    
    async def _create_execution_plan(self, decision: Dict, analysis: Dict) -> List[Dict]:
        """Create execution plan for decision"""
        return [
            {
                "step": 1,
                "action": "Initial action",
                "responsible_agent": "DREDGE",
                "timeline": "Immediate",
                "success_criteria": "Clear criteria"
            },
            {
                "step": 2,
                "action": "Follow-up action",
                "responsible_agent": "DREDGE",
                "timeline": "Short-term",
                "success_criteria": "Verification criteria"
            }
        ]
    
    async def _decompose_task(self, task: Dict) -> List[Dict]:
        """Decompose task into sub-tasks"""
        return [
            {"sub_task": 1, "description": "Sub-task 1"},
            {"sub_task": 2, "description": "Sub-task 2"},
            {"sub_task": 3, "description": "Sub-task 3"}
        ]
    
    async def _match_agents_to_tasks(self, agents: List[str], tasks: List[Dict]) -> List[Dict]:
        """Match agents to tasks based on capabilities"""
        return [
            {"agent": agents[0], "task": tasks[0]} if len(agents) > 0 else None
        ]
    
    async def _determine_execution_order(self, assignments: List[Dict]) -> List[Dict]:
        """Determine optimal execution order"""
        return assignments
    
    async def _execute_orchestrated_tasks(self, execution_order: List, assignments: List) -> Dict:
        """Execute orchestrated tasks"""
        return {"executed_tasks": len(execution_order)}
    
    async def _aggregate_results(self, results: Dict) -> Dict:
        """Aggregate results from multiple agents"""
        return {"aggregation": "Complete"}
    
    async def _analyze_performance(self, data: Dict) -> Dict:
        """Analyze performance data"""
        return {"analysis": "Performance metrics analyzed"}
    
    async def _identify_patterns(self, analysis: Dict) -> List[str]:
        """Identify patterns in performance"""
        return ["Pattern 1", "Pattern 2", "Pattern 3"]
    
    async def _formulate_improvements(self, patterns: List[str]) -> List[Dict]:
        """Formulate improvements based on patterns"""
        return [
            {"improvement": "Improvement 1", "impact": "high"},
            {"improvement": "Improvement 2", "impact": "medium"}
        ]
    
    async def _update_strategy(self, improvements: List[Dict]) -> Dict:
        """Update strategy based on improvements"""
        return {"updated": True, "improvements_applied": len(improvements)}
    
    # ========================================================================
    # UTILITIES
    # ========================================================================
    
    def _log_reasoning(self, record: Optional[DecisionRecord], message: str):
        """Log reasoning step"""
        logger.info(f"[Tresh] {message}")
        if record:
            record.reasoning_chain.append(message)
    
    def _update_metrics(self, metric_name: str):
        """Update performance metrics"""
        if metric_name in self.performance_metrics:
            if "total" in metric_name or "_count" in metric_name or "loops" in metric_name:
                self.performance_metrics[metric_name] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return self.performance_metrics.copy()
    
    def get_decision_history(self, limit: int = 10) -> List[Dict]:
        """Get decision history"""
        decisions = list(self.decisions.values())[-limit:]
        return [d.to_dict() for d in decisions]
    
    def register_agent(self, agent_name: str, capabilities: List[str]):
        """Register an agent"""
        self.agents[agent_name] = {
            "name": agent_name,
            "capabilities": capabilities,
            "registered_at": datetime.utcnow().isoformat()
        }
        logger.info(f"Agent {agent_name} registered with {len(capabilities)} capabilities")
    
    def get_registered_agents(self) -> Dict[str, Dict]:
        """Get registered agents"""
        return self.agents.copy()


# ============================================================================
# GLOBAL ENGINE INSTANCE
# ============================================================================

tresh_engine = TreshOrchestrationEngine()


# ============================================================================
# FASTAPI INTEGRATION
# ============================================================================

def setup_tresh(app, gpt_sol_engine=None):
    """Setup Tresh routes"""
    from fastapi import APIRouter, HTTPException
    
    router = APIRouter(prefix="/tresh", tags=["Tresh Decision Layer"])
    
    @router.post("/decide")
    async def make_decision(request: Dict[str, Any]):
        """Make strategic decision"""
        try:
            result = await tresh_engine.make_decision(request, gpt_sol_engine)
            return result
        except Exception as e:
            logger.error(f"Decision error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.post("/orchestrate")
    async def orchestrate_agents(request: Dict[str, Any]):
        """Orchestrate agents for task"""
        try:
            agents = request.get("agents", [])
            task = request.get("task", {})
            result = await tresh_engine.orchestrate_agents(agents, task)
            return result
        except Exception as e:
            logger.error(f"Orchestration error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.post("/adapt")
    async def adapt_strategy(request: Dict[str, Any]):
        """Adapt strategy based on performance"""
        try:
            performance_data = request.get("performance_data", {})
            result = await tresh_engine.adapt_strategy(performance_data, gpt_sol_engine)
            return result
        except Exception as e:
            logger.error(f"Adaptation error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/metrics")
    async def get_metrics():
        """Get performance metrics"""
        return tresh_engine.get_metrics()
    
    @router.get("/history")
    async def get_history(limit: int = 10):
        """Get decision history"""
        return {"decisions": tresh_engine.get_decision_history(limit)}
    
    @router.post("/register-agent")
    async def register_agent(request: Dict[str, Any]):
        """Register an agent"""
        agent_name = request.get("name", "")
        capabilities = request.get("capabilities", [])
        tresh_engine.register_agent(agent_name, capabilities)
        return {"status": "registered", "agent": agent_name}
    
    @router.get("/agents")
    async def get_agents():
        """Get registered agents"""
        return {"agents": tresh_engine.get_registered_agents()}
    
    app.include_router(router)


__all__ = [
    'TreshOrchestrationEngine',
    'DecisionRecord',
    'DecisionType',
    'DecisionState',
    'tresh_engine',
    'setup_tresh'
]
