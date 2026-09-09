"""
webMCP - Web-based Model Context Protocol Agentic System
Based on DREDGE orchestration architecture with polished visual identity
Ports: 3000 (webUI), 3002 (MCP), 8001 (DREDGE), 8080 (Gateway)
"""

from fastapi import FastAPI, WebSocket, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import logging
import json
from typing import Dict, Any, List
from datetime import datetime
from pathlib import Path
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# webMCP APPLICATION (Port 3000)
# ============================================================================

app = FastAPI(
    title="webMCP - Agentic System",
    description="Web-based Model Context Protocol with intelligent agents",
    version="1.0.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# AGENT STATE & ORCHESTRATION
# ============================================================================

class AgentOrchestrator:
    """Manages agent lifecycle and state based on DREDGE orchestration"""
    
    def __init__(self):
        self.agents: Dict[str, Dict[str, Any]] = {}
        self.event_log: List[Dict[str, Any]] = []
        self.state: Dict[str, Any] = {
            "active_agents": 0,
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "system_status": "online"
        }
    
    def register_agent(self, agent_id: str, agent_type: str, config: Dict[str, Any] = None):
        """Register a new agent"""
        self.agents[agent_id] = {
            "id": agent_id,
            "type": agent_type,
            "status": "idle",
            "created_at": datetime.utcnow().isoformat(),
            "config": config or {},
            "tasks_completed": 0,
            "last_activity": datetime.utcnow().isoformat()
        }
        self.state["active_agents"] = len(self.agents)
        
        self._emit_event("agent_registered", {
            "agent_id": agent_id,
            "agent_type": agent_type
        })
        
        logger.info(f"Agent registered: {agent_id} ({agent_type})")
    
    def emit_task(self, agent_id: str, task_id: str, task_type: str, payload: Dict[str, Any]):
        """Emit a task to an agent"""
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        self.agents[agent_id]["status"] = "busy"
        self.agents[agent_id]["last_activity"] = datetime.utcnow().isoformat()
        self.state["total_tasks"] += 1
        
        self._emit_event("task_emitted", {
            "agent_id": agent_id,
            "task_id": task_id,
            "task_type": task_type,
            "payload": payload
        })
    
    def complete_task(self, agent_id: str, task_id: str, result: Dict[str, Any]):
        """Mark task as complete"""
        if agent_id in self.agents:
            self.agents[agent_id]["tasks_completed"] += 1
            self.agents[agent_id]["status"] = "idle"
            self.agents[agent_id]["last_activity"] = datetime.utcnow().isoformat()
        
        self.state["completed_tasks"] += 1
        
        self._emit_event("task_completed", {
            "agent_id": agent_id,
            "task_id": task_id,
            "result": result
        })
    
    def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit an event to the log"""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": event_type,
            "data": data
        }
        self.event_log.append(event)
        
        # Keep only last 1000 events
        if len(self.event_log) > 1000:
            self.event_log = self.event_log[-1000:]

orchestrator = AgentOrchestrator()

# Pre-register system agents
orchestrator.register_agent("analysis-agent", "analyzer", {"timeout": 30})
orchestrator.register_agent("research-agent", "researcher", {"timeout": 60})
orchestrator.register_agent("planning-agent", "planner", {"timeout": 45})
orchestrator.register_agent("execution-agent", "executor", {"timeout": 120})

# ============================================================================
# CORE ROUTES
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve webMCP UI"""
    return await get_webmcp_ui()

@app.get("/api/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "service": "webMCP",
        "port": 3000,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/status")
async def status():
    """System status"""
    return {
        "status": "operational",
        "system": orchestrator.state,
        "agents": {
            agent_id: {
                "status": agent["status"],
                "tasks_completed": agent["tasks_completed"],
                "type": agent["type"]
            }
            for agent_id, agent in orchestrator.agents.items()
        }
    }

@app.get("/api/agents")
async def list_agents():
    """List all agents"""
    return {
        "agents": orchestrator.agents,
        "total": len(orchestrator.agents)
    }

@app.get("/api/agents/{agent_id}")
async def get_agent(agent_id: str):
    """Get agent details"""
    if agent_id not in orchestrator.agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    return orchestrator.agents[agent_id]

@app.post("/api/agents/{agent_id}/task")
async def emit_task(agent_id: str, request: Request):
    """Emit task to agent"""
    body = await request.json()
    
    task_id = body.get("task_id", f"task-{datetime.utcnow().timestamp()}")
    task_type = body.get("task_type", "generic")
    payload = body.get("payload", {})
    
    try:
        orchestrator.emit_task(agent_id, task_id, task_type, payload)
        return {
            "status": "success",
            "task_id": task_id,
            "agent_id": agent_id,
            "message": "Task emitted successfully"
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/api/tasks/{task_id}/complete")
async def complete_task(task_id: str, request: Request):
    """Mark task as complete"""
    body = await request.json()
    agent_id = body.get("agent_id")
    result = body.get("result", {})
    
    if not agent_id:
        raise HTTPException(status_code=400, detail="agent_id required")
    
    orchestrator.complete_task(agent_id, task_id, result)
    
    return {
        "status": "success",
        "task_id": task_id,
        "message": "Task marked as complete"
    }

@app.get("/api/events")
async def get_events(limit: int = 100):
    """Get event log"""
    events = orchestrator.event_log[-limit:]
    return {
        "events": events,
        "total": len(orchestrator.event_log),
        "returned": len(events)
    }

@app.websocket("/ws/events")
async def websocket_events(websocket: WebSocket):
    """WebSocket stream of events"""
    await websocket.accept()
    last_index = len(orchestrator.event_log)
    
    try:
        while True:
            # Send new events
            if len(orchestrator.event_log) > last_index:
                new_events = orchestrator.event_log[last_index:]
                for event in new_events:
                    await websocket.send_json(event)
                last_index = len(orchestrator.event_log)
            
            await asyncio.sleep(0.5)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")

# ============================================================================
# UI HTML
# ============================================================================

async def get_webmcp_ui() -> str:
    """Generate webMCP UI HTML"""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>webMCP - Agentic System</title>
    
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        
        header h1 {
            font-size: 2.5em;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 10px;
        }
        
        header p {
            color: #666;
            font-size: 1.1em;
        }
        
        .status-bar {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        
        .status-item {
            background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
            border-left: 4px solid #667eea;
            padding: 15px;
            border-radius: 8px;
        }
        
        .status-item h3 {
            color: #667eea;
            font-size: 0.9em;
            margin-bottom: 8px;
        }
        
        .status-item .value {
            font-size: 2em;
            font-weight: bold;
            color: #764ba2;
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .card {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(0,0,0,0.2);
        }
        
        .card h2 {
            font-size: 1.3em;
            margin-bottom: 15px;
            color: #667eea;
        }
        
        .agent-list {
            list-style: none;
        }
        
        .agent-item {
            background: linear-gradient(135deg, #667eea10 0%, #764ba210 100%);
            padding: 12px;
            margin-bottom: 10px;
            border-radius: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-left: 3px solid #667eea;
        }
        
        .agent-name {
            font-weight: 600;
            color: #333;
        }
        
        .agent-status {
            background: #4caf50;
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85em;
        }
        
        .agent-status.busy {
            background: #ff9800;
        }
        
        .event-stream {
            background: linear-gradient(135deg, #667eea10 0%, #764ba210 100%);
            border-radius: 12px;
            padding: 20px;
            max-height: 400px;
            overflow-y: auto;
            font-family: 'Courier New', monospace;
            font-size: 0.85em;
        }
        
        .event {
            padding: 8px;
            margin-bottom: 8px;
            background: white;
            border-left: 3px solid #667eea;
            border-radius: 4px;
            animation: slideIn 0.3s ease;
        }
        
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateX(-10px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        
        .event-type {
            color: #667eea;
            font-weight: bold;
        }
        
        .event-time {
            color: #999;
            font-size: 0.9em;
        }
        
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
            transition: transform 0.2s ease;
        }
        
        button:hover {
            transform: translateY(-2px);
        }
        
        .controls {
            display: flex;
            gap: 10px;
            margin-top: 15px;
        }
        
        input[type="text"], input[type="number"], select {
            padding: 10px;
            border: 2px solid #ddd;
            border-radius: 6px;
            font-size: 1em;
        }
        
        input[type="text"]:focus, input[type="number"]:focus, select:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .footer {
            text-align: center;
            color: white;
            margin-top: 50px;
            padding: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>webMCP - Agentic System</h1>
            <p>Web-based Model Context Protocol with intelligent orchestration</p>
            
            <div class="status-bar">
                <div class="status-item">
                    <h3>Active Agents</h3>
                    <div class="value" id="active-agents">0</div>
                </div>
                <div class="status-item">
                    <h3>Total Tasks</h3>
                    <div class="value" id="total-tasks">0</div>
                </div>
                <div class="status-item">
                    <h3>Completed</h3>
                    <div class="value" id="completed-tasks">0</div>
                </div>
                <div class="status-item">
                    <h3>System Status</h3>
                    <div class="value" id="system-status" style="color: #4caf50;">ONLINE</div>
                </div>
            </div>
        </header>
        
        <div class="grid">
            <div class="card">
                <h2>Registered Agents</h2>
                <ul class="agent-list" id="agent-list">
                    <li class="agent-item">
                        <span>Loading...</span>
                    </li>
                </ul>
            </div>
            
            <div class="card">
                <h2>Quick Actions</h2>
                <div style="display: flex; flex-direction: column; gap: 10px;">
                    <select id="agent-select">
                        <option>Select Agent...</option>
                    </select>
                    <input type="text" id="task-type" placeholder="Task Type" value="analysis">
                    <input type="text" id="task-payload" placeholder="Payload (JSON)" value='{"query":"test"}'>
                    <button onclick="emitTask()">Emit Task</button>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>Event Stream</h2>
            <div class="event-stream" id="event-stream"></div>
        </div>
    </div>
    
    <div class="footer">
        <p>webMCP v1.0 | Ports: 3000 (UI) | 3002 (MCP) | 8001 (DREDGE) | 8080 (Gateway)</p>
    </div>
    
    <script>
        // Load initial data
        async function loadStatus() {
            const response = await fetch('/api/status');
            const data = await response.json();
            
            document.getElementById('active-agents').textContent = data.system.active_agents;
            document.getElementById('total-tasks').textContent = data.system.total_tasks;
            document.getElementById('completed-tasks').textContent = data.system.completed_tasks;
        }
        
        async function loadAgents() {
            const response = await fetch('/api/agents');
            const data = await response.json();
            
            const list = document.getElementById('agent-list');
            const select = document.getElementById('agent-select');
            
            list.innerHTML = '';
            select.innerHTML = '<option>Select Agent...</option>';
            
            for (const agent of Object.values(data.agents)) {
                const item = document.createElement('li');
                item.className = 'agent-item';
                item.innerHTML = `
                    <span class="agent-name">${agent.id}</span>
                    <span class="agent-status ${agent.status === 'busy' ? 'busy' : ''}">${agent.status}</span>
                `;
                list.appendChild(item);
                
                const option = document.createElement('option');
                option.value = agent.id;
                option.textContent = agent.id;
                select.appendChild(option);
            }
        }
        
        async function emitTask() {
            const agentId = document.getElementById('agent-select').value;
            const taskType = document.getElementById('task-type').value;
            const payload = JSON.parse(document.getElementById('task-payload').value || '{}');
            
            if (!agentId || agentId === 'Select Agent...') {
                alert('Please select an agent');
                return;
            }
            
            const response = await fetch(`/api/agents/${agentId}/task`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    task_type: taskType,
                    payload: payload
                })
            });
            
            const data = await response.json();
            console.log('Task emitted:', data);
        }
        
        // WebSocket for events
        const eventStream = document.getElementById('event-stream');
        const ws = new WebSocket('ws://' + window.location.host + '/ws/events');
        
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            const eventEl = document.createElement('div');
            eventEl.className = 'event';
            eventEl.innerHTML = `
                <span class="event-type">${data.type}</span>
                <span class="event-time">${new Date(data.timestamp).toLocaleTimeString()}</span>
                <br>
                <small>${JSON.stringify(data.data).substring(0, 100)}...</small>
            `;
            
            eventStream.insertBefore(eventEl, eventStream.firstChild);
            
            // Keep only last 50 events visible
            while (eventStream.children.length > 50) {
                eventStream.removeChild(eventStream.lastChild);
            }
            
            // Update status
            loadStatus();
        };
        
        // Load data periodically
        loadStatus();
        loadAgents();
        setInterval(() => {
            loadStatus();
            loadAgents();
        }, 5000);
    </script>
</body>
</html>
"""

# ============================================================================
# STARTUP
# ============================================================================

@app.on_event("startup")
async def startup():
    """Startup initialization"""
    logger.info("=" * 80)
    logger.info("webMCP - Web-based Agentic System")
    logger.info("=" * 80)
    logger.info("Port: 3000")
    logger.info("UI: http://127.0.0.1:3000/")
    logger.info("API: http://127.0.0.1:3000/api/")
    logger.info("Docs: http://127.0.0.1:3000/api/docs")
    logger.info("")
    logger.info("System Agents:")
    for agent_id, agent in orchestrator.agents.items():
        logger.info(f"  - {agent_id} ({agent['type']})")
    logger.info("=" * 80)

# ============================================================================
# EXPORT
# ============================================================================

__all__ = ["app", "orchestrator"]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=3000, log_level="info")
