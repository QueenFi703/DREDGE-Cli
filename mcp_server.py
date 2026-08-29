"""
MCP (Model Context Protocol) Server for DREDGE Studio
Port: 3002
Provides tool definitions and capabilities for Claude and other LLM clients
"""

import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Dict, Any
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# MCP SERVER APPLICATION
# ============================================================================

mcp_app = FastAPI(
    title="DREDGE MCP Server",
    description="Model Context Protocol Server for AI/Claude Integration",
    version="1.0.0",
    docs_url="/docs",
    openapi_url="/openapi.json"
)

# ============================================================================
# MCP TOOLS DEFINITION
# ============================================================================

MCP_TOOLS = [
    {
        "name": "analyze_request",
        "description": "Perform multi-modal reasoning analysis on a request",
        "inputSchema": {
            "type": "object",
            "properties": {
                "problem": {
                    "type": "string",
                    "description": "The problem to analyze"
                },
                "context": {
                    "type": "object",
                    "description": "Additional context information"
                },
                "reasoning_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Types of reasoning: deductive, inductive, abductive, causal"
                }
            },
            "required": ["problem"]
        }
    },
    {
        "name": "make_decision",
        "description": "Make a strategic decision using the Tresh decision layer",
        "inputSchema": {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "description": "Decision type: tactical, strategic, adaptive, emergency, collaborative"
                },
                "problem": {
                    "type": "string",
                    "description": "The problem requiring a decision"
                },
                "options": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Available options to choose from"
                },
                "constraints": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Constraints to consider"
                }
            },
            "required": ["type", "problem"]
        }
    },
    {
        "name": "execute_plan",
        "description": "Execute a plan using the DREDGE execution layer",
        "inputSchema": {
            "type": "object",
            "properties": {
                "plan_id": {
                    "type": "string",
                    "description": "Unique plan identifier"
                },
                "steps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Steps to execute"
                },
                "resources": {
                    "type": "object",
                    "description": "Resources needed"
                }
            },
            "required": ["plan_id", "steps"]
        }
    },
    {
        "name": "cognitive_pipeline",
        "description": "Run full cognitive pipeline (reasoning → decision → execution)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "request_type": {
                    "type": "string",
                    "description": "Type of request"
                },
                "problem": {
                    "type": "string",
                    "description": "The problem to solve"
                },
                "context": {
                    "type": "object",
                    "description": "Context information"
                },
                "include_reasoning": {
                    "type": "boolean",
                    "description": "Include detailed reasoning chains"
                },
                "include_telemetry": {
                    "type": "boolean",
                    "description": "Include performance telemetry"
                }
            },
            "required": ["request_type", "problem"]
        }
    },
    {
        "name": "get_system_status",
        "description": "Get current system status and health metrics",
        "inputSchema": {
            "type": "object",
            "properties": {
                "include_metrics": {
                    "type": "boolean",
                    "description": "Include detailed metrics"
                },
                "include_adapters": {
                    "type": "boolean",
                    "description": "Include adapter status"
                }
            }
        }
    },
    {
        "name": "manage_api_keys",
        "description": "Create, list, and manage API keys",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Action: create, list, revoke, rotate"
                },
                "key_name": {
                    "type": "string",
                    "description": "Name for the API key"
                },
                "tier": {
                    "type": "string",
                    "description": "API key tier: starter, professional, enterprise"
                }
            },
            "required": ["action"]
        }
    },
    {
        "name": "monitor_performance",
        "description": "Monitor and analyze API performance",
        "inputSchema": {
            "type": "object",
            "properties": {
                "metric": {
                    "type": "string",
                    "description": "Metric to monitor: latency, throughput, error_rate"
                },
                "time_window": {
                    "type": "string",
                    "description": "Time window: 1h, 6h, 24h, 7d"
                },
                "include_forecast": {
                    "type": "boolean",
                    "description": "Include performance forecast"
                }
            }
        }
    },
    {
        "name": "query_documentation",
        "description": "Query API documentation and guides",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Documentation query"
                },
                "section": {
                    "type": "string",
                    "description": "Documentation section: api, architecture, authentication, deployment"
                }
            },
            "required": ["query"]
        }
    }
]

# ============================================================================
# MCP CAPABILITIES
# ============================================================================

MCP_CAPABILITIES = {
    "cognitive_layers": {
        "gpt_sol": {
            "name": "GPT Sol Reasoning Engine",
            "description": "Multi-modal reasoning with deductive, inductive, abductive, and causal analysis",
            "port": 8080,
            "endpoints": [
                "POST /gpt-sol/analyze",
                "POST /gpt-sol/strategic-plan",
                "POST /gpt-sol/ethical-analysis",
                "GET /gpt-sol/state"
            ],
            "capabilities": [
                "Deductive reasoning",
                "Inductive reasoning",
                "Abductive reasoning",
                "Causal analysis",
                "Ethical assessment",
                "Confidence scoring"
            ]
        },
        "tresh": {
            "name": "Tresh Decision Layer",
            "description": "Strategic decision-making with multi-agent orchestration",
            "port": 8080,
            "endpoints": [
                "POST /tresh/decide",
                "POST /tresh/orchestrate",
                "POST /tresh/adapt",
                "GET /tresh/metrics"
            ],
            "capabilities": [
                "Strategic decisions",
                "Tactical decisions",
                "Adaptive decisions",
                "Emergency decisions",
                "Agent orchestration",
                "Performance learning",
                "Strategy adaptation"
            ]
        },
        "dredge": {
            "name": "DREDGE Execution Layer",
            "description": "Application execution with resource management and telemetry",
            "port": 8001,
            "endpoints": [
                "POST /execute/plan",
                "GET /execute/status",
                "GET /execute/telemetry",
                "POST /execute/rollback"
            ],
            "capabilities": [
                "Plan execution",
                "Resource management",
                "Performance telemetry",
                "Error handling",
                "Graceful degradation",
                "Rollback support"
            ]
        }
    },
    "security": {
        "authentication": [
            "Email/password login",
            "OAuth (GitHub, Google)",
            "API key management",
            "Session management",
            "2FA support (planned)"
        ],
        "protection": [
            "CORS restrictions",
            "Rate limiting",
            "Input validation",
            "SQL injection prevention",
            "XSS protection",
            "CSRF protection",
            "Security headers"
        ],
        "audit": [
            "Request logging",
            "Performance tracking",
            "Error monitoring",
            "Access control",
            "Compliance reporting"
        ]
    },
    "features": {
        "dashboard": {
            "name": "DREDGE Studio Dashboard",
            "port": 8080,
            "url": "http://127.0.0.1:8080/dashboard",
            "sections": [
                "Overview (metrics, status)",
                "Cognitive Architecture (layer info)",
                "Models (deployment, management)",
                "Insights (recommendations, patterns)",
                "Analytics (performance, usage)",
                "Settings (user, API keys)"
            ]
        },
        "api_documentation": {
            "name": "API Documentation",
            "port": 8080,
            "url": "http://127.0.0.1:8080/swagger",
            "includes": [
                "Interactive API explorer",
                "Endpoint documentation",
                "Request/response examples",
                "Authentication details",
                "Rate limit information",
                "Error codes reference"
            ]
        },
        "analytics": {
            "name": "Real-time Analytics",
            "port": 8080,
            "features": [
                "Request tracking",
                "Latency monitoring",
                "Error rate analysis",
                "Performance forecasting",
                "Usage patterns",
                "Cost estimation"
            ]
        },
        "monitoring": {
            "name": "System Monitoring",
            "port": 8080,
            "endpoints": [
                "GET /health - Basic health",
                "GET /health/detailed - Detailed status",
                "GET /health/readiness - K8s readiness",
                "GET /health/liveness - K8s liveness",
                "GET /status - Gateway status"
            ]
        }
    },
    "integration": {
        "mcp_server": {
            "name": "MCP (Model Context Protocol)",
            "port": 3002,
            "url": "http://127.0.0.1:3002",
            "tools_available": len(MCP_TOOLS),
            "capabilities": [
                "Tool invocation",
                "Resource access",
                "Sampling support",
                "Logging support",
                "Async operations"
            ]
        },
        "gateway": {
            "name": "API Gateway",
            "port": 8080,
            "url": "http://127.0.0.1:8080",
            "features": [
                "Request routing",
                "Load balancing",
                "Rate limiting",
                "Caching",
                "Compression"
            ]
        },
        "dredge_server": {
            "name": "DREDGE Server",
            "port": 8001,
            "url": "http://127.0.0.1:8001",
            "features": [
                "Execution engine",
                "Resource manager",
                "Telemetry collector",
                "Performance optimizer"
            ]
        }
    }
}

# ============================================================================
# MCP ENDPOINTS
# ============================================================================

@mcp_app.get("/", tags=["Info"])
async def mcp_root() -> Dict[str, Any]:
    """MCP Server root - overview of capabilities"""
    return {
        "name": "DREDGE MCP Server",
        "version": "1.0.0",
        "port": 3002,
        "protocol": "Model Context Protocol",
        "tools_available": len(MCP_TOOLS),
        "documentation": "/docs",
        "capabilities": "/capabilities",
        "tools": "/tools"
    }

@mcp_app.get("/tools", tags=["Tools"])
async def list_tools() -> Dict[str, Any]:
    """List all available MCP tools"""
    return {
        "status": "success",
        "total_tools": len(MCP_TOOLS),
        "tools": MCP_TOOLS
    }

@mcp_app.get("/tools/{tool_name}", tags=["Tools"])
async def get_tool(tool_name: str) -> Dict[str, Any]:
    """Get specific tool definition"""
    for tool in MCP_TOOLS:
        if tool["name"] == tool_name:
            return {"status": "success", "tool": tool}
    
    raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")

@mcp_app.post("/tools/{tool_name}/invoke", tags=["Tools"])
async def invoke_tool(tool_name: str, request: Request) -> Dict[str, Any]:
    """Invoke a tool"""
    body = await request.json()
    
    # Log tool invocation
    logger.info(f"[MCP] Tool invoked: {tool_name}")
    
    # Route to appropriate backend
    return {
        "status": "success",
        "tool": tool_name,
        "message": f"Tool '{tool_name}' invocation routed to backend",
        "params": body
    }

@mcp_app.get("/capabilities", tags=["Capabilities"])
async def get_capabilities() -> Dict[str, Any]:
    """Get all DREDGE system capabilities"""
    return {
        "status": "success",
        "system": "DREDGE Studio",
        "version": "2.5.0",
        "capabilities": MCP_CAPABILITIES,
        "mcp_port": 3002,
        "gateway_port": 8080,
        "dredge_port": 8001
    }

@mcp_app.get("/capabilities/cognitive", tags=["Capabilities"])
async def get_cognitive_capabilities() -> Dict[str, Any]:
    """Get cognitive layers capabilities"""
    return {
        "status": "success",
        "layers": MCP_CAPABILITIES["cognitive_layers"]
    }

@mcp_app.get("/capabilities/security", tags=["Capabilities"])
async def get_security_capabilities() -> Dict[str, Any]:
    """Get security capabilities"""
    return {
        "status": "success",
        "security": MCP_CAPABILITIES["security"]
    }

@mcp_app.get("/capabilities/features", tags=["Capabilities"])
async def get_features_capabilities() -> Dict[str, Any]:
    """Get feature capabilities"""
    return {
        "status": "success",
        "features": MCP_CAPABILITIES["features"]
    }

@mcp_app.get("/capabilities/integration", tags=["Capabilities"])
async def get_integration_capabilities() -> Dict[str, Any]:
    """Get integration capabilities"""
    return {
        "status": "success",
        "integration": MCP_CAPABILITIES["integration"]
    }

@mcp_app.get("/status", tags=["Status"])
async def mcp_status() -> Dict[str, Any]:
    """MCP server status"""
    return {
        "status": "operational",
        "service": "DREDGE MCP Server",
        "port": 3002,
        "tools": len(MCP_TOOLS),
        "uptime": "n/a",
        "last_invocation": "n/a"
    }

@mcp_app.get("/health", tags=["Health"])
async def mcp_health() -> Dict[str, str]:
    """Health check"""
    return {"status": "healthy"}

# ============================================================================
# ERROR HANDLING
# ============================================================================

@mcp_app.exception_handler(404)
async def not_found_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=404,
        content={
            "status": "error",
            "message": "Not found",
            "path": str(request.url.path),
            "documentation": "http://127.0.0.1:3002/docs"
        }
    )

# ============================================================================
# STARTUP
# ============================================================================

@mcp_app.on_event("startup")
async def startup():
    """MCP server startup"""
    logger.info("=" * 80)
    logger.info("DREDGE MCP SERVER - Starting on port 3002")
    logger.info("=" * 80)
    logger.info(f"Tools available: {len(MCP_TOOLS)}")
    logger.info("Gateway fallback: http://127.0.0.1:8080")
    logger.info("DREDGE server: http://127.0.0.1:8001")
    logger.info("")
    logger.info("Access:")
    logger.info("  - MCP Root: http://127.0.0.1:3002/")
    logger.info("  - Tools: http://127.0.0.1:3002/tools")
    logger.info("  - Capabilities: http://127.0.0.1:3002/capabilities")
    logger.info("  - Documentation: http://127.0.0.1:3002/docs")
    logger.info("=" * 80)

# ============================================================================
# EXPORT
# ============================================================================

__all__ = ["mcp_app", "MCP_TOOLS", "MCP_CAPABILITIES"]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        mcp_app,
        host="127.0.0.1",
        port=3002,
        log_level="info"
    )
