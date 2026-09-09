#!/usr/bin/env python3
"""
DREDGE Complete System Launcher
Starts webMCP, MCP Server, DREDGE Server, and Gateway on correct ports
"""

import subprocess
import time
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# COMPLETE SERVER CONFIGURATION
# ============================================================================

SERVERS = [
    {
        "name": "webMCP",
        "script": "webmcp.py",
        "port": 3000,
        "url": "http://127.0.0.1:3000",
        "description": "Web-based Agentic System UI",
        "type": "ui"
    },
    {
        "name": "MCP Server",
        "script": "mcp_server.py",
        "port": 3002,
        "url": "http://127.0.0.1:3002",
        "description": "Model Context Protocol Server",
        "type": "service"
    },
    {
        "name": "DREDGE Server",
        "script": "dredge_server.py",
        "port": 8001,
        "url": "http://127.0.0.1:8001",
        "description": "Execution Layer and Resource Management",
        "type": "service"
    },
    {
        "name": "Gateway",
        "script": "core_gateway.py",
        "port": 8080,
        "url": "http://127.0.0.1:8080",
        "description": "API Gateway (Fallback Port)",
        "type": "gateway"
    }
]

# ============================================================================
# LAUNCHER
# ============================================================================

def start_server(server_config):
    """Start a single server"""
    name = server_config["name"]
    script = server_config["script"]
    port = server_config["port"]
    
    logger.info(f"Starting {name} on port {port}...")
    
    try:
        proc = subprocess.Popen(
            [sys.executable, script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=Path(__file__).parent
        )
        
        logger.info(f"✓ {name} started (PID: {proc.pid})")
        return proc
    except Exception as e:
        logger.error(f"✗ Failed to start {name}: {e}")
        return None

def display_banner():
    """Display system banner"""
    banner = """
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║          DREDGE Complete Agentic System - System Launcher                  ║
║                                                                            ║
║  Components:                                                               ║
║    • webMCP - Web-based Agentic System (Port 3000)                         ║
║    • MCP Server - Model Context Protocol (Port 3002)                       ║
║    • DREDGE Server - Execution Layer (Port 8001)                           ║
║    • Gateway - API Gateway (Port 8080)                                     ║
║                                                                            ║
║  Architecture:                                                             ║
║    webMCP UI → Agent Orchestrator → MCP Server → DREDGE → Gateway          ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def display_access_points():
    """Display access points"""
    print("\n" + "=" * 80)
    print("ACCESS POINTS")
    print("=" * 80)
    print()
    
    # Group by type
    ui_servers = [s for s in SERVERS if s["type"] == "ui"]
    services = [s for s in SERVERS if s["type"] == "service"]
    gateways = [s for s in SERVERS if s["type"] == "gateway"]
    
    if ui_servers:
        print("🎨 User Interface:")
        for server in ui_servers:
            print(f"  • {server['name']}: {server['url']}/")
        print()
    
    if services:
        print("⚙️  Services:")
        for server in services:
            print(f"  • {server['name']}: {server['url']}/")
            print(f"    Docs: {server['url']}/docs")
        print()
    
    if gateways:
        print("🚪 Gateways:")
        for server in gateways:
            print(f"  • {server['name']}: {server['url']}/")
            print(f"    API: {server['url']}/swagger")
        print()
    
    print("🔗 Quick Links:")
    print(f"  • webMCP Dashboard: http://127.0.0.1:3000/")
    print(f"  • MCP Tools: http://127.0.0.1:3000/mcp/tools")
    print(f"  • System Status: http://127.0.0.1:8080/status")
    print(f"  • Health Check: http://127.0.0.1:3000/api/health")
    print()

def main():
    """Launch all servers"""
    display_banner()
    
    processes = []
    
    # Start all servers
    print("Starting services...")
    print()
    for server_config in SERVERS:
        logger.info(f"Starting {server_config['name']}...")
        proc = start_server(server_config)
        if proc:
            processes.append((server_config, proc))
            time.sleep(2)  # Give each server time to start
        else:
            logger.warning(f"Failed to start {server_config['name']}")
    
    display_access_points()
    
    print("=" * 80)
    print("System Status:")
    print("=" * 80)
    print()
    
    for server_config, proc in processes:
        status = "✓ Running" if proc.poll() is None else "✗ Stopped"
        print(f"{status} | {server_config['name']:20} | PID: {proc.pid}")
    
    print()
    print("=" * 80)
    print("Running. Press CTRL+C to stop all services.")
    print("=" * 80)
    print()
    
    # Keep running
    try:
        while True:
            time.sleep(1)
            # Check if any process has died
            for server_config, proc in processes:
                if proc.poll() is not None:
                    logger.warning(f"{server_config['name']} died (PID: {proc.pid})")
                    # Optionally restart here
    except KeyboardInterrupt:
        print()
        print()
        print("=" * 80)
        print("Shutting down all services...")
        print("=" * 80)
        print()
        
        for server_config, proc in processes:
            logger.info(f"Stopping {server_config['name']}...")
            proc.terminate()
            try:
                proc.wait(timeout=5)
                logger.info(f"✓ {server_config['name']} stopped")
            except subprocess.TimeoutExpired:
                logger.warning(f"Force killing {server_config['name']}")
                proc.kill()
        
        print()
        print("=" * 80)
        print("All services stopped.")
        print("=" * 80)
        print()

if __name__ == "__main__":
    main()
