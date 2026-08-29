#!/usr/bin/env python3
"""
DREDGE Multi-Server Launcher
Starts all three servers on correct ports with proper configuration
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
# SERVER CONFIGURATION
# ============================================================================

SERVERS = [
    {
        "name": "MCP Server",
        "script": "mcp_server.py",
        "port": 3002,
        "url": "http://127.0.0.1:3002",
        "description": "Model Context Protocol Server"
    },
    {
        "name": "DREDGE Server",
        "script": "dredge_server.py",
        "port": 8001,
        "url": "http://127.0.0.1:8001",
        "description": "Execution Layer and Resource Management"
    },
    {
        "name": "Gateway",
        "script": "core_gateway.py",
        "port": 8080,
        "url": "http://127.0.0.1:8080",
        "description": "API Gateway (Fallback Port)"
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

def main():
    """Launch all servers"""
    logger.info("=" * 80)
    logger.info("DREDGE MULTI-SERVER LAUNCHER")
    logger.info("=" * 80)
    logger.info("")
    
    processes = []
    
    # Start all servers
    for server_config in SERVERS:
        logger.info(f"Starting {server_config['name']}...")
        proc = start_server(server_config)
        if proc:
            processes.append((server_config, proc))
            time.sleep(1)  # Give each server time to start
        else:
            logger.warning(f"Failed to start {server_config['name']}")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("ALL SERVERS STARTED")
    logger.info("=" * 80)
    logger.info("")
    
    # Display server info
    for server_config in SERVERS:
        logger.info(f"{server_config['name']}:")
        logger.info(f"  Port: {server_config['port']}")
        logger.info(f"  URL: {server_config['url']}")
        logger.info(f"  Docs: {server_config['url']}/docs")
        logger.info("")
    
    # Display access points
    logger.info("Access Points:")
    logger.info("  - Gateway: http://127.0.0.1:8080/")
    logger.info("  - MCP Info: http://127.0.0.1:8080/mcp")
    logger.info("  - Dashboard: http://127.0.0.1:8080/dashboard")
    logger.info("  - MCP Tools: http://127.0.0.1:3002/tools")
    logger.info("  - MCP Docs: http://127.0.0.1:3002/docs")
    logger.info("  - DREDGE Docs: http://127.0.0.1:8001/docs")
    logger.info("")
    logger.info("=" * 80)
    logger.info("Running. Press CTRL+C to stop.")
    logger.info("=" * 80)
    logger.info("")
    
    # Keep running
    try:
        while True:
            time.sleep(1)
            # Check if any process has died
            for server_config, proc in processes:
                if proc.poll() is not None:
                    logger.warning(f"{server_config['name']} died (PID: {proc.pid})")
    except KeyboardInterrupt:
        logger.info("")
        logger.info("Stopping all servers...")
        
        for server_config, proc in processes:
            logger.info(f"Stopping {server_config['name']}...")
            proc.terminate()
            try:
                proc.wait(timeout=5)
                logger.info(f"✓ {server_config['name']} stopped")
            except subprocess.TimeoutExpired:
                logger.warning(f"Force killing {server_config['name']}")
                proc.kill()
        
        logger.info("")
        logger.info("All servers stopped.")

if __name__ == "__main__":
    main()
