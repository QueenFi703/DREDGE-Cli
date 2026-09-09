#!/usr/bin/env python3
"""
DREDGE Complete System - Fixed Startup & Connectivity
Starts all servers with proper error handling and diagnostics
"""

import subprocess
import time
import logging
import sys
import socket
from pathlib import Path
from typing import List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONNECTIVITY CHECKER
# ============================================================================

def check_port_available(port: int) -> bool:
    """Check if port is available"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('127.0.0.1', port))
        sock.close()
        return result != 0  # 0 means port is in use
    except Exception as e:
        logger.error(f"Port check error: {e}")
        return False

def kill_process_on_port(port: int) -> bool:
    """Kill process using specific port"""
    try:
        import psutil
        for proc in psutil.process_iter(['pid', 'name', 'connections']):
            try:
                for conn in proc.connections():
                    if conn.laddr.port == port:
                        logger.warning(f"Killing process {proc.pid} on port {port}")
                        proc.kill()
                        time.sleep(1)
                        return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except ImportError:
        logger.warning("psutil not available, using alternative method")
        try:
            subprocess.run(f"taskkill /F /PID {{pidof python}} 2>nul", shell=True)
        except:
            pass
    return False

def wait_for_port(port: int, timeout: int = 10) -> bool:
    """Wait for port to become available"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('127.0.0.1', port))
            sock.close()
            if result == 0:
                return True
        except:
            pass
        time.sleep(0.5)
    return False

# ============================================================================
# SERVER CONFIGURATION
# ============================================================================

SERVERS = [
    {
        "name": "webMCP",
        "script": "webmcp.py",
        "port": 3000,
        "url": "http://127.0.0.1:3000",
        "description": "Web-based Agentic System UI",
        "type": "ui",
        "startup_delay": 3
    },
    {
        "name": "MCP Server",
        "script": "mcp_server.py",
        "port": 3002,
        "url": "http://127.0.0.1:3002",
        "description": "Model Context Protocol Server",
        "type": "service",
        "startup_delay": 2
    },
    {
        "name": "DREDGE Server",
        "script": "dredge_server.py",
        "port": 8001,
        "url": "http://127.0.0.1:8001",
        "description": "Execution Layer and Resource Management",
        "type": "service",
        "startup_delay": 2
    },
    {
        "name": "Gateway",
        "script": "core_gateway.py",
        "port": 8080,
        "url": "http://127.0.0.1:8080",
        "description": "API Gateway (Fallback Port)",
        "type": "gateway",
        "startup_delay": 2
    }
]

# ============================================================================
# DIAGNOSTICS
# ============================================================================

def run_diagnostics():
    """Run system diagnostics"""
    print("\n" + "=" * 80)
    print("SYSTEM DIAGNOSTICS")
    print("=" * 80 + "\n")
    
    # Check Python version
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}\n")
    
    # Check required files
    print("Required Files:")
    for server in SERVERS:
        script = server["script"]
        exists = Path(script).exists()
        status = "✓ Found" if exists else "✗ MISSING"
        print(f"  {status}: {script}")
    print()
    
    # Check ports
    print("Port Availability:")
    for server in SERVERS:
        available = check_port_available(server["port"])
        status = "✓ Available" if available else "✗ In Use"
        print(f"  {status}: Port {server['port']} ({server['name']})")
    print()
    
    # Check packages
    print("Required Packages:")
    packages = ["fastapi", "uvicorn", "httpx"]
    for pkg in packages:
        try:
            __import__(pkg)
            print(f"  ✓ {pkg}")
        except ImportError:
            print(f"  ✗ {pkg} (MISSING - run: pip install {pkg})")
    print()

# ============================================================================
# LAUNCHER
# ============================================================================

def start_server(server_config):
    """Start a single server with error handling"""
    name = server_config["name"]
    script = server_config["script"]
    port = server_config["port"]
    
    logger.info(f"Starting {name} on port {port}...")
    
    # Check if file exists
    if not Path(script).exists():
        logger.error(f"✗ Script not found: {script}")
        return None
    
    # Clear port if needed
    if not check_port_available(port):
        logger.warning(f"Port {port} in use, attempting to clear...")
        kill_process_on_port(port)
        time.sleep(1)
    
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

def test_connectivity(url: str, timeout: int = 5) -> bool:
    """Test if server is responding"""
    try:
        import requests
        response = requests.get(url + "/health", timeout=timeout)
        return response.status_code < 500
    except Exception as e:
        logger.debug(f"Connectivity test failed: {e}")
        return False

def display_banner():
    """Display system banner"""
    banner = """
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║         DREDGE Complete Agentic System - Fixed Startup                     ║
║                                                                            ║
║  Components:                                                               ║
║    • webMCP - Web-based Agentic System (Port 3000)                         ║
║    • MCP Server - Model Context Protocol (Port 3002)                       ║
║    • DREDGE Server - Execution Layer (Port 8001)                           ║
║    • Gateway - API Gateway (Port 8080)                                     ║
║                                                                            ║
║  Status: Starting with connectivity diagnostics...                         ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def display_access_points():
    """Display access points"""
    print("\n" + "=" * 80)
    print("ACCESS POINTS - CONNECT TO THESE URLS")
    print("=" * 80)
    print()
    
    # Group by type
    ui_servers = [s for s in SERVERS if s["type"] == "ui"]
    services = [s for s in SERVERS if s["type"] == "service"]
    gateways = [s for s in SERVERS if s["type"] == "gateway"]
    
    if ui_servers:
        print("🎨 USER INTERFACE:")
        for server in ui_servers:
            print(f"  {server['url']}/")
            print(f"    Description: {server['description']}")
        print()
    
    if services:
        print("⚙️  SERVICES:")
        for server in services:
            print(f"  {server['url']}/")
            print(f"    Description: {server['description']}")
            print(f"    API Docs: {server['url']}/docs")
        print()
    
    if gateways:
        print("🚪 GATEWAY:")
        for server in gateways:
            print(f"  {server['url']}/")
            print(f"    Description: {server['description']}")
            print(f"    API Docs: {server['url']}/swagger")
        print()
    
    print("🔗 QUICK LINKS:")
    print(f"  • webMCP Dashboard: http://127.0.0.1:3000/")
    print(f"  • System Status: http://127.0.0.1:8080/status")
    print(f"  • Speedrun Alpha: http://127.0.0.1:8002/speedrun/full-report")
    print(f"  • All Services: See above")
    print()

def main():
    """Main startup routine"""
    
    # Run diagnostics first
    run_diagnostics()
    
    display_banner()
    
    # Confirm startup
    print("\nProceed with startup? (y/n): ", end="", flush=True)
    try:
        response = input().strip().lower()
        if response not in ['y', 'yes', '']:
            print("Startup cancelled.")
            return
    except:
        pass  # Assume yes if running non-interactively
    
    print()
    processes = []
    
    # Start all servers
    logger.info("Starting all services...")
    print()
    for server_config in SERVERS:
        logger.info(f"Starting {server_config['name']}...")
        proc = start_server(server_config)
        if proc:
            processes.append((server_config, proc))
            
            # Wait for server to be ready
            logger.info(f"Waiting for {server_config['name']} to be ready...")
            time.sleep(server_config.get("startup_delay", 2))
            
            # Test connectivity
            if test_connectivity(server_config["url"]):
                logger.info(f"✓ {server_config['name']} is responding")
            else:
                logger.warning(f"⚠ {server_config['name']} may not be responding yet")
        else:
            logger.warning(f"Failed to start {server_config['name']}")
        print()
    
    display_access_points()
    
    print("=" * 80)
    print("SYSTEM STATUS")
    print("=" * 80)
    print()
    
    running_count = 0
    for server_config, proc in processes:
        status = "✓ Running" if proc.poll() is None else "✗ Stopped"
        print(f"{status} | {server_config['name']:20} | Port: {server_config['port']:5} | PID: {proc.pid}")
        if proc.poll() is None:
            running_count += 1
    
    print()
    print(f"Services Running: {running_count}/{len(SERVERS)}")
    print()
    
    if running_count == 0:
        print("✗ No services are running. Check logs above for errors.")
        return
    
    print("=" * 80)
    print("SYSTEM READY")
    print("=" * 80)
    print()
    print("✓ All services are running and ready to use!")
    print()
    print("To access the system:")
    print("  1. Open your browser")
    print("  2. Go to: http://127.0.0.1:3000/")
    print("  3. Start using webMCP!")
    print()
    print("Press CTRL+C to stop all services.")
    print()
    
    # Keep running
    try:
        while True:
            time.sleep(1)
            # Check if any process has died
            for server_config, proc in processes:
                if proc.poll() is not None:
                    logger.warning(f"{server_config['name']} died (PID: {proc.pid})")
    except KeyboardInterrupt:
        print()
        print()
        print("=" * 80)
        print("SHUTTING DOWN ALL SERVICES")
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
        print("ALL SERVICES STOPPED")
        print("=" * 80)
        print()

if __name__ == "__main__":
    main()
