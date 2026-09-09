#!/usr/bin/env python3
"""
DREDGE System - Simple Working Startup (Windows Compatible)
Starts all servers sequentially and keeps them running
"""

import subprocess
import time
import sys
import os
from pathlib import Path

# Fix Unicode on Windows
if sys.platform == "win32":
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 80)
print("DREDGE COMPLETE SYSTEM - STARTUP")
print("=" * 80)
print()

# Servers to start
servers = [
    ("webMCP (UI)", "webmcp.py", 3000, 3),
    ("MCP Server", "mcp_server.py", 3002, 2),
    ("DREDGE Server", "dredge_server.py", 8001, 2),
    ("Gateway", "core_gateway.py", 8080, 2),
]

processes = []

print("Starting servers...")
print()

for name, script, port, delay in servers:
    print(f"[*] Starting {name} (port {port})...")
    
    # Check if script exists
    if not Path(script).exists():
        print(f"    ERROR: {script} not found!")
        continue
    
    try:
        # Start the process
        proc = subprocess.Popen(
            [sys.executable, script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        processes.append((name, proc, port))
        print(f"    [OK] Started (PID: {proc.pid})")
        
        # Wait for server to initialize
        time.sleep(delay)
        
    except Exception as e:
        print(f"    ERROR: {e}")

print()
print("=" * 80)
print("SERVERS STARTED")
print("=" * 80)
print()

# Display access points (no Unicode)
print("ACCESS YOUR SERVERS:")
print()
print("  [UI] webMCP Dashboard:")
print("     http://127.0.0.1:3000/")
print()
print("  [API] MCP Server Docs:")
print("     http://127.0.0.1:3002/docs")
print()
print("  [API] DREDGE Server Docs:")
print("     http://127.0.0.1:8001/docs")
print()
print("  [API] Gateway Swagger:")
print("     http://127.0.0.1:8080/swagger")
print()
print("  [ADMIN] Speedrun Alpha:")
print("     http://127.0.0.1:8002/speedrun/full-report")
print()
print("=" * 80)
print()

# Monitor processes
print("Status: All servers running")
print("Press CTRL+C to stop all servers")
print()

running = len(processes)
print(f"Running: {running}/{len(servers)} servers")
for name, proc, port in processes:
    status = "[OK]" if proc.poll() is None else "[X]"
    print(f"  {status} {name} (port {port}, PID {proc.pid})")

print()

try:
    # Keep running and monitor
    while True:
        time.sleep(5)
        
        # Check if any process died
        still_running = 0
        for name, proc, port in processes:
            if proc.poll() is None:
                still_running += 1
            else:
                print(f"[!] WARNING: {name} (PID {proc.pid}) stopped!")
        
        if still_running != running:
            running = still_running
            print(f"Running: {running}/{len(servers)} servers")
            
            if running == 0:
                print("[!] ERROR: All servers have stopped!")
                break

except KeyboardInterrupt:
    print()
    print()
    print("=" * 80)
    print("SHUTTING DOWN")
    print("=" * 80)
    print()
    
    for name, proc, port in processes:
        if proc.poll() is None:
            print(f"Stopping {name}...")
            proc.terminate()
            try:
                proc.wait(timeout=3)
                print(f"  [OK] Stopped")
            except subprocess.TimeoutExpired:
                print(f"  Force killing...")
                proc.kill()
    
    print()
    print("All servers stopped.")
    print()
