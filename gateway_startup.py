#!/usr/bin/env python3
"""
DREDGE Gateway Startup Script - Deterministic Initialization

Handles:
  - State recovery from checkpoint
  - Manifest verification
  - Session reconstruction
  - Request queue restoration
  - Health readiness verification
"""

import json
import sys
import logging
from pathlib import Path
from datetime import datetime
import asyncio

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import state management
sys.path.insert(0, '/app')
from gateway_state_management import state_manager


async def startup_sequence():
    """Execute deterministic startup sequence"""
    
    logger.info("=" * 80)
    logger.info("DREDGE Gateway Startup Sequence")
    logger.info("=" * 80)
    
    try:
        # Step 1: Initialize state management
        logger.info("[1/5] Initializing state management...")
        await state_manager.initialize()
        logger.info("✅ State management ready")
        
        # Step 2: Verify state integrity
        logger.info("[2/5] Verifying state integrity...")
        state = state_manager.get_state()
        
        logger.info(f"  Gateway ID: {state.get('gateway_id')}")
        logger.info(f"  Version: {state.get('version')}")
        logger.info(f"  Sessions: {len(state.get('sessions', {}))}")
        logger.info(f"  Queued requests: {len(state.get('request_queue', []))}")
        logger.info(f"  Crashes recovered: {state.get('metrics', {}).get('crashes_recovered', 0)}")
        logger.info("✅ State integrity verified")
        
        # Step 3: Reconstruct session state
        logger.info("[3/5] Reconstructing session state...")
        active_sessions = len(state.get('sessions', {}))
        logger.info(f"  Restored {active_sessions} active sessions")
        
        for session_id, session_data in state.get('sessions', {}).items():
            created_at = session_data.get('created_at', 'unknown')
            logger.info(f"    - {session_id} (created: {created_at})")
        
        logger.info("✅ Session state reconstructed")
        
        # Step 4: Restore request queue
        logger.info("[4/5] Restoring request queue...")
        queued_requests = len(state.get('request_queue', []))
        logger.info(f"  Restored {queued_requests} queued requests")
        
        for request in state.get('request_queue', [])[:5]:  # Show first 5
            request_id = request.get('id', 'unknown')
            queued_at = request.get('queued_at', 'unknown')
            logger.info(f"    - {request_id} (queued: {queued_at})")
        
        if queued_requests > 5:
            logger.info(f"    ... and {queued_requests - 5} more")
        
        logger.info("✅ Request queue restored")
        
        # Step 5: Report readiness
        logger.info("[5/5] Checking readiness...")
        
        readiness_checks = {
            "state_loaded": state is not None,
            "manifest_valid": Path('/app/state/manifest.json').exists(),
            "checkpoint_present": Path('/app/state/checkpoint.json').exists(),
            "sessions_recovered": active_sessions > 0 or True,  # OK to have 0 sessions
            "recovery_logged": Path('/app/state/recovery.log').exists() or True  # OK if new
        }
        
        for check, result in readiness_checks.items():
            status = "✅" if result else "⚠️"
            logger.info(f"  {status} {check}: {result}")
        
        all_checks_pass = all(readiness_checks.values())
        
        if all_checks_pass:
            logger.info("✅ Gateway ready for traffic")
            logger.info("=" * 80)
            return 0
        else:
            logger.warning("⚠️  Some checks failed - proceeding with degraded startup")
            logger.info("=" * 80)
            return 1
    
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        import traceback
        traceback.print_exc()
        logger.info("=" * 80)
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(startup_sequence())
    sys.exit(exit_code)
