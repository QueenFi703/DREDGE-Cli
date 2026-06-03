"""
REPL Engine - Swift and Python REPL execution
"""

import asyncio
import subprocess
import logging
import sys
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4
import time
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class REPLSession:
    """REPL session state"""
    id: str
    language: str
    created_at: datetime = field(default_factory=datetime.now)
    commands: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    state: Dict[str, Any] = field(default_factory=dict)
    process: Optional[subprocess.Popen] = None
    temp_dir: Optional[Path] = None


class REPLEngine:
    """REPL execution engine for Swift and Python"""

    def __init__(self, config):
        self.config = config
        self.sessions: Dict[str, REPLSession] = {}
        self.swift_executable = "swift"
        self.python_executable = sys.executable

    def create_session(self, language: str = "swift") -> REPLSession:
        """Create a new REPL session"""
        session = REPLSession(
            id=str(uuid4()),
            language=language,
            temp_dir=Path(tempfile.mkdtemp(prefix=f"dredge_repl_{language}_"))
        )
        self.sessions[session.id] = session
        logger.info(f"Created {language} REPL session: {session.id}")
        return session

    def get_session(self, session_id: str) -> Optional[REPLSession]:
        """Get a REPL session"""
        return self.sessions.get(session_id)

    def get_or_create_session(
        self, 
        session_id: Optional[str] = None, 
        language: str = "swift"
    ) -> REPLSession:
        """Get or create a REPL session"""
        if session_id and session_id in self.sessions:
            return self.sessions[session_id]
        return self.create_session(language)

    def delete_session(self, session_id: str) -> bool:
        """Delete a REPL session"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            if session.process:
                session.process.terminate()
            if session.temp_dir and session.temp_dir.exists():
                import shutil
                shutil.rmtree(session.temp_dir)
            del self.sessions[session_id]
            logger.info(f"Deleted REPL session: {session_id}")
            return True
        return False

    async def execute(self, session: REPLSession, command: str) -> Dict[str, Any]:
        """Execute a REPL command"""
        start_time = time.time()

        try:
            if session.language == "swift":
                result = await self._execute_swift(session, command)
            elif session.language == "python":
                result = await self._execute_python(session, command)
            else:
                result = {
                    "output": "",
                    "error": f"Unknown language: {session.language}"
                }

            session.commands.append(command)
            session.outputs.append(result.get("output", ""))

            execution_time = time.time() - start_time
            result["execution_time"] = execution_time

            return result
        except Exception as e:
            logger.error(f"REPL execution error: {e}")
            return {
                "output": "",
                "error": str(e),
                "execution_time": time.time() - start_time
            }

    async def _execute_swift(self, session: REPLSession, command: str) -> Dict[str, Any]:
        """Execute Swift REPL command"""
        # Create Swift script file
        script_file = session.temp_dir / "repl_command.swift"

        # Prepare Swift code
        swift_code = f"""
import Foundation

{command}
"""
        script_file.write_text(swift_code)

        try:
            # Run Swift
            process = await asyncio.create_subprocess_exec(
                self.swift_executable,
                str(script_file),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=30.0
            )

            return {
                "output": stdout.decode("utf-8", errors="replace"),
                "error": stderr.decode("utf-8", errors="replace") if stderr else None
            }
        except asyncio.TimeoutError:
            return {
                "output": "",
                "error": "Command timed out (30s)"
            }
        except Exception as e:
            return {
                "output": "",
                "error": str(e)
            }

    async def _execute_python(self, session: REPLSession, command: str) -> Dict[str, Any]:
        """Execute Python REPL command"""
        try:
            # Run Python
            process = await asyncio.create_subprocess_exec(
                self.python_executable,
                "-c",
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(session.temp_dir)
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=30.0
            )

            return {
                "output": stdout.decode("utf-8", errors="replace"),
                "error": stderr.decode("utf-8", errors="replace") if stderr else None
            }
        except asyncio.TimeoutError:
            return {
                "output": "",
                "error": "Command timed out (30s)"
            }
        except Exception as e:
            return {
                "output": "",
                "error": str(e)
            }

    def execute_sync(self, session: REPLSession, command: str) -> Dict[str, Any]:
        """Execute REPL command synchronously"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.execute(session, command))
        finally:
            loop.close()

    def get_session_history(self, session_id: str) -> Dict[str, Any]:
        """Get session command history"""
        session = self.get_session(session_id)
        if not session:
            return {}

        return {
            "session_id": session_id,
            "language": session.language,
            "created_at": session.created_at.isoformat(),
            "commands": session.commands,
            "outputs": session.outputs,
            "command_count": len(session.commands)
        }

    def clear_session_history(self, session_id: str) -> bool:
        """Clear session history"""
        session = self.get_session(session_id)
        if session:
            session.commands.clear()
            session.outputs.clear()
            return True
        return False

    def export_session(self, session_id: str, format: str = "json") -> Optional[str]:
        """Export session to file"""
        session = self.get_session(session_id)
        if not session:
            return None

        import json

        data = {
            "session_id": session_id,
            "language": session.language,
            "created_at": session.created_at.isoformat(),
            "commands": session.commands,
            "outputs": session.outputs
        }

        if format == "json":
            return json.dumps(data, indent=2)
        elif format == "txt":
            lines = []
            for cmd, output in zip(session.commands, session.outputs):
                lines.append(f"> {cmd}")
                if output:
                    lines.append(output)
                lines.append("")
            return "\n".join(lines)

        return None
