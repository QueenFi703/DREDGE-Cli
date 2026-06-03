"""
Interactive DREDGE API - FastAPI backend for REPL, Config, Testing & Debugging
"""

import json
import logging
import asyncio
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime
from enum import Enum
import time

from fastapi import FastAPI, WebSocket, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field
import uvicorn

from .repl_engine import REPLEngine, REPLSession
from .config import DREDGEConfig
from .command_executor import CommandExecutor
from .web_ui_html import get_index_html
from .xcode_integration import XcodeIntegration

logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Models
# ============================================================================

class REPLCommand(BaseModel):
    """REPL command request"""
    command: str
    session_id: Optional[str] = None
    language: str = "swift"  # swift, python


class REPLResponse(BaseModel):
    """REPL command response"""
    session_id: str
    command: str
    output: str
    error: Optional[str] = None
    execution_time: float
    timestamp: datetime


class ConfigurationWizardStep(BaseModel):
    """Configuration wizard step"""
    step_id: int
    title: str
    description: str
    fields: List[Dict[str, Any]]
    validation_rules: Optional[Dict[str, Any]] = None


class ConfigUpdate(BaseModel):
    """Configuration update request"""
    model_config = ConfigDict(populate_by_name=True)

    category: str
    key: str
    value: Any
    should_validate: bool = Field(default=True, alias="validate")


class TestCase(BaseModel):
    """Test case definition"""
    name: str
    description: Optional[str] = None
    test_file: str
    language: str = "swift"
    tags: List[str] = []


class TestResult(BaseModel):
    """Test execution result"""
    test_name: str
    status: str  # passed, failed, skipped
    duration: float
    output: str
    error: Optional[str] = None
    timestamp: datetime


class DebugBreakpoint(BaseModel):
    """Debug breakpoint"""
    file: str
    line: int
    condition: Optional[str] = None
    enabled: bool = True


# ============================================================================
# FastAPI Application
# ============================================================================

class InteractiveDREDGEApp:
    """Main interactive DREDGE application"""

    def __init__(self, config_path: Optional[str] = None):
        self.app = FastAPI(
            title="Interactive DREDGE",
            description="Web UI for interactive DREDGE development",
            version="1.0.0"
        )

        # Initialize components
        self.config = DREDGEConfig(config_path)
        self.repl_engine = REPLEngine(self.config)
        self.executor = CommandExecutor()
        self.xcode = XcodeIntegration()
        self.sessions: Dict[str, REPLSession] = {}
        self.breakpoints: List[DebugBreakpoint] = []

        # Setup middleware
        self._setup_middleware()

        # Setup routes
        self._setup_routes()

        # Setup static files
        self._setup_static_files()

    def _setup_middleware(self):
        """Setup CORS and other middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_static_files(self):
        """Setup static file serving"""
        static_dir = Path(__file__).parent / "web_ui" / "dist"
        if static_dir.exists():
            self.app.mount("/static", StaticFiles(directory=static_dir), name="static")

    def _setup_routes(self):
        """Setup API routes"""

        # Health check
        @self.app.get("/health")
        async def health():
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "repl_sessions": len(self.sessions)
            }

        # =====================================================================
        # REPL Endpoints
        # =====================================================================

        @self.app.post("/api/repl/execute", response_model=REPLResponse)
        async def execute_repl_command(cmd: REPLCommand):
            """Execute a REPL command"""
            try:
                session = self.repl_engine.get_or_create_session(
                    session_id=cmd.session_id,
                    language=cmd.language
                )

                result = await self.repl_engine.execute(
                    session=session,
                    command=cmd.command
                )

                return REPLResponse(
                    session_id=session.id,
                    command=cmd.command,
                    output=result["output"],
                    error=result.get("error"),
                    execution_time=result["execution_time"],
                    timestamp=datetime.now()
                )
            except Exception as e:
                logger.error(f"REPL execution error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/repl/sessions")
        async def create_repl_session(language: str = "swift"):
            """Create a new REPL session"""
            session = self.repl_engine.create_session(language)
            return {
                "session_id": session.id,
                "language": language,
                "created_at": session.created_at.isoformat()
            }

        @self.app.get("/api/repl/sessions/{session_id}")
        async def get_repl_session(session_id: str):
            """Get REPL session info"""
            session = self.repl_engine.get_session(session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")

            return {
                "session_id": session.id,
                "language": session.language,
                "created_at": session.created_at.isoformat(),
                "command_count": len(session.commands),
                "last_command": session.commands[-1] if session.commands else None
            }

        @self.app.delete("/api/repl/sessions/{session_id}")
        async def delete_repl_session(session_id: str):
            """Delete a REPL session"""
            self.repl_engine.delete_session(session_id)
            return {"status": "deleted", "session_id": session_id}

        # =====================================================================
        # Configuration Wizard Endpoints
        # =====================================================================

        @self.app.get("/api/wizard/steps", response_model=List[ConfigurationWizardStep])
        async def get_wizard_steps():
            """Get configuration wizard steps"""
            return self._get_wizard_steps()

        @self.app.post("/api/config/update")
        async def update_configuration(update: ConfigUpdate):
            """Update configuration"""
            try:
                result = self.config.update(
                    category=update.category,
                    key=update.key,
                    value=update.value,
                    validate=update.should_validate
                )
                return {"status": "updated", "result": result}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

        @self.app.get("/api/config/current")
        async def get_current_config():
            """Get current configuration"""
            return self.config.to_dict()

        @self.app.get("/api/config/schema")
        async def get_config_schema():
            """Get configuration schema"""
            return self.config.get_schema()

        # =====================================================================
        # Testing Endpoints
        # =====================================================================

        @self.app.post("/api/tests/run")
        async def run_tests(test: TestCase):
            """Run a test"""
            try:
                result = await self.executor.run_test(test)
                return TestResult(
                    test_name=test.name,
                    status=result["status"],
                    duration=result["duration"],
                    output=result["output"],
                    error=result.get("error"),
                    timestamp=datetime.now()
                )
            except Exception as e:
                logger.error(f"Test execution error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/tests/discover")
        async def discover_tests(directory: str = "swift/Tests"):
            """Discover available tests"""
            tests = await self.executor.discover_tests(directory)
            return {"tests": tests}

        @self.app.post("/api/tests/run-all")
        async def run_all_tests(directory: str = "swift/Tests"):
            """Run all tests in directory"""
            try:
                results = await self.executor.run_all_tests(directory)
                return {"results": results}
            except Exception as e:
                logger.error(f"Test suite error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # =====================================================================
        # Debugging Endpoints
        # =====================================================================

        @self.app.post("/api/debug/breakpoint")
        async def set_breakpoint(breakpoint: DebugBreakpoint):
            """Set a debug breakpoint"""
            self.breakpoints.append(breakpoint)
            return {"status": "set", "breakpoint": breakpoint}

        @self.app.delete("/api/debug/breakpoint/{file}/{line}")
        async def delete_breakpoint(file: str, line: int):
            """Delete a debug breakpoint"""
            self.breakpoints = [
                bp for bp in self.breakpoints
                if not (bp.file == file and bp.line == line)
            ]
            return {"status": "deleted"}

        @self.app.get("/api/debug/breakpoints")
        async def list_breakpoints():
            """List all breakpoints"""
            return {"breakpoints": self.breakpoints}

        # =====================================================================
        # WebSocket for Real-time Streaming
        # =====================================================================

        @self.app.websocket("/ws/repl/{session_id}")
        async def websocket_repl(websocket: WebSocket, session_id: str):
            """WebSocket endpoint for REPL streaming"""
            await websocket.accept()
            session = self.repl_engine.get_session(session_id)

            if not session:
                await websocket.close(code=4004, reason="Session not found")
                return

            try:
                while True:
                    data = await websocket.receive_text()
                    cmd_data = json.loads(data)

                    result = await self.repl_engine.execute(
                        session=session,
                        command=cmd_data["command"]
                    )

                    await websocket.send_json({
                        "command": cmd_data["command"],
                        "output": result["output"],
                        "error": result.get("error"),
                        "execution_time": result["execution_time"]
                    })
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                await websocket.close(code=1011, reason=str(e))

        @self.app.websocket("/ws/debug/{session_id}")
        async def websocket_debug(websocket: WebSocket, session_id: str):
            """WebSocket endpoint for debug streaming"""
            await websocket.accept()

            try:
                while True:
                    data = await websocket.receive_text()
                    debug_cmd = json.loads(data)

                    # Send debug info
                    await websocket.send_json({
                        "type": debug_cmd["type"],
                        "data": await self._handle_debug_command(debug_cmd)
                    })
            except Exception as e:
                logger.error(f"Debug WebSocket error: {e}")
                await websocket.close(code=1011, reason=str(e))

        # =====================================================================
        # Shell Execution Endpoints
        # =====================================================================

        @self.app.post("/api/shell/execute")
        async def execute_shell_command(command: Dict[str, str]):
            """Execute a shell command"""
            try:
                result = await self.executor.execute_shell(command["cmd"])
                return {
                    "command": command["cmd"],
                    "stdout": result["stdout"],
                    "stderr": result["stderr"],
                    "return_code": result["return_code"]
                }
            except Exception as e:
                logger.error(f"Shell execution error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # =====================================================================
        # Build & Compilation Endpoints
        # =====================================================================

        @self.app.post("/api/build/swift")
        async def build_swift(target: str = "debug"):
            """Build Swift package"""
            try:
                result = await self.executor.build_swift(target)
                return {
                    "status": result["status"],
                    "output": result["output"],
                    "errors": result.get("errors", [])
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/swift/dependencies/resolve")
        async def resolve_swift_dependencies():
            """Resolve Swift dependencies for root and nested packages"""
            try:
                return await self.executor.resolve_swift_dependencies()
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/swift/dependencies/build")
        async def build_swift_dependency():
            """Build the local DREDGE Swift dependency package"""
            try:
                return await self.executor.build_swift_dependency()
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/swift/dependencies/describe")
        async def describe_swift_dependencies():
            """Describe the Swift dependency graph"""
            try:
                return await self.executor.describe_swift_dependencies()
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/build/python")
        async def build_python():
            """Build Python package"""
            try:
                result = await self.executor.build_python()
                return {
                    "status": result["status"],
                    "output": result["output"],
                    "errors": result.get("errors", [])
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        # =====================================================================
        # Xcode Integration Endpoints
        # =====================================================================

        @self.app.get("/api/xcode/project-info")
        async def get_xcode_project_info(project_path: str = "."):
            """Get Xcode project information"""
            try:
                info = await self.xcode.get_project_info(project_path)
                return info
            except Exception as e:
                logger.error(f"Failed to get Xcode info: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/xcode/schemes")
        async def get_xcode_schemes(project_path: str = "."):
            """Get available Xcode schemes"""
            try:
                schemes = await self.xcode.get_schemes(project_path)
                return {"schemes": schemes}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/xcode/targets")
        async def get_xcode_targets(project_path: str = "."):
            """Get available build targets"""
            try:
                targets = await self.xcode.get_targets(project_path)
                return {"targets": targets}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/xcode/build")
        async def build_xcode_scheme(
            scheme: str,
            configuration: str = "Debug",
            project_path: str = "."
        ):
            """Build an Xcode scheme"""
            try:
                result = await self.xcode.build_scheme(
                    scheme=scheme,
                    configuration=configuration,
                    project_path=project_path
                )
                return result
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/xcode/test")
        async def test_xcode_scheme(
            scheme: str,
            configuration: str = "Debug",
            project_path: str = "."
        ):
            """Test an Xcode scheme"""
            try:
                result = await self.xcode.test_scheme(
                    scheme=scheme,
                    configuration=configuration,
                    project_path=project_path
                )
                return result
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/xcode/analyze")
        async def analyze_xcode_scheme(
            scheme: str,
            project_path: str = "."
        ):
            """Analyze an Xcode scheme"""
            try:
                result = await self.xcode.analyze_scheme(
                    scheme=scheme,
                    project_path=project_path
                )
                return result
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/xcode/archive")
        async def archive_xcode_scheme(
            scheme: str,
            archive_path: str,
            project_path: str = "."
        ):
            """Archive an Xcode scheme"""
            try:
                result = await self.xcode.archive_scheme(
                    scheme=scheme,
                    archive_path=archive_path,
                    project_path=project_path
                )
                return result
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/xcode/export-archive")
        async def export_archive(
            archive_path: str,
            export_path: str,
            export_options: Optional[Dict[str, Any]] = None
        ):
            """Export an Xcode archive"""
            try:
                result = await self.xcode.export_archive(
                    archive_path=archive_path,
                    export_path=export_path,
                    export_options=export_options
                )
                return result
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/xcode/derived-data")
        async def get_derived_data_path():
            """Get Xcode derived data path"""
            path = self.xcode.get_derived_data_path()
            return {"path": path if path else "Not found"}

        # =====================================================================
        # File Management Endpoints
        # =====================================================================

        @self.app.post("/api/files/upload")
        async def upload_file(file: UploadFile = File(...)):
            """Upload a file"""
            try:
                contents = await file.read()
                path = Path("uploads") / file.filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(contents)
                return {
                    "filename": file.filename,
                    "size": len(contents),
                    "path": str(path)
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        # =====================================================================
        # UI Endpoints
        # =====================================================================

        @self.app.get("/", response_class=HTMLResponse)
        async def root():
            """Serve main UI"""
            return self._get_index_html()

    def _get_wizard_steps(self) -> List[ConfigurationWizardStep]:
        """Get configuration wizard steps"""
        return [
            ConfigurationWizardStep(
                step_id=1,
                title="Project Setup",
                description="Configure basic project settings",
                fields=[
                    {"name": "project_name", "type": "text", "required": True},
                    {"name": "description", "type": "textarea"},
                    {"name": "version", "type": "text", "value": "0.1.0"},
                ]
            ),
            ConfigurationWizardStep(
                step_id=2,
                title="Swift Configuration",
                description="Configure Swift compilation settings",
                fields=[
                    {"name": "swift_version", "type": "select", "options": ["5.8", "5.9", "5.10"]},
                    {"name": "optimization_level", "type": "select", "options": ["-O", "-Osize", "-Onone"]},
                    {"name": "enable_testing", "type": "checkbox", "value": True},
                ]
            ),
            ConfigurationWizardStep(
                step_id=3,
                title="Python Configuration",
                description="Configure Python settings",
                fields=[
                    {"name": "python_version", "type": "select", "options": ["3.8", "3.9", "3.10", "3.11", "3.12"]},
                    {"name": "virtual_env", "type": "checkbox", "value": True},
                ]
            ),
            ConfigurationWizardStep(
                step_id=4,
                title="Docker Configuration",
                description="Configure Docker settings",
                fields=[
                    {"name": "enable_docker", "type": "checkbox", "value": True},
                    {"name": "docker_image", "type": "text", "value": "dredge-dev:latest"},
                    {"name": "port", "type": "number", "value": 8000},
                ]
            ),
            ConfigurationWizardStep(
                step_id=5,
                title="Testing Configuration",
                description="Configure test settings",
                fields=[
                    {"name": "test_framework", "type": "select", "options": ["XCTest", "pytest"]},
                    {"name": "coverage_threshold", "type": "number", "value": 80},
                    {"name": "auto_run_tests", "type": "checkbox", "value": False},
                ]
            ),
        ]

    async def _handle_debug_command(self, cmd: Dict[str, Any]) -> Dict[str, Any]:
        """Handle debug command"""
        cmd_type = cmd.get("type")

        if cmd_type == "get_stack":
            return {"stack": "mock stack trace"}
        elif cmd_type == "get_variables":
            return {"variables": {}}
        elif cmd_type == "continue":
            return {"status": "continued"}
        else:
            return {"error": f"Unknown command: {cmd_type}"}

    def _get_index_html(self) -> str:
        """Get index HTML"""
        return get_index_html()

    def run(self, host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
        """Run the application"""
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Interactive DREDGE API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--config", help="Path to config file")

    args = parser.parse_args()

    app = InteractiveDREDGEApp(config_path=args.config)
    app.run(host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
