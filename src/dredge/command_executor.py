"""
Command Executor - Handles Swift/Python build, test, and execution
"""

import asyncio
import subprocess
import logging
import json
import sys
from types import SimpleNamespace
from typing import Dict, Any, List, Optional
from pathlib import Path
import shutil

logger = logging.getLogger(__name__)


class CommandExecutor:
    """Executes build, test, and shell commands"""

    def __init__(self):
        self.swift_path = shutil.which("swift") or "swift"
        self.python_path = sys.executable
        self.xcodebuild_path = shutil.which("xcodebuild") or "xcodebuild"
        self.project_root = Path(__file__).resolve().parents[2]

    async def _run_process(
        self,
        args: List[str],
        cwd: Optional[Path] = None,
        timeout: float = 300.0,
    ) -> Dict[str, Any]:
        """Run a process and return normalized command output."""
        try:
            process = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(cwd or self.project_root),
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )

            return {
                "status": "success" if process.returncode == 0 else "failed",
                "command": " ".join(args),
                "cwd": str(cwd or self.project_root),
                "output": stdout.decode("utf-8", errors="replace"),
                "errors": stderr.decode("utf-8", errors="replace").splitlines(),
                "return_code": process.returncode,
            }
        except asyncio.TimeoutError:
            return {
                "status": "failed",
                "command": " ".join(args),
                "cwd": str(cwd or self.project_root),
                "output": "",
                "errors": [f"Command timed out after {timeout:.0f}s"],
                "return_code": -1,
            }
        except Exception as e:
            logger.error(f"Process execution error: {e}")
            return {
                "status": "failed",
                "command": " ".join(args),
                "cwd": str(cwd or self.project_root),
                "output": "",
                "errors": [str(e)],
                "return_code": -1,
            }

    async def execute_shell(self, command: str) -> Dict[str, Any]:
        """Execute a shell command"""
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=60.0
            )

            return {
                "stdout": stdout.decode("utf-8", errors="replace"),
                "stderr": stderr.decode("utf-8", errors="replace"),
                "return_code": process.returncode
            }
        except asyncio.TimeoutError:
            return {
                "stdout": "",
                "stderr": "Command timed out",
                "return_code": -1
            }
        except Exception as e:
            logger.error(f"Shell execution error: {e}")
            return {
                "stdout": "",
                "stderr": str(e),
                "return_code": -1
            }

    async def build_swift(self, target: str = "debug") -> Dict[str, Any]:
        """Build Swift package"""
        return await self._run_process(
            [self.swift_path, "build", "-c", target],
            cwd=self.project_root / "swift",
            timeout=300.0,
        )

    async def resolve_swift_dependencies(self) -> Dict[str, Any]:
        """Resolve Swift dependencies for both root and nested Swift packages."""
        root_result = await self._run_process(
            [self.swift_path, "package", "resolve"],
            cwd=self.project_root,
            timeout=120.0,
        )
        swift_result = await self._run_process(
            [self.swift_path, "package", "resolve"],
            cwd=self.project_root / "swift",
            timeout=120.0,
        )

        status = (
            "success"
            if root_result["status"] == "success" and swift_result["status"] == "success"
            else "failed"
        )
        return {
            "status": status,
            "steps": [
                {"name": "Root Package.swift", **root_result},
                {"name": "swift/Package.swift", **swift_result},
            ],
            "output": "\n".join(
                part
                for part in [root_result.get("output", ""), swift_result.get("output", "")]
                if part
            ),
            "errors": root_result.get("errors", []) + swift_result.get("errors", []),
        }

    async def build_swift_dependency(self) -> Dict[str, Any]:
        """Build the local DREDGE Swift dependency package."""
        return await self._run_process(
            [self.swift_path, "build"],
            cwd=self.project_root / "swift" / "DREDGE",
            timeout=180.0,
        )

    async def describe_swift_dependencies(self) -> Dict[str, Any]:
        """Describe the resolved Swift package graph."""
        return await self._run_process(
            [self.swift_path, "package", "describe"],
            cwd=self.project_root / "swift",
            timeout=60.0,
        )

    async def build_python(self) -> Dict[str, Any]:
        """Build Python package"""
        try:
            process = await asyncio.create_subprocess_exec(
                self.python_path,
                "setup.py",
                "build",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.project_root)
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=120.0
            )

            return {
                "status": "success" if process.returncode == 0 else "failed",
                "output": stdout.decode("utf-8", errors="replace")
            }
        except asyncio.TimeoutError:
            return {
                "status": "failed",
                "output": "Build timed out"
            }
        except Exception as e:
            logger.error(f"Python build error: {e}")
            return {
                "status": "failed",
                "output": str(e)
            }

    async def discover_tests(self, directory: str = "swift/Tests") -> List[Dict[str, Any]]:
        """Discover test files"""
        tests = []
        test_dir = Path(directory)
        if not test_dir.is_absolute():
            test_dir = self.project_root / test_dir

        if not test_dir.exists():
            return tests

        # Find test files
        for test_file in test_dir.rglob("*Tests.swift"):
            tests.append({
                "name": test_file.stem,
                "path": str(test_file),
                "language": "swift"
            })

        for test_file in test_dir.rglob("test_*.py"):
            tests.append({
                "name": test_file.stem,
                "path": str(test_file),
                "language": "python"
            })

        return tests

    async def run_test(self, test) -> Dict[str, Any]:
        """Run a single test"""
        try:
            if test.language == "swift":
                return await self._run_swift_test(test)
            elif test.language == "python":
                return await self._run_python_test(test)
            else:
                return {
                    "status": "failed",
                    "duration": 0,
                    "output": "",
                    "error": f"Unknown language: {test.language}"
                }
        except Exception as e:
            logger.error(f"Test execution error: {e}")
            return {
                "status": "failed",
                "duration": 0,
                "output": "",
                "error": str(e)
            }

    async def _run_swift_test(self, test) -> Dict[str, Any]:
        """Run Swift test"""
        import time
        start = time.time()

        try:
            process = await asyncio.create_subprocess_exec(
                self.swift_path,
                "test",
                "--filter", test.name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.project_root / "swift")
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=120.0
            )

            duration = time.time() - start
            output = stdout.decode("utf-8", errors="replace")

            return {
                "status": "passed" if process.returncode == 0 else "failed",
                "duration": duration,
                "output": output,
                "error": stderr.decode("utf-8", errors="replace") if stderr else None
            }
        except asyncio.TimeoutError:
            return {
                "status": "failed",
                "duration": time.time() - start,
                "output": "",
                "error": "Test timed out"
            }

    async def _run_python_test(self, test) -> Dict[str, Any]:
        """Run Python test"""
        import time
        start = time.time()

        try:
            process = await asyncio.create_subprocess_exec(
                self.python_path,
                "-m", "pytest",
                str(test.test_file),
                "-v",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.project_root)
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=120.0
            )

            duration = time.time() - start
            output = stdout.decode("utf-8", errors="replace")

            return {
                "status": "passed" if process.returncode == 0 else "failed",
                "duration": duration,
                "output": output,
                "error": stderr.decode("utf-8", errors="replace") if stderr else None
            }
        except asyncio.TimeoutError:
            return {
                "status": "failed",
                "duration": time.time() - start,
                "output": "",
                "error": "Test timed out"
            }

    async def run_all_tests(self, directory: str = "swift/Tests") -> List[Dict[str, Any]]:
        """Run all tests"""
        tests = await self.discover_tests(directory)
        results = []

        for test in tests:
            result = await self.run_test(SimpleNamespace(**test, test_file=test["path"]))
            result["test_name"] = test["name"]
            results.append(result)

        return results
