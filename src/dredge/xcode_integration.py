"""
Xcode Integration Helpers - Build and debug with Xcode
"""

import subprocess
import asyncio
import logging
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
import plistlib
import shutil

logger = logging.getLogger(__name__)


class XcodeIntegration:
    """Xcode build and debug integration"""

    def __init__(self):
        self.xcodebuild_path = shutil.which("xcodebuild") or "xcodebuild"
        self.swift_path = shutil.which("swift") or "swift"
        self.lldb_path = shutil.which("lldb") or "lldb"

    async def get_schemes(self, project_path: str = ".") -> List[str]:
        """Get available Xcode schemes"""
        try:
            process = await asyncio.create_subprocess_exec(
                self.xcodebuild_path,
                "-list",
                "-json",
                cwd=project_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=30.0
            )

            if process.returncode == 0:
                data = json.loads(stdout.decode("utf-8"))
                return data.get("project", {}).get("schemes", [])
            return []
        except Exception as e:
            logger.error(f"Failed to get Xcode schemes: {e}")
            return []

    async def get_targets(self, project_path: str = ".") -> List[str]:
        """Get available build targets"""
        try:
            process = await asyncio.create_subprocess_exec(
                self.xcodebuild_path,
                "-list",
                "-json",
                cwd=project_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=30.0
            )

            if process.returncode == 0:
                data = json.loads(stdout.decode("utf-8"))
                return data.get("project", {}).get("targets", [])
            return []
        except Exception as e:
            logger.error(f"Failed to get Xcode targets: {e}")
            return []

    async def build_scheme(
        self,
        scheme: str,
        configuration: str = "Debug",
        project_path: str = "."
    ) -> Dict[str, Any]:
        """Build an Xcode scheme"""
        try:
            process = await asyncio.create_subprocess_exec(
                self.xcodebuild_path,
                "build",
                "-scheme", scheme,
                "-configuration", configuration,
                cwd=project_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=300.0
            )

            output = stdout.decode("utf-8", errors="replace")
            errors = stderr.decode("utf-8", errors="replace")

            return {
                "status": "success" if process.returncode == 0 else "failed",
                "scheme": scheme,
                "configuration": configuration,
                "output": output,
                "errors": errors,
                "return_code": process.returncode
            }
        except asyncio.TimeoutError:
            return {
                "status": "failed",
                "scheme": scheme,
                "output": "",
                "errors": "Build timed out",
                "return_code": -1
            }
        except Exception as e:
            logger.error(f"Xcode build error: {e}")
            return {
                "status": "failed",
                "scheme": scheme,
                "output": "",
                "errors": str(e),
                "return_code": -1
            }

    async def test_scheme(
        self,
        scheme: str,
        configuration: str = "Debug",
        project_path: str = "."
    ) -> Dict[str, Any]:
        """Test an Xcode scheme"""
        try:
            process = await asyncio.create_subprocess_exec(
                self.xcodebuild_path,
                "test",
                "-scheme", scheme,
                "-configuration", configuration,
                cwd=project_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=300.0
            )

            output = stdout.decode("utf-8", errors="replace")
            errors = stderr.decode("utf-8", errors="replace")

            return {
                "status": "success" if process.returncode == 0 else "failed",
                "scheme": scheme,
                "configuration": configuration,
                "output": output,
                "errors": errors,
                "return_code": process.returncode
            }
        except asyncio.TimeoutError:
            return {
                "status": "failed",
                "scheme": scheme,
                "output": "",
                "errors": "Test timed out",
                "return_code": -1
            }
        except Exception as e:
            logger.error(f"Xcode test error: {e}")
            return {
                "status": "failed",
                "scheme": scheme,
                "output": "",
                "errors": str(e),
                "return_code": -1
            }

    async def analyze_scheme(
        self,
        scheme: str,
        project_path: str = "."
    ) -> Dict[str, Any]:
        """Analyze an Xcode scheme for issues"""
        try:
            process = await asyncio.create_subprocess_exec(
                self.xcodebuild_path,
                "analyze",
                "-scheme", scheme,
                cwd=project_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=300.0
            )

            output = stdout.decode("utf-8", errors="replace")
            errors = stderr.decode("utf-8", errors="replace")

            return {
                "status": "success" if process.returncode == 0 else "failed",
                "scheme": scheme,
                "output": output,
                "errors": errors,
                "return_code": process.returncode
            }
        except Exception as e:
            logger.error(f"Xcode analyze error: {e}")
            return {
                "status": "failed",
                "scheme": scheme,
                "output": "",
                "errors": str(e),
                "return_code": -1
            }

    async def get_project_info(self, project_path: str = ".") -> Dict[str, Any]:
        """Get Xcode project information"""
        try:
            # Find .xcodeproj or .xcworkspace
            project_dir = Path(project_path)
            xcodeproj = list(project_dir.glob("*.xcodeproj"))
            xcworkspace = list(project_dir.glob("*.xcworkspace"))

            project_file = None
            if xcworkspace:
                project_file = str(xcworkspace[0])
                project_type = "workspace"
            elif xcodeproj:
                project_file = str(xcodeproj[0])
                project_type = "project"
            else:
                return {
                    "status": "not_found",
                    "message": "No Xcode project or workspace found"
                }

            # Get schemes and targets
            schemes = await self.get_schemes(project_path)
            targets = await self.get_targets(project_path)

            return {
                "status": "found",
                "project_file": project_file,
                "project_type": project_type,
                "schemes": schemes,
                "targets": targets
            }
        except Exception as e:
            logger.error(f"Failed to get project info: {e}")
            return {
                "status": "error",
                "message": str(e)
            }

    def get_derived_data_path(self) -> Optional[str]:
        """Get Xcode derived data path"""
        derived_data = Path.home() / "Library/Developer/Xcode/DerivedData"
        if derived_data.exists():
            return str(derived_data)
        return None

    async def archive_scheme(
        self,
        scheme: str,
        archive_path: str,
        project_path: str = "."
    ) -> Dict[str, Any]:
        """Archive an Xcode scheme"""
        try:
            process = await asyncio.create_subprocess_exec(
                self.xcodebuild_path,
                "archive",
                "-scheme", scheme,
                "-archivePath", archive_path,
                cwd=project_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=600.0
            )

            output = stdout.decode("utf-8", errors="replace")
            errors = stderr.decode("utf-8", errors="replace")

            return {
                "status": "success" if process.returncode == 0 else "failed",
                "scheme": scheme,
                "archive_path": archive_path,
                "output": output,
                "errors": errors,
                "return_code": process.returncode
            }
        except Exception as e:
            logger.error(f"Xcode archive error: {e}")
            return {
                "status": "failed",
                "scheme": scheme,
                "output": "",
                "errors": str(e),
                "return_code": -1
            }

    async def export_archive(
        self,
        archive_path: str,
        export_path: str,
        export_options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Export an Xcode archive"""
        try:
            args = [
                self.xcodebuild_path,
                "-exportArchive",
                "-archivePath", archive_path,
                "-exportPath", export_path,
            ]

            if export_options:
                # Create temporary export options plist
                options_file = Path(export_path) / "ExportOptions.plist"
                options_file.parent.mkdir(parents=True, exist_ok=True)
                with open(options_file, "wb") as f:
                    plistlib.dump(export_options, f)
                args.extend(["-exportOptionsPlist", str(options_file)])

            process = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=300.0
            )

            output = stdout.decode("utf-8", errors="replace")
            errors = stderr.decode("utf-8", errors="replace")

            return {
                "status": "success" if process.returncode == 0 else "failed",
                "archive_path": archive_path,
                "export_path": export_path,
                "output": output,
                "errors": errors,
                "return_code": process.returncode
            }
        except Exception as e:
            logger.error(f"Export archive error: {e}")
            return {
                "status": "failed",
                "output": "",
                "errors": str(e),
                "return_code": -1
            }
