import argparse
import os
import platform
import shutil
import sys
import json
from pathlib import Path
from . import __version__
from .health import get_system_info, format_system_info, check_health, validate_server_config
from .config import load_config, save_config, init_config, get_config_path, DEFAULT_CONFIG


def _detect_mobile_context():
    uname = platform.uname()
    is_termux = "TERMUX_VERSION" in os.environ
    # Heuristic: iSH often reports Alpine in release; keep conservative
    is_ish = "alpine" in uname.release.lower() or "ish" in uname.release.lower()
    width = shutil.get_terminal_size(fallback=(80, 24)).columns
    if is_termux or is_ish:
        width = min(width, 80)
    return {
        "is_termux": is_termux,
        "is_ish": is_ish,
        "is_mobile": is_termux or is_ish,
        "term_width": width,
    }


class MobileHelpFormatter(argparse.HelpFormatter):
    def __init__(self, prog, width=80, **kwargs):
        super().__init__(prog, width=width, max_help_position=24, **kwargs)


def main(argv=None):
    ctx = _detect_mobile_context()
    formatter = lambda prog: MobileHelpFormatter(prog, width=ctx["term_width"])
    parser = argparse.ArgumentParser(
        prog="dredge",
        description="DREDGE x Dolly - GPU-CPU Lifter · Save · Files · Print",
        formatter_class=formatter,
    )
    parser.add_argument("--version", action="store_true", help="Print version and exit")
    parser.add_argument(
        "--version-info", 
        action="store_true", 
        help="Print detailed version and system information"
    )
    parser.add_argument(
        "--no-spinner",
        action="store_true",
        help="Disable spinners/progress (default: enabled; disable for CI/pipes)",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Server command
    server_parser = subparsers.add_parser(
        "serve", help="Start the DREDGE x Dolly web server", formatter_class=formatter
    )
    server_parser.add_argument(
        "--host", 
        default="0.0.0.0", 
        help="Host to bind to (default: 0.0.0.0)"
    )
    server_parser.add_argument(
        "--port", 
        type=int, 
        default=3001, 
        help="Port to listen on (default: 3001)"
    )
    server_parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug mode"
    )
    server_parser.add_argument(
        "--threads",
        type=int,
        default=1 if ctx["is_mobile"] else 0,
        help="Worker threads (mobile-safe default: 1; set >1 to override)",
    )
    
    # MCP Server command
    mcp_parser = subparsers.add_parser(
        "mcp", help="Start the DREDGE MCP server (Quasimoto models)", formatter_class=formatter
    )
    mcp_parser.add_argument(
        "--host", 
        default="0.0.0.0", 
        help="Host to bind to (default: 0.0.0.0)"
    )
    mcp_parser.add_argument(
        "--port", 
        type=int, 
        default=3002, 
        help="Port to listen on (default: 3002)"
    )
    mcp_parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug mode"
    )
    mcp_parser.add_argument(
        "--threads",
        type=int,
        default=1 if ctx["is_mobile"] else 0,
        help="Worker threads (mobile-safe default: 1; set >1 to override)",
    )
    mcp_parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device to use for computation (default: auto-detect)"
    )
    
    # Health check command
    health_parser = subparsers.add_parser(
        "health", help="Check system health and dependencies", formatter_class=formatter
    )
    health_parser.add_argument(
        "--json",
        action="store_true",
        help="Output health check as JSON"
    )
    
    # Info command
    info_parser = subparsers.add_parser(
        "info", help="Show system information", formatter_class=formatter
    )
    
    # Config command
    config_parser = subparsers.add_parser(
        "config", help="Manage configuration", formatter_class=formatter
    )
    config_subparsers = config_parser.add_subparsers(dest="config_action", help="Config actions")
    
    config_show = config_subparsers.add_parser(
        "show", help="Show current configuration", formatter_class=formatter
    )
    config_init = config_subparsers.add_parser(
        "init", help="Initialize default configuration file", formatter_class=formatter
    )
    config_path_parser = config_subparsers.add_parser(
        "path", help="Show configuration file path", formatter_class=formatter
    )
    
    args = parser.parse_args(argv)
    
    if args.version:
        print(__version__)
        return 0
    
    if args.version_info:
        print(f"DREDGE version {__version__}")
        print()
        sys_info = get_system_info()
        print(format_system_info(sys_info))
        return 0
    
    if args.command == "health":
        health = check_health()
        if args.json:
            print(json.dumps(health, indent=2))
        else:
            print(f"Health Status: {health['status']}")
            print()
            print("Dependency Checks:")
            for dep, available in health['checks']['dependencies'].items():
                status = "✓" if available else "✗"
                print(f"  {status} {dep}")
            
            if 'missing_dependencies' in health and health['missing_dependencies']:
                print()
                print("Missing dependencies:")
                for dep in health['missing_dependencies']:
                    print(f"  - {dep}")
                print()
                print("Run 'make install-python' or 'pip install -r requirements.txt' to install")
                return 1
        return 0 if health['status'] == 'healthy' else 1
    
    if args.command == "info":
        sys_info = get_system_info()
        print(format_system_info(sys_info))
        return 0
    
    if args.command == "config":
        if args.config_action == "show":
            config = load_config()
            print(json.dumps(config, indent=2))
            return 0
        elif args.config_action == "init":
            try:
                path = init_config()
                print(f"Configuration file created at: {path}")
                print()
                print("Edit the file to customize your settings.")
                return 0
            except FileExistsError as e:
                print(f"Error: {e}", file=sys.stderr)
                print(f"Use 'dredge-cli config show' to view current config", file=sys.stderr)
                return 1
        elif args.config_action == "path":
            path = get_config_path()
            exists = "exists" if path.exists() else "does not exist"
            print(f"{path} ({exists})")
            return 0
        else:
            config_parser.print_help()
            return 0
    
    if args.command == "serve":
        # Load config and merge with CLI args
        config = load_config()
        server_config = config.get("server", {})
        
        host = args.host if args.host != "0.0.0.0" else server_config.get("host", "0.0.0.0")
        port = args.port if args.port != 3001 else server_config.get("port", 3001)
        debug = args.debug or server_config.get("debug", False)
        
        try:
            validate_server_config(host, port, debug)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        
        from .server import run_server
        run_server(host=host, port=port, debug=debug)
        return 0
    
    if args.command == "mcp":
        # Load config and merge with CLI args
        config = load_config()
        mcp_config = config.get("mcp", {})
        
        host = args.host if args.host != "0.0.0.0" else mcp_config.get("host", "0.0.0.0")
        port = args.port if args.port != 3002 else mcp_config.get("port", 3002)
        debug = args.debug or mcp_config.get("debug", False)
        device = args.device if args.device != "auto" else mcp_config.get("device", "auto")
        
        try:
            validate_server_config(host, port, debug)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        
        from .mcp_server import run_mcp_server
        run_mcp_server(host=host, port=port, debug=debug, device=device)
        return 0
    
    parser.print_help()
    return 0

if __name__ == "__main__":
    sys.exit(main())
