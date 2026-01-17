import argparse
import os
import platform
import shutil
import sys
from . import __version__


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
    
    args = parser.parse_args(argv)
    
    if args.version:
        print(__version__)
        return 0
    
    if args.command == "serve":
        from .server import run_server
        run_server(host=args.host, port=args.port, debug=args.debug)
        return 0
    
    if args.command == "mcp":
        from .mcp_server import run_mcp_server
        run_mcp_server(host=args.host, port=args.port, debug=args.debug)
        return 0
    
    parser.print_help()
    return 0

if __name__ == "__main__":
    sys.exit(main())
