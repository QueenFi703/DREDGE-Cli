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


def _merge_server_args(args, config_key: str, default_host: str, default_port: int):
    """Merge CLI arguments with configuration for server commands."""
    config = load_config()
    server_config = config.get(config_key, {})
    
    host = args.host if hasattr(args, 'host') and args.host != default_host else server_config.get("host", default_host)
    port = args.port if hasattr(args, 'port') and args.port != default_port else server_config.get("port", default_port)
    debug = args.debug if hasattr(args, 'debug') else False
    debug = debug or server_config.get("debug", False)
    
    return host, port, debug, server_config


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
        default=3000, 
        help="Port to listen on (default: 3000)"
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

    # run-agent command
    run_agent_parser = subparsers.add_parser(
        "run-agent",
        help="Dispatch an AI inference workload to the Jetson Thor GPU",
        formatter_class=formatter,
    )
    run_agent_parser.add_argument(
        "--agent",
        choices=["vision", "planner", "reasoning", "pipeline"],
        default="pipeline",
        help=(
            "Agent to run: 'vision', 'planner', 'reasoning', or 'pipeline' "
            "(full Vision→Planner→Reasoning chain, default: pipeline)"
        ),
    )
    run_agent_parser.add_argument(
        "--input",
        metavar="FLOATS",
        default=None,
        help=(
            "Comma-separated list of float values to use as input data "
            "(e.g. '0.1,0.2,0.3').  Defaults to a random tensor."
        ),
    )
    run_agent_parser.add_argument(
        "--input-dim",
        type=int,
        default=64,
        help="Input feature dimensionality (default: 64; used with --agent=vision)",
    )
    run_agent_parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Target device for JetsonThor adapter (default: auto-detect)",
    )
    run_agent_parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON",
    )

    # agent-server command
    agent_server_parser = subparsers.add_parser(
        "agent-server",
        help="Start the DREDGE MCP AI-agent server (Jetson Thor / FastAPI)",
        formatter_class=formatter,
    )
    agent_server_parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    agent_server_parser.add_argument(
        "--port",
        type=int,
        default=3003,
        help="Port to listen on (default: 3003)",
    )
    agent_server_parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Target device for JetsonThor adapter (default: auto-detect)",
    )
    agent_server_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (development mode)",
    )

    # GitHub event command
    github_event_parser = subparsers.add_parser(
        "github-event",
        help="Process one or more GitHub events (streaming batch mode)",
        formatter_class=formatter,
    )
    github_event_parser.add_argument(
        "--event-file",
        metavar="FILE",
        help="JSON file containing a single event or a newline-delimited list of events",
    )
    github_event_parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read newline-delimited JSON events from stdin",
    )
    github_event_parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent event-processing workers (default: 4)",
    )
    github_event_parser.add_argument(
        "--json",
        action="store_true",
        help="Output processing summary as JSON",
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
        host, port, debug, _ = _merge_server_args(args, "server", "0.0.0.0", 3000)
        
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
        host, port, debug, mcp_config = _merge_server_args(args, "mcp", "0.0.0.0", 3002)
        device = args.device if args.device != "auto" else mcp_config.get("device", "auto")
        
        try:
            validate_server_config(host, port, debug)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        
        from .mcp_server import run_mcp_server
        run_mcp_server(host=host, port=port, debug=debug, device=device)
        return 0
    
    if args.command == "github-event":
        from .event_stream_handler import (
            process_events_from_list,
            process_events_from_stdin,
        )

        def _default_handler(event):
            """Echo the event type back as a minimal stub handler."""
            return {"event_type": event.get("type", "unknown"), "status": "processed"}

        output_json = getattr(args, "json", False)

        if getattr(args, "stdin", False):
            summary = process_events_from_stdin(
                handler=_default_handler,
                num_workers=args.workers,
            )
        elif args.event_file:
            event_path = Path(args.event_file)
            if not event_path.exists():
                print(f"Error: event file not found: {args.event_file}", file=sys.stderr)
                return 1
            raw = event_path.read_text()
            # Support both a single JSON object and newline-delimited objects
            events = []
            for line in raw.splitlines():
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
            if not events:
                print("Error: no valid JSON events found in file", file=sys.stderr)
                return 1
            summary = process_events_from_list(
                events=events,
                handler=_default_handler,
                num_workers=args.workers,
            )
        else:
            github_event_parser.print_help()
            return 0

        if output_json:
            print(json.dumps(summary, indent=2))
        else:
            print(f"Processed: {summary['total_processed']} events")
            print(f"Successful: {summary['successful']}")
            print(f"Errors:    {summary['errors']}")
            print(f"Elapsed:   {summary['elapsed_seconds']:.3f}s")
            print(f"Throughput:{summary['throughput_per_sec']:.1f} events/sec")
        return 0

    if args.command == "run-agent":
        import torch

        from .events.ai_event import AIEvent, dispatch
        from .hardware.jetson_thor import JetsonThor

        agent_type = args.agent
        device = args.device
        output_json = getattr(args, "json", False)

        # Build input tensor
        if args.input:
            try:
                values = [float(v) for v in args.input.split(",")]
            except ValueError:
                print(
                    "Error: --input must be a comma-separated list of floats "
                    "(e.g. '0.1,0.2,0.3')",
                    file=sys.stderr,
                )
                return 1
        else:
            values = None

        try:
            if agent_type == "pipeline":
                from .agents.planner_agent import PlannerAgent
                from .agents.reasoning_agent import ReasoningAgent
                from .agents.vision_agent import VisionAgent

                input_dim = len(values) if values else args.input_dim
                sensor_input = (
                    torch.tensor(values, dtype=torch.float32).unsqueeze(0)
                    if values
                    else torch.randn(1, input_dim)
                )
                vision = VisionAgent(input_dim=input_dim, output_dim=32)
                planner = PlannerAgent(input_dim=32, num_actions=8)
                reasoning = ReasoningAgent(input_dim=8, output_dim=1)

                embedding = vision.run(sensor_input)
                action_logits = planner.run(embedding)
                decision = reasoning.run(action_logits)

                result = {
                    "pipeline": "Vision → Planner → Reasoning",
                    "embedding": embedding.detach().cpu().tolist(),
                    "action_logits": action_logits.detach().cpu().tolist(),
                    "decision": decision.detach().cpu().tolist(),
                }
            else:
                # Single-agent dispatch
                if agent_type == "vision":
                    from .agents.vision_agent import VisionAgent

                    input_dim = len(values) if values else args.input_dim
                    agent = VisionAgent(input_dim=input_dim, output_dim=32)
                elif agent_type == "planner":
                    from .agents.planner_agent import PlannerAgent

                    input_dim = len(values) if values else args.input_dim
                    agent = PlannerAgent(input_dim=input_dim, num_actions=8)
                else:  # reasoning
                    from .agents.reasoning_agent import ReasoningAgent

                    input_dim = len(values) if values else args.input_dim
                    agent = ReasoningAgent(input_dim=input_dim, output_dim=1)

                tensor = (
                    torch.tensor(values, dtype=torch.float32).unsqueeze(0)
                    if values
                    else torch.randn(1, input_dim)
                )
                output = agent.run(tensor)
                result = {
                    "agent": agent_type,
                    "result": output.detach().cpu().tolist(),
                }

            if output_json:
                print(json.dumps(result, indent=2))
            else:
                if "pipeline" in result:
                    print(f"Pipeline : {result['pipeline']}")
                    print(f"Decision : {result['decision']}")
                else:
                    print(f"Agent  : {result['agent']}")
                    print(f"Result : {result['result']}")
            return 0

        except Exception as exc:
            print(f"Error running agent: {exc}", file=sys.stderr)
            return 1

    if args.command == "agent-server":
        try:
            from .mcp.server import run_mcp_agent_server
        except ImportError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

        try:
            validate_server_config(args.host, args.port, False)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

        run_mcp_agent_server(
            host=args.host,
            port=args.port,
            device=args.device,
            reload=args.reload,
        )
        return 0

    parser.print_help()
    return 0

if __name__ == "__main__":
    sys.exit(main())
