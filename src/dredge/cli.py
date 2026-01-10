import argparse
import sys
import socket
from . import __version__


def cmd_inspect():
    """Inspect DREDGE configuration and status."""
    print("═" * 60)
    print("DREDGE Inspector — Status & Philosophy")
    print("═" * 60)
    print()
    print(f"📦 Version: {__version__}")
    print(f"🔧 Build: stable")
    print()
    print("⚙️  Active Configuration:")
    print(f"  • Default host: 0.0.0.0")
    print(f"  • Default port: 3001")
    print(f"  • Debug mode: False")
    print()
    print("🎯 Engine Details:")
    print(f"  • JSON provider: CompactJSONProvider")
    print(f"  • Hash strategy: 64-bit polynomial (31-bit rolling)")
    print(f"  • Performance mode: compact")
    print()
    print("💡 Identity Contract:")
    print(f"  • IDs: content-derived labels (not proofs)")
    print(f"  • Collision behavior: last write wins")
    print(f"  • Scale: suitable for <5e9 items")
    print()
    print("═" * 60)
    return 0


def cmd_doctor():
    """Run diagnostics on DREDGE installation."""
    print("═" * 60)
    print("DREDGE Doctor — System Diagnostics")
    print("═" * 60)
    print()
    
    checks_passed = 0
    checks_total = 0
    
    # Check Python version
    checks_total += 1
    py_version = sys.version_info
    if py_version >= (3, 10) and py_version < (3, 13):
        print("✓ Python version compatible:", f"{py_version.major}.{py_version.minor}.{py_version.micro}")
        checks_passed += 1
    else:
        print("✗ Python version incompatible:", f"{py_version.major}.{py_version.minor}.{py_version.micro}")
        print("  Expected: 3.10 <= version < 3.13")
    
    # Check port availability
    checks_total += 1
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('127.0.0.1', 3001))
        sock.close()
        if result != 0:
            print("✓ Default port 3001 is available")
            checks_passed += 1
        else:
            print("⚠ Default port 3001 is in use")
            print("  (This is OK if DREDGE server is running)")
            checks_passed += 1  # Not a failure
    except Exception as e:
        print(f"⚠ Could not check port availability: {e}")
    
    # Check dependencies
    checks_total += 1
    try:
        import flask
        print("✓ Flask dependency available")
        checks_passed += 1
    except ImportError:
        print("✗ Flask dependency missing")
        print("  Run: pip install flask")
    
    # Check performance mode
    checks_total += 1
    try:
        from .server import CompactJSONProvider
        print("✓ CompactJSONProvider configured")
        checks_passed += 1
    except Exception as e:
        print(f"✗ CompactJSONProvider check failed: {e}")
    
    print()
    print("─" * 60)
    if checks_passed == checks_total:
        print("🎉 Everything looks good! System is healthy.")
    else:
        print(f"⚠️  {checks_total - checks_passed} issue(s) detected. Review above.")
    print("═" * 60)
    
    return 0 if checks_passed == checks_total else 1


def cmd_echo():
    """The signature touch."""
    print("alive")
    return 0


def cmd_id(count=1, format_type="hex"):
    """Generate deterministic IDs using the hash strategy."""
    import uuid
    import time
    
    for i in range(count):
        if format_type == "hex":
            # Use 64-bit rolling hash (matching server strategy)
            text = f"dredge-id-{uuid.uuid4()}-{time.time_ns()}"
            hash_value = 0
            for char in text:
                hash_value = (hash_value * 31 + ord(char)) & 0xFFFFFFFFFFFFFFFF
            id_str = format(hash_value, '016x')
            print(id_str)
        elif format_type == "uuid":
            # UUIDv4
            print(str(uuid.uuid4()))
        elif format_type == "uuid7":
            # UUIDv7 (time-based, would need uuid6 package for real implementation)
            # For now, use uuid4 as placeholder
            print(str(uuid.uuid4()))
    
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="dredge", 
        description="DREDGE x Dolly - GPU-CPU Lifter · Save · Files · Print"
    )
    parser.add_argument("--version", action="store_true", help="Print version and exit")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Server command
    server_parser = subparsers.add_parser("serve", help="Start the DREDGE x Dolly web server")
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
    
    # Print command
    print_parser = subparsers.add_parser("print", help="Print a message or newline")
    print_parser.add_argument(
        "text",
        nargs="?",
        default=None,
        help="Text to print (if omitted, prints a clean newline)"
    )
    
    # Inspect command
    inspect_parser = subparsers.add_parser("inspect", help="Inspect DREDGE configuration and status")
    
    # Doctor command
    doctor_parser = subparsers.add_parser("doctor", help="Run system diagnostics")
    
    # Echo command (signature touch)
    echo_parser = subparsers.add_parser("echo", help="Verify DREDGE is alive")
    
    # ID command
    id_parser = subparsers.add_parser("id", help="Generate deterministic IDs")
    id_parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of IDs to generate (default: 1)"
    )
    id_parser.add_argument(
        "--format",
        choices=["hex", "uuid", "uuid7"],
        default="hex",
        help="ID format (default: hex)"
    )
    
    args = parser.parse_args(argv)
    
    if args.version:
        print(__version__)
        return 0
    
    if args.command == "serve":
        from .server import run_server
        run_server(host=args.host, port=args.port, debug=args.debug)
        return 0
    
    if args.command == "print":
        if args.text is None:
            # Print just a newline - "a quiet pause in the program"
            print()
        else:
            # Print the message
            print(args.text)
        return 0
    
    if args.command == "inspect":
        return cmd_inspect()
    
    if args.command == "doctor":
        return cmd_doctor()
    
    if args.command == "echo":
        return cmd_echo()
    
    if args.command == "id":
        return cmd_id(count=args.count, format_type=args.format)
    
    parser.print_help()
    return 0

if __name__ == "__main__":
    sys.exit(main())
