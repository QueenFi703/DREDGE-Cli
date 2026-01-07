import argparse
import sys
from . import __version__

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
    
    parser.print_help()
    return 0

if __name__ == "__main__":
    sys.exit(main())
