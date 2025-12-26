"""Command-line interface for dredge."""

import argparse
from dredge import __version__


def main():
    """Main entry point for the dredge CLI."""
    parser = argparse.ArgumentParser(
        prog="dredge",
        description="DREDGE: A lightweight system for lifting, preserving, and releasing insights",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    
    args = parser.parse_args()
    
    # Future: Add subcommands and functionality here
    # For now, if no arguments, show help
    parser.print_help()


if __name__ == "__main__":
    main()
