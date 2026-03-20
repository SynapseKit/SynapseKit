"""SynapseKit CLI — ``synapsekit serve`` and ``synapsekit test``."""

from __future__ import annotations

import argparse
import sys


def _add_serve_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser("serve", help="Serve a SynapseKit app as a FastAPI server")
    p.add_argument("app", help="Import path, e.g. 'my_module:rag'")
    p.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    p.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    p.add_argument("--reload", action="store_true", help="Enable auto-reload")


def _add_test_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser("test", help="Run evaluation test suites")
    p.add_argument("path", nargs="?", default=".", help="Directory or file to scan (default: .)")
    p.add_argument(
        "--threshold", type=float, default=0.7, help="Min score threshold (default: 0.7)"
    )
    p.add_argument(
        "--format",
        dest="output_format",
        choices=["json", "table"],
        default="table",
        help="Output format (default: table)",
    )


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="synapsekit",
        description="SynapseKit CLI — serve apps and run evaluations",
    )
    parser.add_argument("--version", action="store_true", help="Show version and exit")

    subparsers = parser.add_subparsers(dest="command")
    _add_serve_parser(subparsers)
    _add_test_parser(subparsers)

    args = parser.parse_args(argv)

    if args.version:
        from synapsekit import __version__

        print(f"synapsekit {__version__}")
        return

    if args.command == "serve":
        from .serve import run_serve

        run_serve(args)
    elif args.command == "test":
        from .test import run_test

        run_test(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
