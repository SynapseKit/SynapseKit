"""SynapseKit observability UI — ``synapsekit ui`` command."""

from __future__ import annotations

import argparse
import threading
import time
import webbrowser


def _run_live(args: argparse.Namespace) -> None:
    """Start the zero-dependency live dashboard (stdlib http.server + SSE)."""
    from ..live import enable

    port = 7900 if getattr(args, "port", 7860) == 7860 else args.port
    from ..live.server import serve

    serve(host=args.host, port=port, open_browser=True)
    enable(open_browser=False, quiet=True)  # turn on silent span instrumentation
    print("Watching for SynapseKit runs. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("\nStopped.")


def run_ui(args: argparse.Namespace) -> None:
    """Start the observability dashboard and open a browser tab."""
    if getattr(args, "live", False):
        _run_live(args)
        return

    import uvicorn

    from .ui_server import create_app

    host: str = args.host
    port: int = args.port

    app = create_app()
    url = f"http://{host}:{port}"
    print(f"Starting SynapseKit Observability Dashboard at {url}")
    print("Press Ctrl+C to stop.")

    def _open_browser() -> None:
        time.sleep(1.2)
        webbrowser.open(url)

    threading.Thread(target=_open_browser, daemon=True).start()

    uvicorn.run(app, host=host, port=port)
