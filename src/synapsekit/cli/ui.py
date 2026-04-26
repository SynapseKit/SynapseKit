"""SynapseKit observability UI — ``synapsekit ui`` command."""

from __future__ import annotations

import argparse
import threading
import time
import webbrowser


def run_ui(args: argparse.Namespace) -> None:
    """Start the observability dashboard and open a browser tab."""
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
