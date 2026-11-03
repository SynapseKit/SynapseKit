"""Serve any SynapseKit agent as a FastAPI REST endpoint.

``serve_agent(agent)`` (or the ``.serve()`` method mixed into the core agents)
turns an in-process agent into a single ``POST`` chat endpoint with Server-Sent
Events streaming when the agent exposes ``stream()`` and a plain-JSON fallback
otherwise. FastAPI and uvicorn live in the ``serve`` extra and are imported
lazily (only when an endpoint is built or served), so a core install never
loads them and ``import synapsekit`` stays extra-free.
"""

import json
import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)

_INSTALL_HINT = "agent.serve() needs FastAPI and uvicorn: pip install 'synapsekit[serve]'"


def build_agent_app(agent: Any, *, path: str = "/chat") -> "FastAPI":
    """Build (but do not run) a FastAPI app exposing ``agent`` at ``path``.

    Returned app has one ``POST {path}`` route accepting ``{"query": "..."}``.
    If ``agent`` has an async ``stream()`` the response is ``text/event-stream``
    (SSE: ``data: {"token": "..."}`` per chunk, terminated by ``data: [DONE]``);
    otherwise it awaits ``run()`` and returns ``{"response": "..."}``.

    Exposed separately from :func:`serve_agent` so the app can be unit-tested
    with ``fastapi.testclient.TestClient`` or mounted into a larger application.
    Raises ``ImportError`` (with the extras hint) if FastAPI is not installed.
    """
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import StreamingResponse
        from pydantic import BaseModel
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc

    class ChatRequest(BaseModel):
        query: str

    app = FastAPI(
        title="SynapseKit Agent Server",
        description="A SynapseKit agent exposed as a REST endpoint via agent.serve().",
        version="1.0.0",
    )

    @app.post(path)
    async def chat_endpoint(req: ChatRequest):  # type: ignore[no-untyped-def]
        if hasattr(agent, "stream"):

            async def event_generator() -> AsyncIterator[str]:
                try:
                    async for chunk in agent.stream(req.query):
                        # Agents yield str, but coerce defensively so a custom
                        # agent streaming non-str chunks doesn't abort the stream.
                        token = chunk if isinstance(chunk, str) else str(chunk)
                        yield f"data: {json.dumps({'token': token})}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception:
                    # Stream headers are already sent, so surface a generic SSE
                    # error event (never the raw exception) and log server-side.
                    logger.exception("agent.stream() failed while serving %s", path)
                    yield f"data: {json.dumps({'error': 'internal server error'})}\n\n"

            return StreamingResponse(event_generator(), media_type="text/event-stream")

        try:
            result = await agent.run(req.query)
        except Exception as exc:
            logger.exception("agent.run() failed while serving %s", path)
            raise HTTPException(status_code=500, detail="internal server error") from exc
        return {"response": result}

    return app


def serve_agent(
    agent: Any,
    host: str = "127.0.0.1",
    port: int = 8000,
    path: str = "/chat",
) -> None:
    """Serve ``agent`` over HTTP with uvicorn (blocks until the server stops).

    Binds to loopback (``127.0.0.1``) by default; pass ``host="0.0.0.0"`` to
    expose it on all interfaces. The endpoint has no authentication, so do that
    deliberately. Raises ``ImportError`` (with the extras hint) if FastAPI or
    uvicorn are not installed.
    """
    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc

    app = build_agent_app(agent, path=path)
    logger.info(
        "SynapseKit agent server listening on http://%s:%s%s (docs at /docs)",
        host,
        port,
        path,
    )
    uvicorn.run(app, host=host, port=port)


class ServeMixin:
    """Adds a native ``.serve()`` to an agent that implements ``run``/``stream``.

    Mixed into the core agents so ``agent.serve(port=8000)`` works out of the
    box. ``serve_agent`` is imported lazily inside the method, keeping FastAPI
    and uvicorn out of the agents' import path.
    """

    def serve(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        path: str = "/chat",
    ) -> None:
        """Serve this agent as a REST endpoint. See :func:`serve_agent`."""
        serve_agent(self, host=host, port=port, path=path)
