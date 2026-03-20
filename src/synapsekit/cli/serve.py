"""``synapsekit serve`` — deploy any SynapseKit app as FastAPI in one command."""

from __future__ import annotations

import importlib
import sys
from typing import Any


def _import_object(app_path: str) -> Any:
    """Import an object from ``module:attribute`` notation."""
    if ":" not in app_path:
        raise ValueError(
            f"Invalid app path '{app_path}'. Expected 'module:attribute' format (e.g. 'my_app:rag')"
        )
    module_path, attr_name = app_path.rsplit(":", 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, attr_name)


def _detect_type(obj: Any) -> str:
    """Detect the type of a SynapseKit object."""
    cls_name = type(obj).__name__
    # Check MRO class names for flexibility
    mro_names = {c.__name__ for c in type(obj).__mro__}

    if "RAGPipeline" in mro_names or cls_name == "RAGPipeline":
        return "rag"
    if "RAG" in mro_names or cls_name == "RAG":
        return "rag"
    if "CompiledGraph" in mro_names or cls_name == "CompiledGraph":
        return "graph"
    if "ReActAgent" in mro_names or cls_name == "ReActAgent":
        return "agent"
    if "FunctionCallingAgent" in mro_names or cls_name == "FunctionCallingAgent":
        return "agent"
    # Fallback: treat as generic callable
    return "agent"


def build_app(obj: Any, app_type: str | None = None) -> Any:
    """Build a FastAPI app from a SynapseKit object.

    Args:
        obj: A RAGPipeline, CompiledGraph, ReActAgent, etc.
        app_type: Override auto-detection (``"rag"``, ``"graph"``, ``"agent"``).

    Returns:
        A FastAPI application instance.
    """
    try:
        from fastapi import FastAPI
    except ImportError:
        raise ImportError(
            "FastAPI is required for 'synapsekit serve'. "
            "Install it with: pip install synapsekit[serve]"
        ) from None

    detected = app_type or _detect_type(obj)
    app = FastAPI(title=f"SynapseKit — {detected}", version="1.0.0")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    if detected == "rag":
        _add_rag_routes(app, obj)
    elif detected == "graph":
        _add_graph_routes(app, obj)
    elif detected == "agent":
        _add_agent_routes(app, obj)

    return app


def _add_rag_routes(app: Any, obj: Any) -> None:
    """Add RAG-specific routes."""
    from fastapi.responses import JSONResponse

    @app.post("/query", response_class=JSONResponse)
    async def query(request: dict[str, Any]) -> Any:
        q = request.get("query", request.get("question", ""))
        if not q:
            return JSONResponse({"error": "Missing 'query' field"}, status_code=400)
        result = await obj.aquery(q) if hasattr(obj, "aquery") else obj.query(q)
        if isinstance(result, str):
            return JSONResponse({"answer": result})
        return JSONResponse({"answer": str(result)})


def _add_graph_routes(app: Any, obj: Any) -> None:
    """Add Graph-specific routes."""
    import json

    from fastapi.responses import JSONResponse, StreamingResponse

    @app.post("/run", response_class=JSONResponse)
    async def run(request: dict[str, Any]) -> Any:
        state = request.get("state", request)
        result = await obj.arun(state) if hasattr(obj, "arun") else obj.run(state)
        return JSONResponse({"result": result})

    @app.get("/stream", response_class=StreamingResponse)
    async def stream(state: str = "{}") -> Any:
        parsed = json.loads(state)

        async def event_generator():  # type: ignore[return]
            if hasattr(obj, "astream"):
                async for event in obj.astream(parsed):
                    yield f"data: {json.dumps(event)}\n\n"
            else:
                result = obj.run(parsed)
                yield f"data: {json.dumps(result)}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")


def _add_agent_routes(app: Any, obj: Any) -> None:
    """Add Agent-specific routes."""
    from fastapi.responses import JSONResponse

    @app.post("/run", response_class=JSONResponse)
    async def run(request: dict[str, Any]) -> Any:
        prompt = request.get("prompt", request.get("input", ""))
        if not prompt:
            return JSONResponse({"error": "Missing 'prompt' field"}, status_code=400)
        result = await obj.arun(prompt) if hasattr(obj, "arun") else obj.run(prompt)
        if isinstance(result, str):
            return JSONResponse({"answer": result})
        return JSONResponse({"answer": str(result)})


def run_serve(args: Any) -> None:
    """Execute the ``synapsekit serve`` command."""
    try:
        import uvicorn
    except ImportError:
        print(
            "Error: uvicorn is required for 'synapsekit serve'. "
            "Install it with: pip install synapsekit[serve]",
            file=sys.stderr,
        )
        sys.exit(1)

    obj = _import_object(args.app)
    app = build_app(obj)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
