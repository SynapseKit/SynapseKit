import json
import logging
import sys
from typing import Any

logger = logging.getLogger(__name__)


def serve_agent(agent: Any, host: str = "0.0.0.0", port: int = 8000, path: str = "/chat"):
    """
    Serve any SynapseKit agent as a FastAPI REST endpoint with SSE streaming support.
    Requires `fastapi` and `uvicorn` to be installed.
    """
    try:
        import uvicorn
        from fastapi import FastAPI
        from fastapi.responses import StreamingResponse
        from pydantic import BaseModel
    except ImportError:
        logger.error("FastAPI and Uvicorn are required to use agent.serve()")
        logger.error("Install them with: pip install 'synapsekit[serve]'")
        sys.exit(1)

    class ChatRequest(BaseModel):
        query: str

    app = FastAPI(
        title="SynapseKit Agent Server",
        description="Instantly converted Agent to REST API using agent.serve()",
        version="1.0.0",
    )

    @app.post(path)
    async def chat_endpoint(req: ChatRequest):
        # Prefer streaming if available
        if hasattr(agent, "stream"):

            async def event_generator():
                try:
                    async for chunk in agent.stream(req.query):
                        # Ensure we don't break SSE format with internal newlines
                        # Safe basic payload for SSE
                        payload = json.dumps({"token": chunk})
                        yield f"data: {payload}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"

            return StreamingResponse(event_generator(), media_type="text/event-stream")

        # Fallback to run()
        try:
            ans = await agent.run(req.query)
            return {"response": ans}
        except Exception as e:
            return {"error": str(e)}

    print(f"\n{'=' * 60}")
    print(f"🚀 SynapseKit Agent Server starting at http://{host}:{port}{path}")
    print(f"📄 Interactive Swagger API Docs available at http://{host}:{port}/docs")
    print(f"{'=' * 60}\n")
    uvicorn.run(app, host=host, port=port)


def patch_agents():
    """Monkey-patch .serve() onto standard agents."""
    try:
        from ..agents.function_calling import FunctionCallingAgent
        from ..agents.react import ReActAgent
        from ..agents.reasoning_agent import ReasoningAgent

        def _serve_method(
            self: Any, host: str = "0.0.0.0", port: int = 8000, path: str = "/chat"
        ) -> None:
            serve_agent(self, host=host, port=port, path=path)

        ReActAgent.serve = _serve_method  # type: ignore[attr-defined]
        FunctionCallingAgent.serve = _serve_method  # type: ignore[attr-defined]
        ReasoningAgent.serve = _serve_method  # type: ignore[attr-defined]
    except ImportError:
        pass
