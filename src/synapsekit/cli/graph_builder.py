import argparse
import json
import threading
import time
import webbrowser
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import FastAPI

from synapsekit.graph import StateGraph
from synapsekit.graph.mermaid import get_mermaid

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>SynapseKit Graph Builder</title>
    <style>
        body { font-family: sans-serif; display: flex; height: 100vh; margin: 0; }
        #sidebar { width: 350px; background: #f0f0f0; padding: 20px; box-sizing: border-box; border-right: 1px solid #ccc; display: flex; flex-direction: column; gap: 10px; }
        #main { flex-grow: 1; display: flex; flex-direction: column; }
        #canvas { flex-grow: 1; background: #fafafa; position: relative; overflow: auto; padding: 20px; }
        #bottom-panel { height: 350px; border-top: 1px solid #ccc; display: flex; }
        #mermaid-preview, #python-preview { flex: 1; padding: 10px; overflow: auto; white-space: pre-wrap; font-family: monospace; border-right: 1px solid #ccc; }
        textarea { width: 100%; height: 250px; font-family: monospace; }
        button { padding: 8px; cursor: pointer; background: #0066cc; color: white; border: none; border-radius: 4px; }
        button:hover { background: #0052a3; }
        #json-editor { flex-grow: 1; display: flex; flex-direction: column; }
    </style>
</head>
<body>
    <div id="sidebar">
        <h3>Graph Builder</h3>
        <p>Edit JSON schema below to update graph:</p>
        <div id="json-editor">
            <textarea id="json-input">{
  "nodes": [
    {"id": "retrieve", "type": "rag_node", "config": {"top_k": 5}},
    {"id": "generate", "type": "llm_node", "config": {"model": "gpt-4o-mini"}}
  ],
  "edges": [
    {"from": "retrieve", "to": "generate"},
    {"from": "generate", "to": "END"}
  ]
}</textarea>
        </div>
        <button onclick="updatePreviews()">Sync & Preview</button>
        <button onclick="downloadPython()">Export Python</button>
    </div>
    <div id="main">
        <div id="canvas">
            <h2>Visual Workflow Builder Canvas</h2>
            <p>To implement full drag-and-drop, integrate a library like React Flow. For this prototype, edit the JSON in the sidebar to sync the graph, code, and diagrams.</p>
        </div>
        <div id="bottom-panel">
            <div id="mermaid-preview">Mermaid will appear here</div>
            <div id="python-preview">Python code will appear here</div>
        </div>
    </div>
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({ startOnLoad: false });

        window.updatePreviews = async function() {
            const jsonStr = document.getElementById('json-input').value;
            try {
                // Update mermaid
                const resMermaid = await fetch('/api/mermaid', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: jsonStr
                });
                const mermaidData = await resMermaid.json();
                if (mermaidData.error) {
                    document.getElementById('mermaid-preview').textContent = "Error: " + mermaidData.error;
                } else {
                    document.getElementById('mermaid-preview').innerHTML = `<div class="mermaid">${mermaidData.mermaid}</div>`;
                    mermaid.run();
                }

                // Update Python
                const resPython = await fetch('/api/codegen', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: jsonStr
                });
                const pythonData = await resPython.json();
                if (pythonData.error) {
                    document.getElementById('python-preview').textContent = "Error: " + pythonData.error;
                } else {
                    document.getElementById('python-preview').textContent = pythonData.code;
                }
            } catch (e) {
                console.error(e);
            }
        };

        window.downloadPython = function() {
            const code = document.getElementById('python-preview').textContent;
            const blob = new Blob([code], { type: 'text/x-python' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'graph.py';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
        };

        // Initial load
        setTimeout(updatePreviews, 500);
    </script>
</body>
</html>
"""


def generate_python_code(payload: dict[str, Any]) -> str:
    lines = [
        "from synapsekit.graph import StateGraph, END",
        "from synapsekit.graph.node import rag_node, llm_node, agent_node",
        "",
        "def create_graph():",
        "    graph = StateGraph()",
    ]

    for node in payload.get("nodes", []):
        node_id = node["id"]
        node_type = node.get("type", "custom_node")
        config = node.get("config", {})
        config_str = ", ".join(f"{k}={v!r}" for k, v in config.items())

        if node_type in ("rag_node", "llm_node", "agent_node"):
            lines.append(f"    graph.add_node({node_id!r}, {node_type}({config_str}))")
        else:
            lines.append(f"    # TODO: Define function for {node_type}")
            lines.append(f"    graph.add_node({node_id!r}, {node_type})")

    for edge in payload.get("edges", []):
        lines.append(f"    graph.add_edge({edge['from']!r}, {edge['to']!r})")

    for c_edge in payload.get("conditional_edges", []):
        mapping = c_edge["mapping"]
        cond = c_edge.get("condition", "custom_condition")
        lines.append(f"    graph.add_conditional_edge({c_edge['from']!r}, {cond}, {mapping!r})")

    entry_point = payload.get("entry_point")
    if entry_point:
        lines.append(f"    graph.set_entry_point({entry_point!r})")
    elif payload.get("nodes"):
        lines.append(f"    graph.set_entry_point({payload['nodes'][0]['id']!r})")

    lines.append("    return graph")
    lines.append("")
    lines.append("if __name__ == '__main__':")
    lines.append("    graph = create_graph()")
    lines.append("    compiled = graph.compile()")

    return "\n".join(lines)


def create_app() -> "FastAPI":
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, JSONResponse

    app = FastAPI(title="SynapseKit Graph Builder")

    @app.get("/")
    def get_ui():
        return HTMLResponse(content=HTML_TEMPLATE)

    @app.post("/api/mermaid")
    async def api_mermaid(request: Request):
        try:
            body = await request.body()
            graph = StateGraph.from_json(body.decode("utf-8"))
            if not graph._entry_point and graph._nodes:
                graph.set_entry_point(next(iter(graph._nodes.keys())))
            mm = get_mermaid(graph)
            return JSONResponse({"mermaid": mm})
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=400)

    @app.post("/api/codegen")
    async def api_codegen(request: Request):
        try:
            body = await request.body()
            payload = json.loads(body)
            code = generate_python_code(payload)
            return JSONResponse({"code": code})
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=400)

    @app.get("/api/schema")
    def api_schema():
        return JSONResponse(
            {
                "node_types": ["rag_node", "llm_node", "agent_node"],
                "edge_types": ["normal", "conditional"],
            }
        )

    return app


def run_graph_builder(args: argparse.Namespace) -> None:
    import uvicorn

    app = create_app()

    url = f"http://{args.host}:{args.port}"
    print(f"Starting SynapseKit Graph Builder at {url}")

    def open_browser():
        time.sleep(1.5)
        webbrowser.open(url)

    threading.Thread(target=open_browser, daemon=True).start()

    uvicorn.run(app, host=args.host, port=args.port)
