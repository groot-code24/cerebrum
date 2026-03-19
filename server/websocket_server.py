"""Real-time WebSocket server for live C. elegans simulation visualisation.

Streams per-timestep simulation state to a browser frontend (D3.js force
graph) via WebSocket.  The browser shows neurons lighting up as spikes
propagate through the connectome in real time.

Architecture
-----------
FastAPI + Starlette WebSocket endpoint.  Falls back to a stdlib http.server
if FastAPI/uvicorn are unavailable (limited functionality).

Usage
-----
::

    # With FastAPI installed:
    python scripts/run_live.py

    # Browser:
    http://localhost:8000

Requirements (optional)
-----------------------
    pip install fastapi uvicorn websockets

If not installed, the server will print an install instruction and exit.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np

from celegans.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# State broadcaster — framework-agnostic
# ---------------------------------------------------------------------------

class SimulationBroadcaster:
    """Holds the live simulation state and broadcasts it to connected clients.

    Can be used standalone (for testing) or wired into a WebSocket handler.

    Parameters
    ----------
    node_names : List[str]
        Names of all neurons — sent to clients on connect for labelling.
    max_history : int
        Maximum timesteps buffered for replaying to late-joining clients.
    """

    def __init__(
        self,
        node_names: List[str],
        max_history: int = 200,
    ) -> None:
        self.node_names = node_names
        self.n_neurons = len(node_names)
        self.max_history = max_history

        self._history: List[Dict[str, Any]] = []
        self._clients: Set = set()   # websocket objects
        self._step: int = 0
        self._food_pos: Optional[List[float]] = None
        self._worm_pos: Optional[List[List[float]]] = None

    def push_step(
        self,
        spikes: np.ndarray,
        membrane_potentials: np.ndarray,
        body_position: Optional[np.ndarray] = None,
        food_position: Optional[np.ndarray] = None,
        food_reached: bool = False,
    ) -> Dict[str, Any]:
        """Package one simulation timestep as a JSON-serializable dict."""
        msg: Dict[str, Any] = {
            "type": "step",
            "step": self._step,
            "ts": time.time(),
            # Top-50 most active neurons (sparse encoding)
            "spikes": _encode_sparse(spikes, top_k=50),
            "membrane": _encode_float16(membrane_potentials, top_k=50),
            "food_reached": bool(food_reached),
        }

        if body_position is not None:
            msg["body"] = body_position.tolist() if hasattr(body_position, "tolist") else body_position
        if food_position is not None:
            msg["food"] = food_position.tolist() if hasattr(food_position, "tolist") else food_position

        # Keep rolling history
        self._history.append(msg)
        if len(self._history) > self.max_history:
            self._history.pop(0)

        self._step += 1
        return msg

    def get_init_message(self) -> Dict[str, Any]:
        """Message sent to newly connected clients."""
        return {
            "type": "init",
            "node_names": self.node_names,
            "n_neurons": self.n_neurons,
            "history": self._history[-50:],   # send last 50 steps as warm-up
        }

    def reset(self) -> None:
        self._history = []
        self._step = 0


def _encode_sparse(arr: np.ndarray, top_k: int = 50) -> Dict[str, Any]:
    """Encode array as {indices: [...], values: [...]} keeping top_k entries."""
    a = np.asarray(arr).ravel()
    if len(a) == 0:
        return {"indices": [], "values": []}
    k = min(top_k, len(a))
    idx = np.argpartition(np.abs(a), -k)[-k:]
    idx = idx[np.argsort(np.abs(a[idx]))[::-1]]
    return {
        "indices": idx.tolist(),
        "values": [round(float(a[i]), 4) for i in idx],
    }


def _encode_float16(arr: np.ndarray, top_k: int = 50) -> Dict[str, Any]:
    """Encode membrane potentials for top_k most active neurons."""
    return _encode_sparse(arr, top_k=top_k)


# ---------------------------------------------------------------------------
# FastAPI WebSocket server
# ---------------------------------------------------------------------------

def create_app(broadcaster: SimulationBroadcaster):
    """Create a FastAPI app wired to the broadcaster.

    Parameters
    ----------
    broadcaster : SimulationBroadcaster
        Shared state object updated by the simulation loop.

    Returns
    -------
    FastAPI app instance.
    """
    try:
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
        from fastapi.responses import HTMLResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError:
        raise ImportError(
            "FastAPI is required for the live visualisation server.\n"
            "Install with:  pip install fastapi uvicorn websockets"
        )

    app = FastAPI(title="C. elegans Live Visualiser")

    # Serve static files (HTML/JS)
    static_dir = Path(__file__).resolve().parent.parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Root → serve the visualiser HTML
    @app.get("/", response_class=HTMLResponse)
    async def index():
        html_path = static_dir / "index.html"
        if html_path.exists():
            return html_path.read_text()
        return "<h1>C. elegans Live</h1><p>Place index.html in server/static/</p>"

    # Info endpoint
    @app.get("/info")
    async def info():
        return {
            "n_neurons": broadcaster.n_neurons,
            "steps_run": broadcaster._step,
            "connected_clients": len(broadcaster._clients),
        }

    # WebSocket endpoint
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        broadcaster._clients.add(websocket)
        logger.info("Client connected. Total: %d", len(broadcaster._clients))
        try:
            # Send init message with node names + recent history
            await websocket.send_text(json.dumps(broadcaster.get_init_message()))
            # Keep alive — simulation pushes via broadcaster
            while True:
                # Ping/pong keepalive
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                if data == "ping":
                    await websocket.send_text('{"type":"pong"}')
        except (WebSocketDisconnect, asyncio.TimeoutError):
            pass
        finally:
            broadcaster._clients.discard(websocket)
            logger.info("Client disconnected. Total: %d", len(broadcaster._clients))

    # Push endpoint: simulation loop calls this to broadcast steps
    async def broadcast_step(msg: Dict[str, Any]) -> None:
        """Broadcast a simulation step to all connected clients."""
        if not broadcaster._clients:
            return
        payload = json.dumps(msg)
        dead = set()
        for ws in list(broadcaster._clients):
            try:
                await ws.send_text(payload)
            except Exception:
                dead.add(ws)
        broadcaster._clients -= dead

    app.broadcast_step = broadcast_step
    return app


# ---------------------------------------------------------------------------
# Fallback: stdlib HTTP server (no WebSocket, polling only)
# ---------------------------------------------------------------------------

class _FallbackServer:
    """Minimal HTTP server that serves the visualiser and a /state endpoint.

    Used when fastapi/uvicorn are not installed.
    """

    def __init__(self, broadcaster: SimulationBroadcaster, port: int = 8000) -> None:
        self.broadcaster = broadcaster
        self.port = port

    def run(self) -> None:
        import http.server
        import urllib.parse

        broadcaster = self.broadcaster

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/state":
                    body = json.dumps(broadcaster.get_init_message()).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    html_path = Path(__file__).resolve().parent / "static" / "index.html"
                    if html_path.exists():
                        body = html_path.read_bytes()
                        self.send_response(200)
                        self.send_header("Content-Type", "text/html")
                        self.end_headers()
                        self.wfile.write(body)
                    else:
                        self.send_response(404)
                        self.end_headers()

            def log_message(self, format, *args):
                pass  # silence default logging

        server = http.server.HTTPServer(("0.0.0.0", self.port), Handler)
        logger.info("Fallback HTTP server on http://localhost:%d/state", self.port)
        server.serve_forever()
