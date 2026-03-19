#!/usr/bin/env python3
"""Launch the live C. elegans simulation + WebSocket visualisation server.

Usage::

    python scripts/run_live.py [--steps N] [--seed S] [--port 8000]
    python scripts/run_live.py --steps 2000 --port 8000 --use-adex

Open http://localhost:8000 in your browser.
Space bar pauses/resumes. Scroll to zoom. Ctrl+C to stop.

Requirements
------------
Core:  pip install numpy pandas matplotlib networkx scipy
Live:  pip install fastapi uvicorn websockets   (optional)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import threading
import time
from pathlib import Path

import numpy as np


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))   
sys.path.insert(0, str(_PROJECT_ROOT))            


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live C. elegans simulation visualiser.")
    p.add_argument("--steps",    type=int,  default=2000)
    p.add_argument("--seed",     type=int,  default=42)
    p.add_argument("--port",     type=int,  default=8000)
    p.add_argument("--data-dir", type=str,  default=None)
    p.add_argument("--use-adex", action="store_true",
                   help="Use AdEx biophysical neuron model (default: LIF)")
    return p.parse_args()


def _build_components(args, cfg):
    """Construct and return all simulation objects."""
    from celegans.connectome import load_connectome
    from celegans.environment import WormEnv
    from celegans.gnn_model import ConnectomeGNN
    from celegans.utils.logging import get_logger

    logger = get_logger("run_live")
    graph = load_connectome(cfg.resolved_data_dir(_PROJECT_ROOT))

    sensory_idx     = graph.get_sensory_indices()
    motor_idx       = graph.get_motor_indices()
    interneuron_idx = graph.get_interneuron_indices()

    if args.use_adex:
        from celegans.adex_neurons import AdExNeuronBank
        neuron_bank = AdExNeuronBank(
            n_total=graph.data.num_nodes,
            sensory_indices=sensory_idx,
            interneuron_indices=interneuron_idx,
            motor_indices=motor_idx,
            dt=cfg.dt,
        )
        logger.info("Neuron model: AdEx (biophysical)")
    else:
        from celegans.lif_neurons import LIFNeuronBank
        neuron_bank = LIFNeuronBank(
            n_total=graph.data.num_nodes,
            sensory_indices=sensory_idx,
            interneuron_indices=interneuron_idx,
            motor_indices=motor_idx,
            dt=cfg.dt,
            threshold=cfg.threshold,
            reset_potential=cfg.reset_potential,
        )
        logger.info("Neuron model: LIF")

    model = ConnectomeGNN(
        input_dim=graph.data.x.shape[1],
        hidden_dim=cfg.gnn_hidden_dim,
        num_layers=cfg.gnn_num_layers,
        sensory_indices=sensory_idx,
        motor_indices=motor_idx,
    )

    n_motor = max(len(motor_idx), 1)
    env = WormEnv(
        n_neurons=graph.data.num_nodes,
        body_segments=cfg.body_segments,
        food_gradient_strength=cfg.food_gradient_strength,
        num_motor_neurons=n_motor,
        seed=args.seed,
    )
    return graph, neuron_bank, model, env, sensory_idx, motor_idx, n_motor


def _run_sim_loop(steps, seed, graph, neuron_bank, model, env,
                  sensory_idx, motor_idx, n_motor, broadcaster, push_fn=None):
    """
    Blocking simulation loop.
    push_fn(msg) is called after each step when provided (used to forward
    messages to the WebSocket event loop from a background thread).
    """
    from celegans.utils.logging import get_logger
    from celegans.utils.reproducibility import set_all_seeds

    logger = get_logger("sim_loop")
    set_all_seeds(seed)

    obs, _ = env.reset(seed=seed)
    neuron_bank.reset_state()
    model.eval()

    for step in range(steps):
        # --- sensory input from food gradient ---
        gradient   = obs["food_gradient"]
        n_sensory  = len(sensory_idx)
        s_input    = np.zeros(max(n_sensory, 1), dtype=np.float32)
        if n_sensory >= 2:
            s_input[0] = float(gradient[0])
            s_input[1] = float(gradient[1])

        # --- GNN → neurons → motor action ---
        gnn_out     = model(graph.data, s_input)
        spikes, mems = neuron_bank.step(gnn_out.ravel())
        motor_rates  = neuron_bank.get_recent_spike_rates(window=10)

        action = np.zeros(n_motor, dtype=np.float32)
        for j, mi in enumerate(motor_idx[:n_motor]):
            action[j] = float(np.clip(motor_rates[mi], -1.0, 1.0))

        obs, _, terminated, _, _ = env.step(action)
        body_pos  = obs["body_position"]
        food_pos  = (body_pos[0] + obs["food_gradient"] * 80.0).astype(np.float32)

        msg = broadcaster.push_step(
            spikes=spikes,
            membrane_potentials=mems,
            body_position=body_pos,
            food_position=food_pos,
            food_reached=bool(terminated),
        )

        if push_fn is not None:
            push_fn(msg)

        if terminated:
            logger.info("Food reached at step %d — resetting", step + 1)
            obs, _ = env.reset()
            neuron_bank.reset_state()

        time.sleep(0.01)   # 100 sim-steps/sec  ← smooth for browser


def main() -> None:
    args = _parse_args()

    import os
    if args.data_dir:
        os.environ["CELEGANS_DATA_DIR"] = args.data_dir

    from celegans.config import load_config
    from celegans.utils.logging import get_logger

    logger  = get_logger("run_live")
    cfg     = load_config()
    graph, neuron_bank, model, env, sensory_idx, motor_idx, n_motor = \
        _build_components(args, cfg)

    from server.websocket_server import SimulationBroadcaster
    broadcaster = SimulationBroadcaster(graph.node_names)

    # ── Mode 1: FastAPI + real WebSocket ──────────────────────────────────
    try:
        import uvicorn
        from server.websocket_server import create_app

        app  = create_app(broadcaster)
        _loop: "asyncio.AbstractEventLoop | None" = None
        _loop_ready = threading.Event()

        async def _capture_loop():
            nonlocal _loop
            _loop = asyncio.get_running_loop()
            _loop_ready.set()

        app.add_event_handler("startup", _capture_loop)

        def _push(msg):
            """Thread-safe broadcast into the FastAPI event loop."""
            if _loop and _loop.is_running():
                asyncio.run_coroutine_threadsafe(app.broadcast_step(msg), _loop)

        def _sim_thread_fn():
            _loop_ready.wait(timeout=15.0)
            _run_sim_loop(args.steps, args.seed, graph, neuron_bank, model, env,
                          sensory_idx, motor_idx, n_motor, broadcaster, _push)

        threading.Thread(target=_sim_thread_fn, daemon=True).start()

        print(f"\n🔬  C. elegans Live Visualiser")
        print(f"    Neurons  : {graph.data.num_nodes}")
        print(f"    Synapses : {graph.data.num_edges}")
        print(f"    Model    : {'AdEx (biophysical)' if args.use_adex else 'LIF'}")
        print(f"\n    Browser  → http://localhost:{args.port}")
        print(f"    Space = pause/resume  |  Scroll = zoom  |  Ctrl+C = stop\n")

        uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")

    except ImportError:
        # ── Mode 2: stdlib polling fallback ──────────────────────────────
        logger.warning(
            "FastAPI/uvicorn not installed — running in polling mode.\n"
            "Install the full server with:\n"
            "    pip install fastapi uvicorn websockets\n"
        )
        from server.websocket_server import _FallbackServer

        threading.Thread(
            target=_FallbackServer(broadcaster, port=args.port).run,
            daemon=True,
        ).start()

        print(f"\n🔬  C. elegans Live (polling mode)")
        print(f"    State endpoint → http://localhost:{args.port}/state")
        print(f"    Ctrl+C to stop\n")

        _run_sim_loop(args.steps, args.seed, graph, neuron_bank, model, env,
                      sensory_idx, motor_idx, n_motor, broadcaster, push_fn=None)


if __name__ == "__main__":
    main()
