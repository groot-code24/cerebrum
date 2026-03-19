"""Adaptive Exponential Integrate-and-Fire (AdEx) neurons.

Replaces the simple LIF model with a biophysically richer model that captures:
  - Sub-threshold resonance
  - Spike-frequency adaptation
  - Graded (non-spiking) potentials — dominant in C. elegans
  - Bursting and regular spiking regimes

Model equations (Brette & Gerstner 2005; Bhattacharya et al. 2019):
  C dV/dt = -g_L(V-E_L) + g_L*Δ_T*exp((V-V_T)/Δ_T) - w + I_syn
  τ_w dw/dt = a(V-E_L) - w

Spike condition: V >= V_peak  →  V ← V_r,  w ← w + b

C. elegans specifics
--------------------
Most somatic neurons show *graded* potentials rather than action potentials.
Set ``spiking=False`` to use the sub-threshold dynamics only (Δ_T term
dropped, no reset), which models graded transmission faithfully.

References
----------
Brette R, Gerstner W (2005). J Neurophysiol 94:3637–3642.
Bhattacharya S et al. (2019). PLoS Comput Biol 15(9): e1007279.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from celegans.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Default parameter sets derived from Bhattacharya et al. 2019 Table 1
# ---------------------------------------------------------------------------

#: Canonical AdEx parameters for C. elegans neuron types.
ADEX_PARAMS: dict = {
    "sensory": dict(
        C=200e-12,      # F  – membrane capacitance
        g_L=10e-9,      # S  – leak conductance
        E_L=-70e-3,     # V  – leak reversal potential
        V_T=-50e-3,     # V  – spike threshold
        delta_T=2e-3,   # V  – slope factor
        a=2e-9,         # S  – sub-threshold adaptation
        tau_w=30e-3,    # s  – adaptation time constant
        b=20e-12,       # A  – spike-triggered adaptation
        V_r=-58e-3,     # V  – reset potential
        V_peak=20e-3,   # V  – spike detection threshold
        spiking=True,
    ),
    "interneuron": dict(
        C=200e-12,
        g_L=10e-9,
        E_L=-65e-3,
        V_T=-52e-3,
        delta_T=2e-3,
        a=0.0,          # no sub-threshold adaptation — tonically active
        tau_w=100e-3,
        b=10e-12,
        V_r=-60e-3,
        V_peak=20e-3,
        spiking=False,  # graded potentials
    ),
    "motor": dict(
        C=200e-12,
        g_L=12e-9,
        E_L=-60e-3,
        V_T=-48e-3,
        delta_T=3e-3,
        a=4e-9,
        tau_w=80e-3,
        b=30e-12,
        V_r=-55e-3,
        V_peak=20e-3,
        spiking=True,
    ),
    "pharyngeal": dict(
        C=100e-12,
        g_L=5e-9,
        E_L=-75e-3,
        V_T=-55e-3,
        delta_T=1.5e-3,
        a=0.5e-9,
        tau_w=50e-3,
        b=5e-12,
        V_r=-65e-3,
        V_peak=20e-3,
        spiking=False,
    ),
}


class AdExLayer:
    """Vectorised AdEx neuron population.

    All voltages are in millivolts internally (converted on input/output).

    Parameters
    ----------
    num_neurons:
        Population size.
    neuron_type:
        One of ``"sensory"``, ``"interneuron"``, ``"motor"``, ``"pharyngeal"``.
        Used to select default biophysical parameters.
    dt:
        Simulation timestep in milliseconds.
    spiking:
        If ``False``, spike mechanism is disabled and the neuron operates in
        graded-potential mode (appropriate for most C. elegans neurons).
    **kwargs:
        Override any parameter from ``ADEX_PARAMS[neuron_type]``.
    """

    def __init__(
        self,
        num_neurons: int,
        neuron_type: str = "interneuron",
        dt: float = 0.1,
        spiking: Optional[bool] = None,
        **kwargs,
    ) -> None:
        if neuron_type not in ADEX_PARAMS:
            raise ValueError(
                f"neuron_type must be one of {list(ADEX_PARAMS)}, got {neuron_type!r}"
            )
        if num_neurons < 1:
            raise ValueError(f"num_neurons must be >= 1, got {num_neurons}")

        p = {**ADEX_PARAMS[neuron_type], **kwargs}
        self.num_neurons = num_neurons
        self.neuron_type = neuron_type
        self.dt_s = dt * 1e-3  # convert ms → s for SI equations

        # Biophysical params (all SI)
        self.C = float(p["C"])
        self.g_L = float(p["g_L"])
        self.E_L = float(p["E_L"])
        self.V_T = float(p["V_T"])
        self.delta_T = float(p["delta_T"])
        self.a = float(p["a"])
        self.tau_w = float(p["tau_w"])
        self.b = float(p["b"])
        self.V_r = float(p["V_r"])
        self.V_peak = float(p["V_peak"])
        self.spiking = spiking if spiking is not None else bool(p.get("spiking", True))

        # State arrays (SI)
        self.V = np.full(num_neurons, self.E_L, dtype=np.float64)
        self.w = np.zeros(num_neurons, dtype=np.float64)

        logger.debug(
            "AdExLayer(%s, n=%d, spiking=%s, dt=%.3fms)",
            neuron_type, num_neurons, self.spiking, dt,
        )

    # ------------------------------------------------------------------
    # Core dynamics
    # ------------------------------------------------------------------

    def step(
        self, I_syn: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Advance one timestep using Euler integration.

        Parameters
        ----------
        I_syn:
            Synaptic input current in picoamperes (pA).  Shape ``[N]``.

        Returns
        -------
        spikes : np.ndarray[float32, N]
            Binary spike array (0 or 1).
        V_mV : np.ndarray[float32, N]
            Membrane potential in millivolts.
        """
        I = np.asarray(I_syn, dtype=np.float64).ravel()
        if len(I) < self.num_neurons:
            I = np.pad(I, (0, self.num_neurons - len(I)))
        else:
            I = I[:self.num_neurons]

        # Convert pA → A
        I_A = I * 1e-12

        dt = self.dt_s
        V = self.V
        w = self.w

        # Exponential term (only for spiking neurons to avoid numerical blowup)
        if self.spiking and self.delta_T > 0:
            exp_arg = np.clip((V - self.V_T) / self.delta_T, -20.0, 20.0)
            exp_term = self.g_L * self.delta_T * np.exp(exp_arg)
        else:
            exp_term = np.zeros(self.num_neurons, dtype=np.float64)

        # Voltage update
        dV = (
            -self.g_L * (V - self.E_L)
            + exp_term
            - w
            + I_A
        ) / self.C
        V_new = V + dt * dV

        # Adaptation update
        dw = (self.a * (V - self.E_L) - w) / self.tau_w
        w_new = w + dt * dw

        # Spike detection and reset
        if self.spiking:
            spikes = (V_new >= self.V_peak).astype(np.float32)
            fired = spikes.astype(bool)
            V_new[fired] = self.V_r
            w_new[fired] += self.b
        else:
            # Graded: normalise membrane potential to [0, 1] range
            spikes = np.zeros(self.num_neurons, dtype=np.float32)

        # Guard against numerical blow-up
        V_new = np.clip(V_new, self.E_L - 0.1, self.V_peak + 0.01)
        w_new = np.clip(w_new, -1e-9, 1e-9)

        self.V = V_new
        self.w = w_new

        return spikes, (V_new * 1e3).astype(np.float32)  # return in mV

    def reset_state(self) -> None:
        """Reset V to resting potential and w to zero."""
        self.V = np.full(self.num_neurons, self.E_L, dtype=np.float64)
        self.w = np.zeros(self.num_neurons, dtype=np.float64)

    def get_graded_output(self) -> np.ndarray:
        """Return normalised graded potential in [0, 1] for non-spiking neurons.

        Maps ``[E_L, V_T]`` → ``[0, 1]`` with sigmoid saturation.
        """
        v_norm = (self.V - self.E_L) / max(self.V_T - self.E_L, 1e-9)
        return (1.0 / (1.0 + np.exp(-5.0 * (v_norm - 0.5)))).astype(np.float32)


class AdExNeuronBank:
    """Full 302-neuron AdEx bank with per-group parameters.

    Drop-in replacement for :class:`~celegans.lif_neurons.LIFNeuronBank`
    with the same public interface plus additional graded-output API.

    Parameters
    ----------
    n_total:
        Total neuron count (default 302 for full connectome).
    sensory_indices / interneuron_indices / motor_indices:
        Index lists for each functional group.
    dt:
        Simulation timestep in ms.
    """

    def __init__(
        self,
        n_total: int = 302,
        sensory_indices: Optional[List[int]] = None,
        interneuron_indices: Optional[List[int]] = None,
        motor_indices: Optional[List[int]] = None,
        dt: float = 0.1,
        **adex_kwargs,
    ) -> None:
        self.n_total = n_total
        self.sensory_indices: List[int] = sensory_indices or []
        self.interneuron_indices: List[int] = interneuron_indices or []
        self.motor_indices: List[int] = motor_indices or []

        n_s = max(len(self.sensory_indices), 1)
        n_i = max(len(self.interneuron_indices), 1)
        n_m = max(len(self.motor_indices), 1)

        self.sensory_layer = AdExLayer(n_s, "sensory", dt=dt, **adex_kwargs)
        self.interneuron_layer = AdExLayer(n_i, "interneuron", dt=dt, **adex_kwargs)
        self.motor_layer = AdExLayer(n_m, "motor", dt=dt, **adex_kwargs)

        self._spike_history: List[np.ndarray] = []
        self._voltage_history: List[np.ndarray] = []

        logger.info(
            "AdExNeuronBank: n=%d  sensory=%d  interneuron=%d  motor=%d  dt=%.2fms",
            n_total, len(self.sensory_indices),
            len(self.interneuron_indices), len(self.motor_indices), dt,
        )

    # ------------------------------------------------------------------

    def step(
        self, gnn_activations: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Advance all groups one timestep.

        Parameters
        ----------
        gnn_activations:
            Per-neuron input current in pA, shape ``[n_total]`` or ``[n_total, 1]``.

        Returns
        -------
        spikes : np.ndarray[float32, n_total]
        membrane_potentials_mV : np.ndarray[float32, n_total]
        """
        act = np.asarray(gnn_activations, dtype=np.float64).ravel()
        if len(act) > self.n_total:
            act = act[:self.n_total]
        elif len(act) < self.n_total:
            act = np.pad(act, (0, self.n_total - len(act)))

        spikes = np.zeros(self.n_total, dtype=np.float32)
        mems = np.zeros(self.n_total, dtype=np.float32)

        for indices, layer in [
            (self.sensory_indices, self.sensory_layer),
            (self.interneuron_indices, self.interneuron_layer),
            (self.motor_indices, self.motor_layer),
        ]:
            if not indices:
                continue
            idx = np.array(indices, dtype=np.int64)
            group_act = act[idx]
            # Scale GNN output (dimensionless ~[0,1]) to pA range
            I_pA = group_act * 200.0  # 200 pA max current
            spk, mem = layer.step(I_pA)
            n_g = min(len(idx), len(spk))
            spikes[idx[:n_g]] = spk[:n_g]
            mems[idx[:n_g]] = mem[:n_g]

        self._spike_history.append(spikes.copy())
        self._voltage_history.append(mems.copy())
        return spikes, mems

    def reset_state(self) -> None:
        self._spike_history = []
        self._voltage_history = []
        self.sensory_layer.reset_state()
        self.interneuron_layer.reset_state()
        self.motor_layer.reset_state()

    def get_spike_history(self) -> np.ndarray:
        """Return ``[T, n_total]`` spike array."""
        if not self._spike_history:
            return np.zeros((0, self.n_total), dtype=np.float32)
        return np.stack(self._spike_history, axis=0)

    def get_voltage_history(self) -> np.ndarray:
        """Return ``[T, n_total]`` membrane potential array (mV)."""
        if not self._voltage_history:
            return np.zeros((0, self.n_total), dtype=np.float32)
        return np.stack(self._voltage_history, axis=0)

    def get_recent_spike_rates(self, window: int = 10) -> np.ndarray:
        """Mean spike rate over the last *window* steps. Shape ``[n_total]``."""
        hist = self.get_spike_history()
        if hist.shape[0] == 0:
            return np.zeros(self.n_total, dtype=np.float32)
        return hist[-window:].mean(axis=0)

    def get_graded_outputs(self) -> np.ndarray:
        """Return graded (non-spiking) membrane outputs for all neurons, ``[n_total]``."""
        out = np.zeros(self.n_total, dtype=np.float32)
        for indices, layer in [
            (self.sensory_indices, self.sensory_layer),
            (self.interneuron_indices, self.interneuron_layer),
            (self.motor_indices, self.motor_layer),
        ]:
            if not indices:
                continue
            idx = np.array(indices, dtype=np.int64)
            graded = layer.get_graded_output()
            n_g = min(len(idx), len(graded))
            out[idx[:n_g]] = graded[:n_g]
        return out
