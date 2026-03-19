# Architecture Overview

## Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SIMULATION LOOP                              │
│                                                                     │
│  ┌──────────┐    sensory    ┌──────────────┐    activations         │
│  │ WormEnv  │──────input───▶│ ConnectomeGNN│──────────────────┐    │
│  │          │               │  (PyG GNN)   │                  │    │
│  │ - body   │               │  SAGEConv ×N │                  ▼    │
│  │ - food   │               └──────────────┘         ┌──────────────┐│
│  │ - physics│                                         │LIFNeuronBank ││
│  └────▲─────┘                                         │ (snnTorch)   ││
│       │                                               │              ││
│       │  motor_spike_rates                            │ spikes       ││
│       └───────────────────────────────────────────────┘              ││
│                                                                      ││
└──────────────────────────────────────────────────────────────────────┘│
                                                                        │
                    Recorded every timestep:                            │
                    - spike_history [T, 302]                            │
                    - trajectory [T, 2]                                 │
                    - membrane_potentials [T, 302]                      │
```

## Component Descriptions

### ConnectomeGraph (`connectome.py`)
Loads the C. elegans wiring diagram from OpenWorm CSV files into a
`torch_geometric.data.Data` object. Neurons are nodes; synapses are weighted
directed edges. Provides immutable ablation operations that return new graphs
without modifying the original.

### ConnectomeGNN (`gnn_model.py`)
A multi-layer Graph Neural Network that propagates activation signals through
the connectome graph. Sensory neuron inputs are injected at the input layer;
motor neuron activations are extracted from the output. Supports both
GraphSAGE (default) and Graph Attention Network (GAT) convolutions.

### LIFNeuronBank (`lif_neurons.py`)
Wraps snnTorch's Leaky Integrate-and-Fire neurons for all 302 cells, with
separate time constants per neuron type (sensory, interneuron, motor). Records
full spike history for post-hoc analysis.

### WormEnv (`environment.py`)
A Gymnasium-compatible 2D physics environment. The worm body is a spring-mass
chain integrated with the Verlet method. Food emits a Gaussian concentration
gradient; sensory neurons (ASE class) receive input proportional to local
gradient magnitude. Motor neurons drive dorsal/ventral muscle contractions
that produce sinusoidal undulation.

### SimulationRunner (`simulation.py`)
Orchestrates the closed-loop feedback cycle: reads sensory input from WormEnv,
runs a GNN pass, steps the LIF bank, extracts motor spike rates, and passes
them as actions back to WormEnv. Saves episode summaries and trajectories.

### AblationExperiment (`ablation.py`)
Systematically silences neurons or synaptic connections and compares resulting
locomotion and chemotaxis scores against an unmodified baseline.

## Data Flow

```
CSV files (OpenWorm)
      │
      ▼
ConnectomeGraph (PyG Data)
      │
      ├──▶ ConnectomeGNN (weights: Xavier init or trained)
      │           │
      │           ▼
      └──▶ LIFNeuronBank ──▶ spike_history
                  │
                  ▼
            motor_rates
                  │
                  ▼
            WormEnv.step()
                  │
                  ├──▶ observation (sensory input for next GNN pass)
                  ├──▶ reward
                  └──▶ trajectory
```
