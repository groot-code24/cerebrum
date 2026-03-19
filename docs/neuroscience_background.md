# Neuroscience Background

## C. elegans as a Model Organism

*Caenorhabditis elegans* is a 1 mm soil nematode that has been central to
neuroscience since Sydney Brenner chose it as a genetic model organism in the
1960s. Its nervous system consists of exactly 302 neurons in hermaphrodites,
connected by approximately 7,000 chemical synapses and 600 gap junctions. The
complete wiring diagram — the *connectome* — was mapped by White et al. (1986)
over more than a decade of electron microscopy serial section reconstruction,
and subsequently refined by Varshney et al. (2011) and the OpenWorm project.

## The Connectome as a Graph

The connectome can be represented as a weighted directed graph where nodes are
neurons and edges are synapses. Chemical synapses carry the edge weight (synapse
count); gap junctions are bidirectional and carry electrical coupling. This
graph structure is not random: sensory neurons project to interneurons, which
project to motor neurons, with extensive recurrent connectivity. The graph has
a small-world topology (high clustering, short path lengths) that is thought to
be computationally efficient.

## Leaky Integrate-and-Fire Neurons

Each neuron is modelled using the Leaky Integrate-and-Fire (LIF) formalism.
The membrane potential V(t) evolves as:

    τ_m · dV/dt = -V(t) + I(t)

where τ_m is the membrane time constant and I(t) is the synaptic input current.
When V(t) reaches a threshold θ, a spike is emitted and V is reset to V_reset.
This model captures the essential integrate-and-fire dynamics without the
computational cost of full Hodgkin-Huxley biophysics.

## Graph Neural Networks for Connectome Emulation

A Graph Neural Network (GNN) operating on the connectome graph learns to
propagate activation signals in a way that respects the synaptic weight
distribution. Unlike hand-coded rule-based models, the GNN can learn
non-obvious signal pathways from data. With random initialisation, the
propagation already reflects connectivity priors; fine-tuning against
recorded neural activity data can improve biological fidelity further.

## Emergent Locomotion and Chemotaxis

The remarkable claim of connectome emulation is that the complex coordinated
behaviour of C. elegans — sinusoidal undulation for locomotion, chemotaxis
toward food, avoidance of noxious stimuli — can emerge from the graph structure
alone, without any explicit behaviour rules. Key circuits include:

- **Locomotion**: AVB/AVA command interneurons drive alternating dorsal (DB/DA)
  and ventral (VB/VA) motor neuron activity, producing undulation.
- **Chemotaxis**: AWC/ASE sensory neurons detect chemical gradients; AIY/AIZ
  interneurons integrate this signal; AVB biases forward locomotion.
- **Tap withdrawal reflex**: PLM/ALM mechanosensory neurons activate AVD/PVC,
  which drive reversal (AVA) or forward escape (AVB) depending on context.

## References

- White, J.G., Southgate, E., Thomson, J.N., & Brenner, S. (1986). The
  structure of the nervous system of the nematode *Caenorhabditis elegans*.
  *Philosophical Transactions of the Royal Society B*, 314(1165), 1–340.

- Varshney, L.R., Chen, B.L., Paniagua, E., Hall, D.H., & Chklovskii, D.B.
  (2011). Structural properties of the *Caenorhabditis elegans* neuronal
  network. *PLOS Computational Biology*, 7(2), e1001066.

- Bhattacharya, S., Bhattacharya, A., et al. (2019). Distinct neural circuits
  control rhythm inhibition and spitting by the myogenic pharynx of
  *C. elegans*. *Scientific Reports*, 9, 9095.

- Yon Artificial Intelligence Lab (2025). A digital simulation of the complete
  *Drosophila melanogaster* connectome with emergent locomotion.
  *bioRxiv* preprint.

- OpenWorm Project (2024). *c302: A framework for computational modelling of
  *C. elegans* neural circuits*. https://github.com/openworm/c302
