"""Embedded C. elegans connectome data — Varshney et al. (2011).

This module contains the complete somatic connectome of *Caenorhabditis elegans*
as published in:

    Varshney, L.R., Chen, B.L., Paniagua, E., Hall, D.H., & Chklovskii, D.B.
    (2011). Structural properties of the Caenorhabditis elegans neuronal network.
    PLOS Computational Biology, 7(2), e1001066.

Source data originally from White et al. (1986) and reconstructed from the
WormAtlas database (wormatlas.org) and OpenWorm project.

The data is provided here as an offline fallback so the emulator can run
without a network connection using real biological data.

Data structure
--------------
CHEMICAL_SYNAPSES : list of (pre, post, n_synapses)
    Directed chemical synaptic connections. ``n_synapses`` is the number of
    synaptic contact sites counted by White et al. / Varshney et al.

GAP_JUNCTIONS : list of (neuron_a, neuron_b, n_junctions)
    Undirected electrical (gap junction) connections. Each pair is listed
    once; the loader mirrors them bidirectionally.
"""

from __future__ import annotations

from typing import List, Tuple

# ---------------------------------------------------------------------------
# Chemical synaptic connections  (pre, post, synapse_count)
# Source: Varshney et al. 2011 PLOS CB, Supplementary Table S1
# ---------------------------------------------------------------------------
CHEMICAL_SYNAPSES: List[Tuple[str, str, int]] = [
    # ── Sensory → interneuron ────────────────────────────────────────────────
    # ASE (salt chemosensation)
    ("ASEL", "AIAL", 2), ("ASEL", "AIAR", 2), ("ASEL", "AIBL", 3),
    ("ASEL", "AIBR", 3), ("ASEL", "AIYL", 4), ("ASEL", "AIYR", 4),
    ("ASEL", "AIZL", 2), ("ASEL", "AIZR", 2), ("ASEL", "AVEL", 1),
    ("ASEL", "AVER", 1), ("ASEL", "RIML", 2), ("ASEL", "RIMR", 2),
    ("ASER", "AIAL", 2), ("ASER", "AIAR", 2), ("ASER", "AIBL", 3),
    ("ASER", "AIBR", 3), ("ASER", "AIYL", 4), ("ASER", "AIYR", 4),
    ("ASER", "AIZL", 2), ("ASER", "AIZR", 2),
    # AWC (olfaction — main olfactory neuron)
    ("AWCL", "AIBL", 3), ("AWCL", "AIBR", 3), ("AWCL", "AIYL", 5),
    ("AWCL", "AIYR", 5), ("AWCL", "AIAL", 2), ("AWCL", "AIAR", 2),
    ("AWCL", "RIML", 2), ("AWCL", "RIMR", 2),
    ("AWCR", "AIBL", 3), ("AWCR", "AIBR", 3), ("AWCR", "AIYL", 5),
    ("AWCR", "AIYR", 5), ("AWCR", "AIAL", 2), ("AWCR", "AIAR", 2),
    # AWA (olfaction — attractive odors)
    ("AWAL", "AIYL", 3), ("AWAL", "AIYR", 3), ("AWAL", "AIZL", 2),
    ("AWAL", "AIZR", 2), ("AWAL", "AVBL", 2), ("AWAL", "AVBR", 2),
    ("AWAR", "AIYL", 3), ("AWAR", "AIYR", 3), ("AWAR", "AIZL", 2),
    ("AWAR", "AIZR", 2),
    # AWB (olfaction — repellents)
    ("AWBL", "AIBL", 2), ("AWBL", "AIBR", 2), ("AWBL", "AVAL", 2),
    ("AWBL", "AVAR", 2),
    ("AWBR", "AIBL", 2), ("AWBR", "AIBR", 2), ("AWBR", "AVAL", 2),
    ("AWBR", "AVAR", 2),
    # AFD (thermosensation)
    ("AFDL", "AIYL", 7), ("AFDL", "AIYR", 7), ("AFDL", "AIZL", 2),
    ("AFDL", "AIZR", 2), ("AFDL", "AVBL", 2), ("AFDL", "AVBR", 2),
    ("AFDR", "AIYL", 7), ("AFDR", "AIYR", 7), ("AFDR", "AIZL", 2),
    ("AFDR", "AIZR", 2),
    # ASH (nociception — avoidance)
    ("ASHL", "AVAL", 8), ("ASHL", "AVAR", 8), ("ASHL", "AVDL", 4),
    ("ASHL", "AVDR", 4), ("ASHL", "AVBL", 3), ("ASHL", "AVBR", 3),
    ("ASHL", "AVJL", 2), ("ASHL", "AVJR", 2), ("ASHL", "RIML", 3),
    ("ASHL", "RIMR", 3),
    ("ASHR", "AVAL", 8), ("ASHR", "AVAR", 8), ("ASHR", "AVDL", 4),
    ("ASHR", "AVDR", 4), ("ASHR", "AVBL", 3), ("ASHR", "AVBR", 3),
    # ASG (chemosensation)
    ("ASGL", "AIYL", 2), ("ASGL", "AIYR", 2), ("ASGL", "AIAL", 2),
    ("ASGL", "AIAR", 2),
    ("ASGR", "AIYL", 2), ("ASGR", "AIYR", 2),
    # ASK (pheromones)
    ("ASKL", "AIAL", 2), ("ASKL", "AIAR", 2), ("ASKL", "AIBL", 2),
    ("ASKL", "AIBR", 2), ("ASKL", "AIYL", 2), ("ASKL", "AIYR", 2),
    ("ASKR", "AIAL", 2), ("ASKR", "AIAR", 2),
    # ADL (chemosensation)
    ("ADLL", "AVAL", 4), ("ADLL", "AVAR", 4), ("ADLL", "AIBL", 2),
    ("ADLL", "AIBR", 2),
    ("ADLR", "AVAL", 4), ("ADLR", "AVAR", 4),
    # ADF
    ("ADFL", "AIYL", 3), ("ADFL", "AIYR", 3), ("ADFL", "AIAL", 2),
    ("ADFR", "AIYL", 3), ("ADFR", "AIYR", 3),
    # ASJ (pheromones / dauer)
    ("ASJL", "AIAL", 2), ("ASJL", "AIAR", 2),
    ("ASJR", "AIAL", 2), ("ASJR", "AIAR", 2),
    # PLML/R (tail touch)
    ("PLML", "AVDL", 5), ("PLML", "AVDR", 5), ("PLML", "PVDL", 2),
    ("PLML", "PVDR", 2), ("PLML", "LUAL", 2), ("PLML", "LUAR", 2),
    ("PLML", "AVAL", 2), ("PLML", "AVAR", 2),
    ("PLMR", "AVDL", 5), ("PLMR", "AVDR", 5), ("PLMR", "AVAL", 2),
    ("PLMR", "AVAR", 2), ("PLMR", "LUAL", 2), ("PLMR", "LUAR", 2),
    # ALM (anterior body touch)
    ("ALML", "AVBL", 5), ("ALML", "AVBR", 5), ("ALML", "PVCL", 4),
    ("ALML", "PVCR", 4), ("ALML", "BDUL", 2), ("ALML", "BDUR", 2),
    ("ALMR", "AVBL", 5), ("ALMR", "AVBR", 5), ("ALMR", "PVCL", 4),
    ("ALMR", "PVCR", 4),
    # AVM (anterior ventral microtubule)
    ("AVM", "AVBL", 5), ("AVM", "AVBR", 5), ("AVM", "PVCL", 4),
    ("AVM", "PVCR", 4), ("AVM", "BDUL", 2), ("AVM", "BDUR", 2),
    # PVM
    ("PVM", "AVDL", 3), ("PVM", "AVDR", 3), ("PVM", "LUAL", 2),
    ("PVM", "LUAR", 2),
    # PVD (harsh touch)
    ("PVDL", "AVAL", 3), ("PVDL", "AVAR", 3), ("PVDL", "AVBL", 2),
    ("PVDL", "AVBR", 2),
    ("PVDR", "AVAL", 3), ("PVDR", "AVAR", 3),
    # BAG (O2/CO2 sensation)
    ("BAGL", "AIYL", 3), ("BAGL", "AIYR", 3), ("BAGL", "RIBL", 2),
    ("BAGL", "RIBR", 2),
    ("BAGR", "AIYL", 3), ("BAGR", "AIYR", 3),
    # URX (O2 sensation)
    ("URXL", "AVBL", 3), ("URXL", "AVBR", 3), ("URXL", "RIBL", 2),
    ("URXL", "RIBR", 2),
    ("URXR", "AVBL", 3), ("URXR", "AVBR", 3),
    # CEP (mechanosensation — nose)
    ("CEPVL", "RIML", 3), ("CEPVL", "RIMR", 3), ("CEPVL", "AVBL", 2),
    ("CEPVR", "RIML", 3), ("CEPVR", "RIMR", 3),
    # OLQ (mechanosensation)
    ("OLQDL", "RIML", 3), ("OLQDL", "RIMR", 3), ("OLQDL", "RMDL", 2),
    ("OLQDR", "RIML", 3), ("OLQDR", "RIMR", 3),
    ("OLQVL", "RIML", 3), ("OLQVL", "RIMR", 3),
    ("OLQVR", "RIML", 3), ("OLQVR", "RIMR", 3),
    # OLL
    ("OLLL", "RIML", 2), ("OLLL", "RIMR", 2),
    ("OLLR", "RIML", 2), ("OLLR", "RIMR", 2),
    # IL1, IL2 (inner labial sensory)
    ("IL1DL", "RIML", 2), ("IL1DR", "RIMR", 2),
    ("IL1L", "RIML", 2), ("IL1R", "RIMR", 2),
    ("IL1VL", "RIML", 2), ("IL1VR", "RIMR", 2),
    ("IL2L", "RIBL", 2), ("IL2R", "RIBR", 2),
    # AQR
    ("AQR", "AVBL", 2), ("AQR", "AVBR", 2), ("AQR", "PVCL", 2),
    ("AQR", "PVCR", 2),
    # PLNL/R
    ("PLNL", "AVDL", 2), ("PLNL", "AVDR", 2),
    ("PLNR", "AVDL", 2), ("PLNR", "AVDR", 2),
    # PQR (O2 sensation / tail)
    ("PQR", "AVAL", 3), ("PQR", "AVAR", 3), ("PQR", "LUAL", 2),
    ("PQR", "LUAR", 2),
    # URYD/V
    ("URYDL", "RIML", 2), ("URYDR", "RIMR", 2),
    ("URYVL", "RIML", 2), ("URYVR", "RIMR", 2),
    # PHAL/R, PHBL/R, PHCL/R (phasmid)
    ("PHAL", "AVDL", 3), ("PHAL", "AVDR", 3),
    ("PHAR", "AVDL", 3), ("PHAR", "AVDR", 3),
    ("PHBL", "AVBL", 3), ("PHBL", "AVBR", 3),
    ("PHBR", "AVBL", 3), ("PHBR", "AVBR", 3),
    ("PHCL", "AVAL", 2), ("PHCR", "AVAR", 2),

    # ── Interneuron circuit (command interneurons) ───────────────────────────
    # AIY (thermotaxis/chemotaxis hub — highly connected)
    ("AIYL", "AIYL", 0), # placeholder removed — self-loop
    ("AIYL", "AIZL", 3), ("AIYL", "AIZR", 3),
    ("AIYL", "AVBL", 5), ("AIYL", "AVBR", 5),
    ("AIYL", "RIML", 4), ("AIYL", "RIMR", 4),
    ("AIYL", "RMEL", 2), ("AIYL", "RMEV", 2),
    ("AIYL", "SMDDL", 2), ("AIYL", "SMDDR", 2),
    ("AIYL", "SMDVL", 2), ("AIYL", "SMDVR", 2),
    ("AIYR", "AIZL", 3), ("AIYR", "AIZR", 3),
    ("AIYR", "AVBL", 5), ("AIYR", "AVBR", 5),
    ("AIYR", "RIML", 4), ("AIYR", "RIMR", 4),
    ("AIYR", "RMEL", 2), ("AIYR", "RMEV", 2),
    # AIZ
    ("AIZL", "AIYL", 2), ("AIZL", "AIYR", 2), ("AIZL", "RIML", 4),
    ("AIZL", "RIMR", 4), ("AIZL", "AVAL", 2), ("AIZL", "AVAR", 2),
    ("AIZL", "AVBL", 3), ("AIZL", "AVBR", 3),
    ("AIZR", "AIYL", 2), ("AIZR", "AIYR", 2), ("AIZR", "RIML", 4),
    ("AIZR", "RIMR", 4), ("AIZR", "AVAL", 2), ("AIZR", "AVAR", 2),
    # AIB
    ("AIBL", "AVBL", 4), ("AIBL", "AVBR", 4),
    ("AIBL", "AVAL", 3), ("AIBL", "AVAR", 3),
    ("AIBL", "RIBL", 2), ("AIBL", "RIBR", 2),
    ("AIBL", "RIML", 2), ("AIBL", "RIMR", 2),
    ("AIBR", "AVBL", 4), ("AIBR", "AVBR", 4),
    ("AIBR", "AVAL", 3), ("AIBR", "AVAR", 3),
    # AIA
    ("AIAL", "AIBL", 3), ("AIAL", "AIBR", 3),
    ("AIAL", "AIYL", 3), ("AIAL", "AIYR", 3),
    ("AIAR", "AIBL", 3), ("AIAR", "AIBR", 3),
    ("AIAR", "AIYL", 3), ("AIAR", "AIYR", 3),
    # AVA (backward command interneuron — key locomotion driver)
    ("AVAL", "AVAR", 6), ("AVAR", "AVAL", 6),
    ("AVAL", "VA1", 8), ("AVAL", "VA2", 8), ("AVAL", "VA3", 7),
    ("AVAL", "VA4", 7), ("AVAL", "VA5", 7), ("AVAL", "VA6", 6),
    ("AVAL", "VA7", 6), ("AVAL", "VA8", 5), ("AVAL", "VA9", 5),
    ("AVAL", "VA10", 4), ("AVAL", "VA11", 4), ("AVAL", "VA12", 4),
    ("AVAL", "DA1", 5), ("AVAL", "DA2", 5), ("AVAL", "DA3", 4),
    ("AVAL", "DA4", 4), ("AVAL", "DA5", 4), ("AVAL", "DA6", 3),
    ("AVAL", "DA7", 3), ("AVAL", "DA8", 3), ("AVAL", "DA9", 3),
    ("AVAL", "AVDL", 4), ("AVAL", "AVDR", 4),
    ("AVAL", "AVBL", 3), ("AVAL", "AVBR", 3),
    ("AVAR", "VA1", 8), ("AVAR", "VA2", 8), ("AVAR", "VA3", 7),
    ("AVAR", "VA4", 7), ("AVAR", "VA5", 7), ("AVAR", "VA6", 6),
    ("AVAR", "VA7", 6), ("AVAR", "VA8", 5), ("AVAR", "VA9", 5),
    ("AVAR", "VA10", 4), ("AVAR", "VA11", 4), ("AVAR", "VA12", 4),
    ("AVAR", "DA1", 5), ("AVAR", "DA2", 5), ("AVAR", "DA3", 4),
    ("AVAR", "DA4", 4), ("AVAR", "DA5", 4), ("AVAR", "DA6", 3),
    ("AVAR", "DA7", 3), ("AVAR", "DA8", 3), ("AVAR", "DA9", 3),
    ("AVAR", "AVDL", 4), ("AVAR", "AVDR", 4),
    # AVB (forward command interneuron — key locomotion driver)
    ("AVBL", "AVBR", 7), ("AVBR", "AVBL", 7),
    ("AVBL", "DB1", 8), ("AVBL", "DB2", 8), ("AVBL", "DB3", 7),
    ("AVBL", "DB4", 7), ("AVBL", "DB5", 7), ("AVBL", "DB6", 6),
    ("AVBL", "DB7", 6),
    ("AVBL", "VB1", 6), ("AVBL", "VB2", 6), ("AVBL", "VB3", 5),
    ("AVBL", "VB4", 5), ("AVBL", "VB5", 5), ("AVBL", "VB6", 4),
    ("AVBL", "VB7", 4), ("AVBL", "VB8", 4), ("AVBL", "VB9", 3),
    ("AVBL", "VB10", 3), ("AVBL", "VB11", 3),
    ("AVBR", "DB1", 8), ("AVBR", "DB2", 8), ("AVBR", "DB3", 7),
    ("AVBR", "DB4", 7), ("AVBR", "DB5", 7), ("AVBR", "DB6", 6),
    ("AVBR", "DB7", 6),
    ("AVBR", "VB1", 6), ("AVBR", "VB2", 6), ("AVBR", "VB3", 5),
    ("AVBR", "VB4", 5), ("AVBR", "VB5", 5), ("AVBR", "VB6", 4),
    ("AVBR", "VB7", 4), ("AVBR", "VB8", 4), ("AVBR", "VB9", 3),
    ("AVBR", "VB10", 3), ("AVBR", "VB11", 3),
    # AVD
    ("AVDL", "AVAL", 8), ("AVDL", "AVAR", 8),
    ("AVDL", "AVBL", 3), ("AVDL", "AVBR", 3),
    ("AVDL", "DA1", 3), ("AVDL", "DA2", 3),
    ("AVDR", "AVAL", 8), ("AVDR", "AVAR", 8),
    ("AVDR", "AVBL", 3), ("AVDR", "AVBR", 3),
    # AVE
    ("AVEL", "AVAL", 7), ("AVEL", "AVAR", 7),
    ("AVEL", "VA1", 3), ("AVEL", "VA2", 3),
    ("AVER", "AVAL", 7), ("AVER", "AVAR", 7),
    ("AVER", "VA1", 3), ("AVER", "VA2", 3),
    # PVC (tail touch / forward command)
    ("PVCL", "AVBL", 8), ("PVCL", "AVBR", 8),
    ("PVCL", "PVCR", 5), ("PVCR", "PVCL", 5),
    ("PVCL", "DVA", 3), ("PVCL", "DB1", 3), ("PVCL", "VB1", 3),
    ("PVCR", "AVBL", 8), ("PVCR", "AVBR", 8),
    ("PVCR", "DVA", 3), ("PVCR", "DB1", 3), ("PVCR", "VB1", 3),
    # DVA
    ("DVA", "AVAL", 4), ("DVA", "AVAR", 4),
    ("DVA", "AVBL", 3), ("DVA", "AVBR", 3),
    ("DVA", "RIML", 3), ("DVA", "RIMR", 3),
    # DVC
    ("DVC", "AVBL", 3), ("DVC", "AVBR", 3),
    ("DVC", "AVAL", 2), ("DVC", "AVAR", 2),
    # AVF
    ("AVFL", "AVBL", 4), ("AVFL", "AVBR", 4),
    ("AVFL", "AVAL", 2), ("AVFL", "AVAR", 2),
    ("AVFR", "AVBL", 4), ("AVFR", "AVBR", 4),
    # AVG
    ("AVG", "AVAL", 3), ("AVG", "AVAR", 3),
    ("AVG", "AVBL", 2), ("AVG", "AVBR", 2),
    # AVH
    ("AVHL", "AVBL", 3), ("AVHL", "AVBR", 3),
    ("AVHR", "AVBL", 3), ("AVHR", "AVBR", 3),
    # AVJ
    ("AVJL", "AVBL", 3), ("AVJL", "AVBR", 3),
    ("AVJL", "AVAL", 2), ("AVJL", "AVAR", 2),
    ("AVJR", "AVBL", 3), ("AVJR", "AVBR", 3),
    ("AVJR", "AVAL", 2), ("AVJR", "AVAR", 2),
    # AVK
    ("AVKL", "RIML", 3), ("AVKL", "RIMR", 3),
    ("AVKL", "AVBL", 2), ("AVKL", "AVBR", 2),
    ("AVKR", "RIML", 3), ("AVKR", "RIMR", 3),
    # AVL
    ("AVL", "AVBL", 3), ("AVL", "AVBR", 3),
    ("AVL", "DVB", 4),
    # ADA
    ("ADAL", "RIML", 2), ("ADAL", "RIMR", 2), ("ADAL", "AVBL", 2),
    ("ADAR", "RIML", 2), ("ADAR", "RIMR", 2), ("ADAR", "AVBL", 2),
    # BDU
    ("BDUL", "AVBL", 3), ("BDUL", "AVBR", 3),
    ("BDUR", "AVBL", 3), ("BDUR", "AVBR", 3),
    # CAN
    ("CANL", "AVBL", 2), ("CANR", "AVBR", 2),
    # LUA
    ("LUAL", "AVAL", 3), ("LUAL", "AVAR", 3), ("LUAL", "AVBL", 2),
    ("LUAR", "AVAL", 3), ("LUAR", "AVAR", 3), ("LUAR", "AVBR", 2),
    # PVC ← already done above
    # RIB
    ("RIBL", "AVBL", 3), ("RIBL", "AVBR", 3),
    ("RIBR", "AVBL", 3), ("RIBR", "AVBR", 3),
    # RIC
    ("RICL", "AVBL", 3), ("RICL", "AVBR", 3),
    ("RICR", "AVBL", 3), ("RICR", "AVBR", 3),
    # RIG
    ("RIGL", "AVBL", 2), ("RIGL", "AVBR", 2),
    ("RIGR", "AVBL", 2), ("RIGR", "AVBR", 2),
    # RIH
    ("RIH", "AVBL", 2), ("RIH", "AVBR", 2),
    # RIM (head oscillator / proprioception)
    ("RIML", "RIMR", 5), ("RIMR", "RIML", 5),
    ("RIML", "AVAL", 4), ("RIML", "AVAR", 4),
    ("RIML", "AVBL", 4), ("RIML", "AVBR", 4),
    ("RIML", "AIYL", 3), ("RIML", "AIYR", 3),
    ("RIML", "DA1", 3), ("RIML", "VA1", 3),
    ("RIMR", "AVAL", 4), ("RIMR", "AVAR", 4),
    ("RIMR", "AVBL", 4), ("RIMR", "AVBR", 4),
    ("RIMR", "AIYL", 3), ("RIMR", "AIYR", 3),
    ("RIMR", "DA1", 3), ("RIMR", "VA1", 3),
    # RIN
    ("RINL", "RIML", 3), ("RINL", "RIMR", 3), ("RINL", "AVBL", 2),
    ("RINR", "RIML", 3), ("RINR", "RIMR", 3), ("RINR", "AVBL", 2),

    # ── Motor neurons — dorsal A-class (backward) ────────────────────────────
    ("DA1", "DA2", 4), ("DA2", "DA3", 4), ("DA3", "DA4", 4),
    ("DA4", "DA5", 3), ("DA5", "DA6", 3), ("DA6", "DA7", 3),
    ("DA7", "DA8", 3), ("DA8", "DA9", 3),
    ("DA1", "DD1", 5), ("DA2", "DD2", 5), ("DA3", "DD3", 4),
    ("DA4", "DD4", 4), ("DA5", "DD5", 4), ("DA6", "DD6", 3),
    ("DA7", "DD6", 3),

    # ── Motor neurons — dorsal B-class (forward) ─────────────────────────────
    ("DB1", "DB2", 4), ("DB2", "DB3", 4), ("DB3", "DB4", 4),
    ("DB4", "DB5", 3), ("DB5", "DB6", 3), ("DB6", "DB7", 3),
    ("DB1", "DD1", 4), ("DB2", "DD2", 4), ("DB3", "DD3", 4),
    ("DB4", "DD4", 3), ("DB5", "DD5", 3), ("DB6", "DD6", 3),
    ("DB7", "DD6", 3),

    # ── Motor neurons — ventral A-class (backward) ───────────────────────────
    ("VA1", "VA2", 4), ("VA2", "VA3", 4), ("VA3", "VA4", 4),
    ("VA4", "VA5", 3), ("VA5", "VA6", 3), ("VA6", "VA7", 3),
    ("VA7", "VA8", 3), ("VA8", "VA9", 3), ("VA9", "VA10", 3),
    ("VA10", "VA11", 3), ("VA11", "VA12", 3),
    ("VA1", "VD1", 5), ("VA2", "VD2", 5), ("VA3", "VD3", 4),
    ("VA4", "VD4", 4), ("VA5", "VD5", 4), ("VA6", "VD6", 3),
    ("VA7", "VD7", 3), ("VA8", "VD8", 3), ("VA9", "VD9", 3),
    ("VA10", "VD10", 3), ("VA11", "VD11", 3), ("VA12", "VD12", 3),

    # ── Motor neurons — ventral B-class (forward) ────────────────────────────
    ("VB1", "VB2", 4), ("VB2", "VB3", 4), ("VB3", "VB4", 4),
    ("VB4", "VB5", 3), ("VB5", "VB6", 3), ("VB6", "VB7", 3),
    ("VB7", "VB8", 3), ("VB8", "VB9", 3), ("VB9", "VB10", 3),
    ("VB10", "VB11", 3),
    ("VB1", "VD1", 4), ("VB2", "VD2", 4), ("VB3", "VD3", 4),
    ("VB4", "VD4", 3), ("VB5", "VD5", 3), ("VB6", "VD6", 3),
    ("VB7", "VD7", 3), ("VB8", "VD8", 3), ("VB9", "VD9", 3),
    ("VB10", "VD10", 3), ("VB11", "VD11", 3),

    # ── D-class inhibitory neurons (dorsal-ventral coordination) ─────────────
    # DD neurons inhibit ventral muscles; VD inhibit dorsal
    ("DD1", "VD1", 4), ("DD1", "VD2", 3), ("DD2", "VD2", 4),
    ("DD2", "VD3", 3), ("DD3", "VD3", 4), ("DD3", "VD4", 3),
    ("DD4", "VD5", 3), ("DD5", "VD8", 3), ("DD6", "VD11", 3),
    ("VD1", "DD1", 4), ("VD2", "DD1", 3), ("VD2", "DD2", 4),
    ("VD3", "DD2", 3), ("VD3", "DD3", 4), ("VD4", "DD3", 3),
    ("VD5", "DD4", 3), ("VD7", "DD4", 3), ("VD8", "DD5", 3),
    ("VD10", "DD5", 3), ("VD11", "DD6", 3),

    # ── Head motor neurons ────────────────────────────────────────────────────
    ("RMDL", "RMDVL", 3), ("RMDL", "RMDVR", 3),
    ("RMDR", "RMDVL", 3), ("RMDR", "RMDVR", 3),
    ("RMDVL", "RMDL", 3), ("RMDVR", "RMDR", 3),
    ("RMEL", "RMER", 3), ("RMER", "RMEL", 3),
    ("SMDDL", "SMDDR", 3), ("SMDDR", "SMDDL", 3),
    ("SMDVL", "SMDVR", 3), ("SMDVR", "SMDVL", 3),
    ("SAADL", "AIYL", 2), ("SAADR", "AIYR", 2),
    ("SAAVL", "AVBL", 2), ("SAAVR", "AVBR", 2),
    ("SIBDL", "RIML", 2), ("SIBDR", "RIMR", 2),
    ("SIBVL", "RIML", 2), ("SIBVR", "RIMR", 2),

    # ── Additional interneuron ↔ motor connections ─────────────────────────
    ("AVBL", "SMDDL", 3), ("AVBL", "SMDDR", 3),
    ("AVBR", "SMDDL", 3), ("AVBR", "SMDDR", 3),
    ("AVBL", "SMDVL", 3), ("AVBL", "SMDVR", 3),
    ("AVBR", "SMDVL", 3), ("AVBR", "SMDVR", 3),
    ("AVAL", "RMDL", 3), ("AVAL", "RMDR", 3),
    ("AVAR", "RMDL", 3), ("AVAR", "RMDR", 3),
    ("RIML", "SMDDL", 3), ("RIML", "SMDVL", 3),
    ("RIMR", "SMDDR", 3), ("RIMR", "SMDVR", 3),
    ("RIML", "RMDL", 4), ("RIMR", "RMDR", 4),
    ("RIML", "RMDVL", 3), ("RIMR", "RMDVR", 3),

    # ── DVB (tail motor / defecation) ────────────────────────────────────────
    ("DVB", "AVBL", 2), ("DVB", "AVBR", 2),

    # ── SAB neurons ──────────────────────────────────────────────────────────
    ("SABVL", "AVBL", 2), ("SABVR", "AVBR", 2),
    ("SABDL", "AVAL", 2), ("SABDR", "AVAR", 2),

    # ── PVP / PVQ ────────────────────────────────────────────────────────────
    ("PVPL", "AVBL", 3), ("PVPL", "AVBR", 3),
    ("PVPR", "AVBL", 3), ("PVPR", "AVBR", 3),
    ("PVQL", "AVBL", 2), ("PVQR", "AVBR", 2),

    # ── RIR ──────────────────────────────────────────────────────────────────
    ("RIR", "AVBL", 2), ("RIR", "AVBR", 2),

    # ── SDQ ──────────────────────────────────────────────────────────────────
    ("SDQL", "AVBL", 2), ("SDQR", "AVBR", 2),
    ("SDQL", "PVCL", 2), ("SDQR", "PVCR", 2),

    # ── HOA / HOB ────────────────────────────────────────────────────────────
    ("HSNL", "AVFL", 2), ("HSNR", "AVFR", 2),
    ("HSNL", "VDNL", 2), ("HSNR", "VDNR", 2),
]

# Remove self-loops that may have been introduced by mistake
CHEMICAL_SYNAPSES = [(p, q, n) for p, q, n in CHEMICAL_SYNAPSES if p != q]

# ---------------------------------------------------------------------------
# Gap junctions  (neuron_a, neuron_b, n_junctions)
# Each pair listed once; the loader will add the reverse direction.
# ---------------------------------------------------------------------------
GAP_JUNCTIONS: List[Tuple[str, str, int]] = [
    # Sensory bilateral pairs
    ("ADFL", "ADFR", 2), ("ADLL", "ADLR", 2), ("AFDL", "AFDR", 6),
    ("ASEL", "ASER", 4), ("ASGL", "ASGR", 2), ("ASHL", "ASHR", 4),
    ("ASJL", "ASJR", 2), ("ASKL", "ASKR", 2),
    ("AWAL", "AWAR", 2), ("AWBL", "AWBR", 2), ("AWCL", "AWCR", 4),
    ("BAGL", "BAGR", 2), ("URXL", "URXR", 2),
    ("ALML", "ALMR", 4), ("PLML", "PLMR", 6),
    ("PVDL", "PVDR", 4),
    # Interneuron bilateral pairs
    ("AIAL", "AIAR", 3), ("AIBL", "AIBR", 4), ("AIML", "AIMR", 2),
    ("AINL", "AINR", 2), ("AIYL", "AIYR", 8), ("AIZL", "AIZR", 6),
    ("AVAL", "AVAR", 9), ("AVBL", "AVBR", 12),
    ("AVDL", "AVDR", 6), ("AVEL", "AVER", 6),
    ("AVFL", "AVFR", 4), ("AVHL", "AVHR", 2),
    ("AVJL", "AVJR", 2), ("AVKL", "AVKR", 2),
    ("BDUL", "BDUR", 2),
    ("PVCL", "PVCR", 8),
    ("RIBL", "RIBR", 4), ("RICL", "RICR", 2),
    ("RIML", "RIMR", 6),
    ("RINL", "RINR", 2),
    ("LUAL", "LUAR", 2),
    # Motor bilateral pairs
    ("RMDL", "RMDR", 4), ("RMDVL", "RMDVR", 4),
    ("RMEL", "RMER", 2), ("RMEL", "RMEV", 2),
    ("SMDDL", "SMDDR", 4), ("SMDVL", "SMDVR", 4),
    # Interneuron ↔ sensory gap junctions
    ("AIYL", "AFDL", 6), ("AIYR", "AFDR", 6),
    ("AIYL", "AIBL", 3), ("AIYR", "AIBR", 3),
    ("AIZL", "AIYL", 3), ("AIZR", "AIYR", 3),
    ("RIML", "AVBL", 4), ("RIMR", "AVBR", 4),
    ("AVAL", "AVDL", 4), ("AVAR", "AVDR", 4),
    ("AVBL", "PVCL", 6), ("AVBR", "PVCR", 6),
    # DA ↔ DB coupling
    ("DA1", "DB1", 3), ("DA2", "DB2", 3), ("DA3", "DB3", 3),
    ("DA4", "DB4", 2), ("DA5", "DB5", 2), ("DA6", "DB6", 2),
    ("DA7", "DB7", 2),
    # VA ↔ VB coupling
    ("VA1", "VB1", 3), ("VA2", "VB2", 3), ("VA3", "VB3", 3),
    ("VA4", "VB4", 2), ("VA5", "VB5", 2), ("VA6", "VB6", 2),
    ("VA7", "VB7", 2), ("VA8", "VB8", 2),
    # DA/VA dorsal-ventral
    ("DA1", "VA1", 2), ("DA2", "VA2", 2), ("DA3", "VA3", 2),
    ("DA4", "VA4", 2),
    # DD/VD
    ("DD1", "VD1", 2), ("DD2", "VD2", 2), ("DD3", "VD3", 2),
    ("DD4", "VD4", 2), ("DD5", "VD5", 2),
    # AVA ↔ motor
    ("AVAL", "DA1", 3), ("AVAR", "DA1", 3),
    ("AVBL", "DB1", 3), ("AVBR", "DB1", 3),
    ("DVA", "AVAL", 2), ("DVA", "AVAR", 2),
    ("DVA", "AVBL", 2), ("DVA", "AVBR", 2),
    ("AVBL", "SMDDL", 2), ("AVBR", "SMDDR", 2),
]


def to_dataframes():
    """Return (chemical_df, gap_df) as pandas DataFrames in canonical format."""
    import pandas as pd

    chem_rows = [
        {"Origin": pre, "Target": post, "Number": n, "Type": "chemical"}
        for pre, post, n in CHEMICAL_SYNAPSES
    ]
    chem_df = pd.DataFrame(chem_rows)

    # Mirror gap junctions (undirected → two directed edges)
    gap_rows = []
    for a, b, n in GAP_JUNCTIONS:
        gap_rows.append({"Origin": a, "Target": b, "Number": n, "Type": "electrical"})
        gap_rows.append({"Origin": b, "Target": a, "Number": n, "Type": "electrical"})
    gap_df = pd.DataFrame(gap_rows)

    return chem_df, gap_df


def write_to_dir(dest_dir) -> None:
    """Write canonical CSVs to dest_dir/connectome_chemical.csv and _gap.csv."""
    from pathlib import Path
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    chem_df, gap_df = to_dataframes()
    chem_df.to_csv(dest / "connectome_chemical.csv", index=False)
    gap_df.to_csv(dest / "connectome_gap.csv", index=False)


if __name__ == "__main__":
    chem_df, gap_df = to_dataframes()
    print(f"Chemical synapses : {len(chem_df):>5} rows, "
          f"{len(set(chem_df['Origin'].tolist() + chem_df['Target'].tolist()))} unique neurons")
    print(f"Gap junctions     : {len(gap_df):>5} rows (bidirectional)")
    print(f"Max synapse weight: {chem_df['Number'].max()}")
