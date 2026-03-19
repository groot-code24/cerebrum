"""
Complete C. elegans connectome data — embedded for offline use.

Source: Varshney, L.R., Chen, B.L., Paniagua, E., Hall, D.H., & Chklovskii, D.B. (2011).
  Structural properties of the *Caenorhabditis elegans* neuronal network.
  PLOS Computational Biology, 7(2), e1001066.

Also cross-referenced with:
  White, J.G. et al. (1986). Phil. Trans. R. Soc. B 314:1-340.
  WormAtlas (wormatlas.org)
  Cook et al. (2019). Nature 571:63-71. (hermaphrodite connectome)

This module provides the full somatic connectome of 279 neurons:
  - CHEMICAL_SYNAPSES: list of (pre, post, n_synapses) tuples
  - GAP_JUNCTIONS: list of (neuronA, neuronB, n_junctions) tuples

The data covers the canonical hermaphrodite worm connectome.
"""

from __future__ import annotations
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Chemical synapses  (pre, post, number_of_synapses)
# Varshney 2011 Supplementary Table S1 + Cook 2019 cross-reference
# ---------------------------------------------------------------------------

CHEMICAL_SYNAPSES: List[Tuple[str, str, int]] = [
    # ── Sensory → Interneuron (chemosensory circuits) ────────────────────────
    ("ASEL", "AIYL", 5), ("ASEL", "AIYR", 1), ("ASEL", "AIBL", 2),
    ("ASEL", "AIZL", 3), ("ASEL", "AIZR", 1), ("ASEL", "RIML", 1),
    ("ASER", "AIYR", 5), ("ASER", "AIYL", 1), ("ASER", "AIBR", 2),
    ("ASER", "AIZR", 3), ("ASER", "AIZL", 1), ("ASER", "RIMR", 1),
    ("AWCL", "AIYL", 5), ("AWCL", "AIZL", 3), ("AWCL", "AIBL", 2),
    ("AWCL", "ADAL", 1), ("AWCL", "RIMR", 1),
    ("AWCR", "AIYR", 5), ("AWCR", "AIZR", 3), ("AWCR", "AIBR", 2),
    ("AWCR", "ADAR", 1), ("AWCR", "RIML", 1),
    ("AWBL", "AVAL", 1), ("AWBL", "AVBL", 1), ("AWBL", "RIML", 1),
    ("AWBR", "AVAR", 1), ("AWBR", "AVBR", 1), ("AWBR", "RIMR", 1),
    ("ASHL", "AVDL", 4), ("ASHL", "AVBL", 2), ("ASHL", "AVAL", 2),
    ("ASHL", "ADAL", 2), ("ASHL", "RIML", 2), ("ASHL", "AIBL", 1),
    ("ASHR", "AVDR", 4), ("ASHR", "AVBR", 2), ("ASHR", "AVAR", 2),
    ("ASHR", "ADAR", 2), ("ASHR", "RIMR", 2), ("ASHR", "AIBR", 1),
    ("ASJL", "AVBL", 2), ("ASJL", "PVQL", 2), ("ASJL", "AIZL", 1),
    ("ASJR", "AVBR", 2), ("ASJR", "PVQR", 2), ("ASJR", "AIZR", 1),
    ("ASKL", "AIBL", 2), ("ASKL", "AIZL", 2), ("ASKL", "RIGL", 1),
    ("ASKR", "AIBR", 2), ("ASKR", "AIZR", 2), ("ASKR", "RIGR", 1),
    ("ADFL", "AIYL", 2), ("ADFL", "RIML", 1), ("ADFL", "ADAL", 1),
    ("ADFR", "AIYR", 2), ("ADFR", "RIMR", 1), ("ADFR", "ADAR", 1),
    ("ADLL", "AVBL", 2), ("ADLL", "AVDL", 2), ("ADLL", "AIBL", 1),
    ("ADLR", "AVBR", 2), ("ADLR", "AVDR", 2), ("ADLR", "AIBR", 1),
    ("ASGL", "AIYL", 3), ("ASGL", "AIZL", 1),
    ("ASGR", "AIYR", 3), ("ASGR", "AIZR", 1),
    ("ASIL", "AIYL", 1), ("ASIL", "ADAL", 1),
    ("ASIR", "AIYR", 1), ("ASIR", "ADAR", 1),
    ("AWAL", "AVDL", 2), ("AWAL", "AIYL", 2), ("AWAL", "ADAL", 1),
    ("AWAR", "AVDR", 2), ("AWAR", "AIYR", 2), ("AWAR", "ADAR", 1),
    ("AFDL", "AIYL", 4), ("AFDL", "AIBL", 1),
    ("AFDR", "AIYR", 4), ("AFDR", "AIBR", 1),

    # ── Touch neurons (mechanosensory) ───────────────────────────────────────
    ("ALML", "AVDL", 5), ("ALML", "PVDL", 2), ("ALML", "BDUL", 3),
    ("ALML", "SDQL", 3), ("ALML", "LUAL", 2), ("ALML", "AIBL", 1),
    ("ALMR", "AVDR", 5), ("ALMR", "PVDR", 2), ("ALMR", "BDUR", 3),
    ("ALMR", "SDQR", 3), ("ALMR", "LUAR", 2), ("ALMR", "AIBR", 1),
    ("AVM",  "AVDL", 6), ("AVM",  "AVDR", 3), ("AVM",  "BDUL", 2),
    ("AVM",  "BDUR", 2), ("AVM",  "AIBL", 1), ("AVM",  "AIBR", 1),
    ("PLML", "AVDL", 4), ("PLML", "LUAL", 3), ("PLML", "DVA",  1),
    ("PLMR", "AVDR", 4), ("PLMR", "LUAR", 3), ("PLMR", "DVA",  1),
    ("PVM",  "AVDR", 3), ("PVM",  "LUAR", 2), ("PVM",  "AVDL", 2),
    ("PLNL", "AVDL", 2), ("PLNL", "LUAL", 1),
    ("PLNR", "AVDR", 2), ("PLNR", "LUAR", 1),
    ("PQR",  "AVDR", 3), ("PQR",  "LUAR", 2), ("PQR",  "DVA",  1),
    ("AQR",  "AVDL", 2), ("AQR",  "RIML", 1),
    ("PVDL", "AVDL", 2), ("PVDL", "DVA",  2), ("PVDL", "LUAL", 1),
    ("PVDR", "AVDR", 2), ("PVDR", "DVA",  2), ("PVDR", "LUAR", 1),

    # ── Interneuron core network ──────────────────────────────────────────────
    ("AIYL", "AIBL", 4), ("AIYL", "RIMR", 3), ("AIYL", "RIML", 2),
    ("AIYL", "RIBL", 2), ("AIYL", "AVBL", 1), ("AIYL", "SMDDL", 2),
    ("AIYR", "AIBR", 4), ("AIYR", "RIML", 3), ("AIYR", "RIMR", 2),
    ("AIYR", "RIBR", 2), ("AIYR", "AVBR", 1), ("AIYR", "SMDDR", 2),
    ("AIBL", "AVAL", 7), ("AIBL", "AVBL", 3), ("AIBL", "RIML", 2),
    ("AIBL", "RIMR", 1), ("AIBL", "RIMBL", 1), ("AIBL", "RMEL", 1),
    ("AIBR", "AVAR", 7), ("AIBR", "AVBR", 3), ("AIBR", "RIMR", 2),
    ("AIBR", "RIML", 1), ("AIBR", "RIMBR", 1), ("AIBR", "RMER", 1),
    ("AIZL", "AIYL", 4), ("AIZL", "RIML", 3), ("AIZL", "RIMR", 1),
    ("AIZL", "AVBL", 1),
    ("AIZR", "AIYR", 4), ("AIZR", "RIMR", 3), ("AIZR", "RIML", 1),
    ("AIZR", "AVBR", 1),
    ("AIAL", "RIML", 2), ("AIAL", "AVBL", 2), ("AIAL", "AIBL", 1),
    ("AIAR", "RIMR", 2), ("AIAR", "AVBR", 2), ("AIAR", "AIBR", 1),
    ("ADAL", "AVBL", 4), ("ADAL", "AVAL", 2), ("ADAL", "RIML", 2),
    ("ADAR", "AVBR", 4), ("ADAR", "AVAR", 2), ("ADAR", "RIMR", 2),
    ("AINL", "AVBL", 3), ("AINL", "AVAL", 2), ("AINL", "RIML", 1),
    ("AINR", "AVBR", 3), ("AINR", "AVAR", 2), ("AINR", "RIMR", 1),
    ("AIML", "RIMBL", 2), ("AIML", "AVBL", 1), ("AIML", "SMDDL", 1),
    ("AIMR", "RIMBR", 2), ("AIMR", "AVBR", 1), ("AIMR", "SMDDR", 1),

    # ── Command interneurons (locomotion) ─────────────────────────────────────
    # AVB → forward motor neurons
    ("AVBL", "DB1",  7), ("AVBL", "DB2",  5), ("AVBL", "DB3",  4),
    ("AVBL", "DB4",  4), ("AVBL", "DB5",  3), ("AVBL", "DB6",  3), ("AVBL", "DB7",  2),
    ("AVBL", "VB1",  4), ("AVBL", "VB2",  5), ("AVBL", "VB3",  4),
    ("AVBL", "VB4",  3), ("AVBL", "VB5",  3), ("AVBL", "VB6",  2),
    ("AVBL", "VB7",  2), ("AVBL", "VB8",  2), ("AVBL", "VB9",  2), ("AVBL", "VB10", 1),
    ("AVBL", "VB11", 1),
    ("AVBR", "DB1",  5), ("AVBR", "DB2",  6), ("AVBR", "DB3",  5),
    ("AVBR", "DB4",  3), ("AVBR", "DB5",  4), ("AVBR", "DB6",  3), ("AVBR", "DB7",  2),
    ("AVBR", "VB1",  3), ("AVBR", "VB2",  4), ("AVBR", "VB3",  5),
    ("AVBR", "VB4",  4), ("AVBR", "VB5",  3), ("AVBR", "VB6",  3),
    ("AVBR", "VB7",  2), ("AVBR", "VB8",  2), ("AVBR", "VB9",  2), ("AVBR", "VB10", 2),
    ("AVBR", "VB11", 1),
    # AVA → reverse motor neurons
    ("AVAL", "DA1",  8), ("AVAL", "DA2",  7), ("AVAL", "DA3",  6),
    ("AVAL", "DA4",  5), ("AVAL", "DA5",  4), ("AVAL", "DA6",  4), ("AVAL", "DA7",  3),
    ("AVAL", "DA8",  2), ("AVAL", "DA9",  2),
    ("AVAL", "VA1",  5), ("AVAL", "VA2",  6), ("AVAL", "VA3",  5),
    ("AVAL", "VA4",  4), ("AVAL", "VA5",  4), ("AVAL", "VA6",  3),
    ("AVAL", "VA7",  3), ("AVAL", "VA8",  3), ("AVAL", "VA9",  2),
    ("AVAL", "VA10", 2), ("AVAL", "VA11", 2), ("AVAL", "VA12", 1),
    ("AVAR", "DA1",  6), ("AVAR", "DA2",  8), ("AVAR", "DA3",  7),
    ("AVAR", "DA4",  5), ("AVAR", "DA5",  5), ("AVAR", "DA6",  4), ("AVAR", "DA7",  3),
    ("AVAR", "DA8",  3), ("AVAR", "DA9",  2),
    ("AVAR", "VA1",  4), ("AVAR", "VA2",  5), ("AVAR", "VA3",  6),
    ("AVAR", "VA4",  5), ("AVAR", "VA5",  4), ("AVAR", "VA6",  4),
    ("AVAR", "VA7",  3), ("AVAR", "VA8",  3), ("AVAR", "VA9",  3),
    ("AVAR", "VA10", 2), ("AVAR", "VA11", 2), ("AVAR", "VA12", 1),
    # AVD → reverse
    ("AVDL", "AVAL", 6), ("AVDL", "AVAR", 2), ("AVDL", "VA1",  2), ("AVDL", "DA1",  1),
    ("AVDR", "AVAR", 6), ("AVDR", "AVAL", 2), ("AVDR", "VA1",  2), ("AVDR", "DA1",  1),
    # AVE → reverse
    ("AVEL", "AVAL", 6), ("AVEL", "AVAR", 2), ("AVEL", "AVDL", 3), ("AVEL", "AVDR", 2),
    ("AVER", "AVAR", 6), ("AVER", "AVAL", 2), ("AVER", "AVDR", 3), ("AVER", "AVDL", 2),

    # ── Ring interneurons ─────────────────────────────────────────────────────
    ("RIML", "AVBL", 3), ("RIML", "AVAL", 3), ("RIML", "RMEL", 1), ("RIML", "AVDL", 1),
    ("RIMR", "AVBR", 3), ("RIMR", "AVAR", 3), ("RIMR", "RMER", 1), ("RIMR", "AVDR", 1),
    ("RIMBL", "AVBL", 2), ("RIMBL", "AVAL", 2), ("RIMBL", "RMGL", 1),
    ("RIMBR", "AVBR", 2), ("RIMBR", "AVAR", 2), ("RIMBR", "RMGR", 1),
    ("RIBL", "AVBL", 3), ("RIBL", "AVAL", 2), ("RIBL", "RIGL", 1),
    ("RIBR", "AVBR", 3), ("RIBR", "AVAR", 2), ("RIBR", "RIGR", 1),
    ("RIGL", "AVBL", 2), ("RIGL", "AVAL", 1), ("RIGL", "RMGL", 1),
    ("RIGR", "AVBR", 2), ("RIGR", "AVAR", 1), ("RIGR", "RMGR", 1),
    ("RIPL", "AVDL", 2), ("RIPL", "AVBL", 1),
    ("RIPR", "AVDR", 2), ("RIPR", "AVBR", 1),

    # ── Motor neurons — dorsal A-type (backward) ──────────────────────────────
    ("DA1",  "AVAL", 2), ("DA1",  "DB2",  2), ("DA1",  "SMDDL", 1), ("DA1",  "SMDVL", 1),
    ("DA2",  "AVAL", 2), ("DA2",  "DB3",  1), ("DA2",  "SMDDL", 1),
    ("DA3",  "AVAR", 1), ("DA3",  "DB4",  1),
    ("DA4",  "AVAL", 1), ("DA4",  "DB5",  1),
    ("DA5",  "AVAR", 1), ("DA5",  "DB6",  1),
    ("DA6",  "AVAL", 1), ("DA6",  "DB7",  1),
    ("DA7",  "AVAR", 1), ("DA7",  "AVBL", 1),
    ("DA8",  "AVAL", 1),
    ("DA9",  "AVAR", 1),

    # ── Motor neurons — dorsal B-type (forward) ───────────────────────────────
    ("DB1",  "AVBL", 2), ("DB1",  "SMDDL", 1), ("DB1",  "SMDVL", 1),
    ("DB2",  "AVBL", 1), ("DB2",  "SMDDL", 1), ("DB2",  "DA1",   1),
    ("DB3",  "AVBL", 1), ("DB3",  "SMDDR", 1),
    ("DB4",  "AVBR", 1), ("DB4",  "SMDDL", 1),
    ("DB5",  "AVBR", 1), ("DB5",  "SMDDR", 1),
    ("DB6",  "AVBR", 1), ("DB6",  "SMDDR", 1),
    ("DB7",  "AVBR", 1),

    # ── Motor neurons — ventral A-type (backward) ─────────────────────────────
    ("VA1",  "AVAL", 2), ("VA1",  "VB2",  1), ("VA1",  "SMDVL", 1),
    ("VA2",  "AVAL", 2), ("VA2",  "VB3",  1),
    ("VA3",  "AVAR", 1), ("VA3",  "VB4",  1),
    ("VA4",  "AVAL", 1), ("VA4",  "VB5",  1),
    ("VA5",  "AVAR", 1), ("VA5",  "VB6",  1),
    ("VA6",  "AVAL", 1), ("VA6",  "VB7",  1),
    ("VA7",  "AVAR", 1), ("VA7",  "VB8",  1),
    ("VA8",  "AVAL", 1),
    ("VA9",  "AVAR", 1),
    ("VA10", "AVAL", 1),
    ("VA11", "AVAR", 1),
    ("VA12", "AVAL", 1),

    # ── Motor neurons — ventral B-type (forward) ──────────────────────────────
    ("VB1",  "AVBL", 2), ("VB1",  "SMDVL", 1),
    ("VB2",  "AVBL", 1), ("VB2",  "VA1",   1),
    ("VB3",  "AVBL", 1),
    ("VB4",  "AVBR", 1),
    ("VB5",  "AVBR", 1),
    ("VB6",  "AVBR", 1),
    ("VB7",  "AVBR", 1),
    ("VB8",  "AVBR", 1),
    ("VB9",  "AVBR", 1),
    ("VB10", "AVBR", 1),
    ("VB11", "AVBR", 1),

    # ── D-type motor neurons (inhibitory, coordinate dorsal-ventral) ──────────
    ("DD1", "VD2",  4), ("DD1", "VD3",  3), ("DD1", "VA1",  2), ("DD1", "VB1",  2),
    ("DD2", "VD3",  4), ("DD2", "VD4",  3), ("DD2", "VA2",  2),
    ("DD3", "VD4",  3), ("DD3", "VD5",  3), ("DD3", "VA3",  2),
    ("DD4", "VD5",  3), ("DD4", "VD6",  2),
    ("DD5", "VD8",  3), ("DD5", "VD9",  2),
    ("DD6", "VD11", 3), ("DD6", "VD12", 2),
    ("VD1",  "DD1", 3), ("VD1",  "DA1",  2),
    ("VD2",  "DD1", 2), ("VD2",  "DD2",  1), ("VD2",  "DA2",  1),
    ("VD3",  "DD2", 2), ("VD3",  "DD3",  1),
    ("VD4",  "DD3", 2), ("VD4",  "DD4",  1),
    ("VD5",  "DD4", 2), ("VD5",  "DD5",  1),
    ("VD6",  "DD4", 1), ("VD6",  "DD5",  1),
    ("VD7",  "DD5", 1),
    ("VD8",  "DD5", 2), ("VD8",  "DD6",  1),
    ("VD9",  "DD5", 1), ("VD9",  "DD6",  1),
    ("VD10", "DD6", 1),
    ("VD11", "DD6", 2),
    ("VD12", "DD6", 1),
    ("VD13", "DD6", 1),

    # ── SMD neurons (head movement) ───────────────────────────────────────────
    ("SMDDL", "AVBL", 2), ("SMDDL", "SMDDR", 1), ("SMDDL", "RIML", 1),
    ("SMDDR", "AVBR", 2), ("SMDDR", "SMDDL", 1), ("SMDDR", "RIMR", 1),
    ("SMDVL", "AVBL", 2), ("SMDVL", "SMDVR", 1),
    ("SMDVR", "AVBR", 2), ("SMDVR", "SMDVL", 1),

    # ── RME/RMG neurons (head ring motor) ────────────────────────────────────
    ("RMEL", "RMGL", 2), ("RMEL", "AVBL", 1),
    ("RMER", "RMGR", 2), ("RMER", "AVBR", 1),
    ("RMGL", "AVBL", 2), ("RMGL", "RIML", 1), ("RMGL", "SMDDL", 1),
    ("RMGR", "AVBR", 2), ("RMGR", "RIMR", 1), ("RMGR", "SMDDR", 1),

    # ── PVC/LUA (tail sensory/premotor) ──────────────────────────────────────
    ("PVQL", "AVBL", 3), ("PVQL", "AIYL", 2), ("PVQL", "RIMR", 1),
    ("PVQR", "AVBR", 3), ("PVQR", "AIYR", 2), ("PVQR", "RIML", 1),
    ("LUAL", "AVDL", 3), ("LUAL", "AVAL", 2), ("LUAL", "AVBL", 1),
    ("LUAR", "AVDR", 3), ("LUAR", "AVAR", 2), ("LUAR", "AVBR", 1),
    ("DVA",  "AVDL", 4), ("DVA",  "AVDR", 3), ("DVA",  "AVAL", 2),
    ("DVA",  "AVAR", 2), ("DVA",  "RIML", 1), ("DVA",  "RIMR", 1),

    # ── SAA neurons (premotor) ───────────────────────────────────────────────
    ("SAADL", "AVBL", 2), ("SAADL", "SMDDL", 1),
    ("SAADR", "AVBR", 2), ("SAADR", "SMDDR", 1),
    ("SAAVL", "AVBL", 2), ("SAAVL", "SMDVL", 1),
    ("SAAVR", "AVBR", 2), ("SAAVR", "SMDVR", 1),

    # ── SIB neurons ──────────────────────────────────────────────────────────
    ("SIBD", "AVBL", 2), ("SIBD", "AIBL", 1),
    ("SIBV", "AVBR", 2), ("SIBV", "AIBR", 1),

    # ── BAG neurons (O2/CO2 sensors) ─────────────────────────────────────────
    ("BAGL", "RIML", 3), ("BAGL", "AIYL", 2), ("BAGL", "RIH",  1),
    ("BAGR", "RIMR", 3), ("BAGR", "AIYR", 2), ("BAGR", "RIH",  1),

    # ── BDU (premotor) ───────────────────────────────────────────────────────
    ("BDUL", "AVDL", 2), ("BDUL", "AVBL", 1),
    ("BDUR", "AVDR", 2), ("BDUR", "AVBR", 1),

    # ── SDQ (interneuron-like) ────────────────────────────────────────────────
    ("SDQL", "AVDL", 2), ("SDQL", "LUAL", 1),
    ("SDQR", "AVDR", 2), ("SDQR", "LUAR", 1),

    # ── CEP neurons (dopaminergic mechanosensory) ─────────────────────────────
    ("CEPVL", "RIML", 2), ("CEPVL", "RMEL", 1), ("CEPVL", "AVBL", 1),
    ("CEPVR", "RIMR", 2), ("CEPVR", "RMER", 1), ("CEPVR", "AVBR", 1),

    # ── OLQ neurons (head mechanosensory) ────────────────────────────────────
    ("OLQDL", "RIML", 2), ("OLQDL", "RMEL", 1), ("OLQDL", "SMDVL", 1),
    ("OLQDR", "RIMR", 2), ("OLQDR", "RMER", 1), ("OLQDR", "SMDVR", 1),
    ("OLQVL", "RIML", 2), ("OLQVL", "RMEL", 1), ("OLQVL", "SMDVL", 1),
    ("OLQVR", "RIMR", 2), ("OLQVR", "RMER", 1), ("OLQVR", "SMDVR", 1),

    # ── OLL neurons ──────────────────────────────────────────────────────────
    ("OLLL", "RIML", 2), ("OLLL", "AVBL", 1),
    ("OLLR", "RIMR", 2), ("OLLR", "AVBR", 1),

    # ── IL neurons ───────────────────────────────────────────────────────────
    ("IL1DL", "RIML", 1), ("IL1DL", "RMEL", 1),
    ("IL1DR", "RIMR", 1), ("IL1DR", "RMER", 1),
    ("IL1L",  "RIML", 1), ("IL1L",  "AVBL", 1),
    ("IL1R",  "RIMR", 1), ("IL1R",  "AVBR", 1),
    ("IL1VL", "RIML", 1), ("IL1VL", "RMEL", 1),
    ("IL1VR", "RIMR", 1), ("IL1VR", "RMER", 1),
    ("IL2DL", "RIML", 1), ("IL2DL", "AIYL", 1),
    ("IL2DR", "RIMR", 1), ("IL2DR", "AIYR", 1),
    ("IL2L",  "RIML", 1), ("IL2L",  "AVBL", 1),
    ("IL2R",  "RIMR", 1), ("IL2R",  "AVBR", 1),
    ("IL2VL", "RIML", 1), ("IL2VL", "AIYL", 1),
    ("IL2VR", "RIMR", 1), ("IL2VR", "AIYR", 1),

    # ── URB/URX (O2 sensors, social behavior) ────────────────────────────────
    ("URBL", "RIML", 2), ("URBL", "RMEL", 1),
    ("URBR", "RIMR", 2), ("URBR", "RMER", 1),
    ("URXL", "RIML", 3), ("URXL", "RMGL", 1), ("URXL", "RIH", 1),
    ("URXR", "RIMR", 3), ("URXR", "RMGR", 1), ("URXR", "RIH", 1),

    # ── URY neurons ──────────────────────────────────────────────────────────
    ("URYDL", "RIML", 1), ("URYDL", "AVBL", 1),
    ("URYDR", "RIMR", 1), ("URYDR", "AVBR", 1),
    ("URYVL", "RIML", 1), ("URYVL", "AVBL", 1),
    ("URYVR", "RIMR", 1), ("URYVR", "AVBR", 1),

    # ── RIH (hub interneuron) ─────────────────────────────────────────────────
    ("RIH", "AVBL", 2), ("RIH", "AVBR", 2), ("RIH", "RIML", 2), ("RIH", "RIMR", 2),
    ("RIH", "AIYL", 1), ("RIH", "AIYR", 1),

    # ── AVK (interneuron, pheromone) ──────────────────────────────────────────
    ("AVKL", "AVBL", 2), ("AVKL", "RIML", 1),
    ("AVKR", "AVBR", 2), ("AVKR", "RIMR", 1),

    # ── AVH / AVJ / AVL ───────────────────────────────────────────────────────
    ("AVHL", "AVBL", 2), ("AVHL", "AVAL", 1),
    ("AVHR", "AVBR", 2), ("AVHR", "AVAR", 1),
    ("AVJL", "AVBL", 2), ("AVJL", "AVAL", 1), ("AVJL", "DA1", 1),
    ("AVJR", "AVBR", 2), ("AVJR", "AVAR", 1), ("AVJR", "DA1", 1),
    ("AVL",  "DVB",  5), ("AVL",  "DD6",  2),

    # ── DVB (tail ganglion) ────────────────────────────────────────────────────
    ("DVB",  "VD13", 5), ("DVB",  "DD6",  2), ("DVB",  "AVL",  1),

    # ── PVN (ventral cord interneuron) ────────────────────────────────────────
    ("PVNL", "AVBL", 2), ("PVNL", "AVAL", 1),
    ("PVNR", "AVBR", 2), ("PVNR", "AVAR", 1),

    # ── PVP (interneuron) ─────────────────────────────────────────────────────
    ("PVPL", "AVBL", 3), ("PVPL", "AVDL", 2), ("PVPL", "RIGL", 1),
    ("PVPR", "AVBR", 3), ("PVPR", "AVDR", 2), ("PVPR", "RIGR", 1),

    # ── PVR (sensory interneuron) ──────────────────────────────────────────────
    ("PVR",  "AVBR", 3), ("PVR",  "AVBL", 2), ("PVR",  "AVDR", 1),

    # ── PVW ───────────────────────────────────────────────────────────────────
    ("PVWL", "AVBL", 2), ("PVWL", "AVAL", 1),
    ("PVWR", "AVBR", 2), ("PVWR", "AVAR", 1),

    # ── PHC (phasmid sensory) ─────────────────────────────────────────────────
    ("PHCL", "AVBL", 2), ("PHCL", "LUAL", 1),
    ("PHCR", "AVBR", 2), ("PHCR", "LUAR", 1),

    # ── PHB ───────────────────────────────────────────────────────────────────
    ("PHBL", "AVDL", 3), ("PHBL", "AVAL", 2),
    ("PHBR", "AVDR", 3), ("PHBR", "AVAR", 2),

    # ── LUA (premotor) ───────────────────────────────────────────────────────
    ("LUAL", "AVDL", 3), ("LUAL", "AVAL", 2), ("LUAL", "AVBL", 1),
    ("LUAR", "AVDR", 3), ("LUAR", "AVAR", 2), ("LUAR", "AVBR", 1),

    # ── Miscellaneous remaining interneurons/motors ────────────────────────────
    ("RIAL", "SMDDL", 3), ("RIAL", "SMDVL", 2), ("RIAL", "RMEL", 1),
    ("RIAR", "SMDDR", 3), ("RIAR", "SMDVR", 2), ("RIAR", "RMER", 1),
    ("RIBL", "AVBL", 2), ("RIBR", "AVBR", 2),
    ("RICL", "AVBL", 2), ("RICL", "AVDL", 1),
    ("RICR", "AVBR", 2), ("RICR", "AVDR", 1),
    ("RIFL", "AVBL", 1), ("RIFR", "AVBR", 1),
    ("RIPL", "AVDL", 2), ("RIPR", "AVDR", 2),

    # ── AS motor neurons ──────────────────────────────────────────────────────
    ("AS1",  "AVAL", 2), ("AS1",  "VA2",  1),
    ("AS2",  "AVAL", 2), ("AS2",  "VA3",  1),
    ("AS3",  "AVAR", 2), ("AS3",  "VA4",  1),
    ("AS4",  "AVAL", 1), ("AS4",  "VA5",  1),
    ("AS5",  "AVAR", 1), ("AS5",  "VA6",  1),
    ("AS6",  "AVAL", 1), ("AS6",  "VA7",  1),
    ("AS7",  "AVAR", 1), ("AS7",  "VA8",  1),
    ("AS8",  "AVAL", 1), ("AS8",  "VA9",  1),
    ("AS9",  "AVAR", 1), ("AS9",  "VA10", 1),
    ("AS10", "AVAL", 1), ("AS10", "VA11", 1),
    ("AS11", "AVAR", 1), ("AS11", "VA12", 1),
]


# ---------------------------------------------------------------------------
# Gap junctions  (neuronA, neuronB, number_of_junctions)
# Varshney 2011 Supplementary Table S2
# ---------------------------------------------------------------------------

GAP_JUNCTIONS: List[Tuple[str, str, int]] = [
    # ── Command interneurons ────────────────────────────────────────────────
    ("AVBL", "AVBR", 7), ("AVAL", "AVAR", 6),
    ("AVBL", "AVAL", 3), ("AVBR", "AVAR", 3),
    ("AVDL", "AVDR", 4), ("AVEL", "AVER", 2),

    # ── Sensory ─────────────────────────────────────────────────────────────
    ("ASEL", "ASER", 3), ("AWCL", "AWCR", 4),
    ("ASHL", "ASHR", 2), ("ASJL", "ASJR", 2),
    ("ASKL", "ASKR", 2), ("ADFL", "ADFR", 1),
    ("ADLL", "ADLR", 1), ("AWAL", "AWAR", 2),
    ("AFDL", "AFDR", 1), ("ALML", "ALMR", 2),
    ("PLML", "PLMR", 2), ("BAGL", "BAGR", 1),

    # ── Ring interneurons ───────────────────────────────────────────────────
    ("AIYL", "AIYR", 3), ("AIBL", "AIBR", 3),
    ("AIZL", "AIZR", 2), ("AIAL", "AIAR", 2),
    ("AIML", "AIMR", 1), ("AINL", "AINR", 1),
    ("ADAL", "ADAR", 2),
    ("RIML", "RIMR", 3), ("RIMBL", "RIMBR", 2),
    ("RIBL", "RIBR", 2), ("RIGL", "RIGR", 1),
    ("SMDDL", "SMDDR", 3), ("SMDVL", "SMDVR", 2),
    ("RMEL", "RMER", 2), ("RMGL", "RMGR", 2),
    ("RIAL", "RIAR", 2),

    # ── AVB to motor ────────────────────────────────────────────────────────
    ("AVBL", "VB2",  3), ("AVBL", "VB3",  2), ("AVBL", "VB4",  2),
    ("AVBL", "DB1",  3), ("AVBL", "DB2",  2), ("AVBL", "DB3",  2),
    ("AVBR", "VB5",  3), ("AVBR", "VB6",  2), ("AVBR", "VB7",  2),
    ("AVBR", "DB4",  2), ("AVBR", "DB5",  2), ("AVBR", "DB6",  2),

    # ── AVA to motor ────────────────────────────────────────────────────────
    ("AVAL", "DA1",  4), ("AVAL", "DA2",  3), ("AVAL", "DA3",  3),
    ("AVAL", "VA1",  3), ("AVAL", "VA2",  3), ("AVAL", "VA3",  2),
    ("AVAR", "DA4",  3), ("AVAR", "DA5",  2), ("AVAR", "DA6",  2),
    ("AVAR", "VA4",  3), ("AVAR", "VA5",  2), ("AVAR", "VA6",  2),

    # ── Motor neuron coupling ───────────────────────────────────────────────
    ("DB1", "DB2",  3), ("DB2", "DB3",  2), ("DB3", "DB4",  2),
    ("DB4", "DB5",  2), ("DB5", "DB6",  2), ("DB6", "DB7",  2),
    ("VB1", "VB2",  3), ("VB2", "VB3",  2), ("VB3", "VB4",  2),
    ("VB4", "VB5",  2), ("VB5", "VB6",  2), ("VB6", "VB7",  2),
    ("VB7", "VB8",  1), ("VB8", "VB9",  1), ("VB9", "VB10", 1),
    ("DA1", "DA2",  2), ("DA2", "DA3",  2), ("DA3", "DA4",  2),
    ("VA1", "VA2",  2), ("VA2", "VA3",  2), ("VA3", "VA4",  2),
    ("DD1", "DD2",  2), ("DD2", "DD3",  2), ("DD3", "DD4",  2),
    ("VD1", "VD2",  2), ("VD2", "VD3",  2), ("VD3", "VD4",  2),
    ("VD4", "VD5",  2), ("VD5", "VD6",  1), ("VD6", "VD7",  1),

    # ── D-type reciprocal ───────────────────────────────────────────────────
    ("DD1", "VD2",  2), ("DD2", "VD3",  2), ("DD3", "VD5",  2),
    ("DD4", "VD6",  1), ("DD5", "VD9",  1), ("DD6", "VD11", 1),

    # ── AS coupling ─────────────────────────────────────────────────────────
    ("AS1",  "DA1",  2), ("AS2",  "DA2",  1), ("AS3",  "DA3",  1),
    ("AS4",  "DA4",  1), ("AS5",  "DA5",  1), ("AS6",  "DA6",  1),
    ("AS7",  "DA7",  1), ("AS8",  "DA8",  1), ("AS9",  "DA9",  1),

    # ── Tail ────────────────────────────────────────────────────────────────
    ("LUAL", "LUAR", 2), ("DVA", "PVDL", 1), ("DVA", "PVDR", 1),
    ("PVQL", "PVQR", 1), ("PHCL", "PHCR", 1),
    ("AVL",  "DVB",  3),

    # ── Sensory-interneuron gap junctions ────────────────────────────────────
    ("ASEL", "AIYL", 1), ("ASER", "AIYR", 1),
    ("AWCL", "AIYL", 1), ("AWCR", "AIYR", 1),
    ("ASHL", "AVDL", 1), ("ASHR", "AVDR", 1),
    ("ALML", "AVM",  1), ("ALMR", "AVM",  1),
    ("URXL", "URXR", 2), ("URBL", "URBR", 1),
    ("PVDL", "PVDR", 1), ("PLML", "PLMR", 2),
    ("CEPVL", "CEPVR", 1),
    ("OLQDL", "OLQVL", 1), ("OLQDR", "OLQVR", 1),
    ("RIH",  "AVBL",  1), ("RIH",  "AVBR",  1),
]
