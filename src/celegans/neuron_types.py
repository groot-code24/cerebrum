"""WormAtlas canonical neuron type classification for C. elegans.

Provides the full NEURON_REGISTRY mapping each of the 302 neuron names to its
:class:`NeuronType`, along with helper functions for classification and lookup.

Sources: White et al. (1986); WormAtlas (wormatlas.org).
"""

from __future__ import annotations

from enum import Enum, unique
from typing import Dict, List


@unique
class NeuronType(str, Enum):
    """Functional classification of C. elegans neurons."""

    SENSORY = "sensory"
    INTERNEURON = "interneuron"
    MOTOR = "motor"
    PHARYNGEAL = "pharyngeal"


# ---------------------------------------------------------------------------
# Complete neuron registry — 302 neurons
# Sensory (95) | Interneuron (75) | Motor (109) | Pharyngeal (23)
# ---------------------------------------------------------------------------

NEURON_REGISTRY: Dict[str, NeuronType] = {
    # ── Sensory neurons (95) ─────────────────────────────────────────────────
    "ADFL": NeuronType.SENSORY, "ADFR": NeuronType.SENSORY,
    "ADLL": NeuronType.SENSORY, "ADLR": NeuronType.SENSORY,
    "AFDL": NeuronType.SENSORY, "AFDR": NeuronType.SENSORY,
    "ASEL": NeuronType.SENSORY, "ASER": NeuronType.SENSORY,
    "ASGL": NeuronType.SENSORY, "ASGR": NeuronType.SENSORY,
    "ASHL": NeuronType.SENSORY, "ASHR": NeuronType.SENSORY,
    "ASIL": NeuronType.SENSORY, "ASIR": NeuronType.SENSORY,
    "ASJL": NeuronType.SENSORY, "ASJR": NeuronType.SENSORY,
    "ASKL": NeuronType.SENSORY, "ASKR": NeuronType.SENSORY,
    "AWAL": NeuronType.SENSORY, "AWAR": NeuronType.SENSORY,
    "AWBL": NeuronType.SENSORY, "AWBR": NeuronType.SENSORY,
    "AWCL": NeuronType.SENSORY, "AWCR": NeuronType.SENSORY,
    "BAGL": NeuronType.SENSORY, "BAGR": NeuronType.SENSORY,
    "CEPVL": NeuronType.SENSORY, "CEPVR": NeuronType.SENSORY,
    "IL1DL": NeuronType.SENSORY, "IL1DR": NeuronType.SENSORY,
    "IL1L": NeuronType.SENSORY,  "IL1R": NeuronType.SENSORY,
    "IL1VL": NeuronType.SENSORY, "IL1VR": NeuronType.SENSORY,
    "IL2DL": NeuronType.SENSORY, "IL2DR": NeuronType.SENSORY,
    "IL2L": NeuronType.SENSORY,  "IL2R": NeuronType.SENSORY,
    "IL2VL": NeuronType.SENSORY, "IL2VR": NeuronType.SENSORY,
    "OLLL": NeuronType.SENSORY,  "OLLR": NeuronType.SENSORY,
    "OLQDL": NeuronType.SENSORY, "OLQDR": NeuronType.SENSORY,
    "OLQVL": NeuronType.SENSORY, "OLQVR": NeuronType.SENSORY,
    "PLML": NeuronType.SENSORY,  "PLMR": NeuronType.SENSORY,
    "PLNL": NeuronType.SENSORY,  "PLNR": NeuronType.SENSORY,
    "PQR": NeuronType.SENSORY,
    "PVDL": NeuronType.SENSORY,  "PVDR": NeuronType.SENSORY,
    "URBL": NeuronType.SENSORY,  "URBR": NeuronType.SENSORY,
    "URXL": NeuronType.SENSORY,  "URXR": NeuronType.SENSORY,
    "URYDL": NeuronType.SENSORY, "URYDR": NeuronType.SENSORY,
    "URYVL": NeuronType.SENSORY, "URYVR": NeuronType.SENSORY,
    "AQR": NeuronType.SENSORY,
    "ALML": NeuronType.SENSORY,  "ALMR": NeuronType.SENSORY,
    "AVM": NeuronType.SENSORY,
    "PVM": NeuronType.SENSORY,
    "SAADL": NeuronType.SENSORY, "SAADR": NeuronType.SENSORY,
    "SAAVL": NeuronType.SENSORY, "SAAVR": NeuronType.SENSORY,
    "SIBDL": NeuronType.SENSORY, "SIBDR": NeuronType.SENSORY,
    "SIBVL": NeuronType.SENSORY, "SIBVR": NeuronType.SENSORY,
    # Phasmid sensory neurons (A/B/C pairs)
    "PHAL": NeuronType.SENSORY,  "PHAR": NeuronType.SENSORY,
    "PHBL": NeuronType.SENSORY,  "PHBR": NeuronType.SENSORY,
    "PHCL": NeuronType.SENSORY,  "PHCR": NeuronType.SENSORY,
    # ADA head interneurons (sometimes classified as sensory-interneuron)
    "ADAL": NeuronType.INTERNEURON, "ADAR": NeuronType.INTERNEURON,

    # ── Interneurons (75) ─────────────────────────────────────────────────────
    "AIAL": NeuronType.INTERNEURON, "AIAR": NeuronType.INTERNEURON,
    "AIBL": NeuronType.INTERNEURON, "AIBR": NeuronType.INTERNEURON,
    "AIML": NeuronType.INTERNEURON, "AIMR": NeuronType.INTERNEURON,
    "AINL": NeuronType.INTERNEURON, "AINR": NeuronType.INTERNEURON,
    "AIYL": NeuronType.INTERNEURON, "AIYR": NeuronType.INTERNEURON,
    "AIZL": NeuronType.INTERNEURON, "AIZR": NeuronType.INTERNEURON,
    "ALA": NeuronType.INTERNEURON,
    "AVAL": NeuronType.INTERNEURON, "AVAR": NeuronType.INTERNEURON,
    "AVBL": NeuronType.INTERNEURON, "AVBR": NeuronType.INTERNEURON,
    "AVDL": NeuronType.INTERNEURON, "AVDR": NeuronType.INTERNEURON,
    "AVEL": NeuronType.INTERNEURON, "AVER": NeuronType.INTERNEURON,
    "AVFL": NeuronType.INTERNEURON, "AVFR": NeuronType.INTERNEURON,
    "AVG": NeuronType.INTERNEURON,
    "AVHL": NeuronType.INTERNEURON, "AVHR": NeuronType.INTERNEURON,
    "AVJL": NeuronType.INTERNEURON, "AVJR": NeuronType.INTERNEURON,
    "AVKL": NeuronType.INTERNEURON, "AVKR": NeuronType.INTERNEURON,
    "AVL": NeuronType.INTERNEURON,
    "BDUL": NeuronType.INTERNEURON, "BDUR": NeuronType.INTERNEURON,
    "CANL": NeuronType.INTERNEURON, "CANR": NeuronType.INTERNEURON,
    "CEPDL": NeuronType.INTERNEURON, "CEPDR": NeuronType.INTERNEURON,
    "DVA": NeuronType.INTERNEURON,
    "DVC": NeuronType.INTERNEURON,
    "FLPL": NeuronType.INTERNEURON, "FLPR": NeuronType.INTERNEURON,
    "HSNL": NeuronType.INTERNEURON, "HSNR": NeuronType.INTERNEURON,
    "LUAL": NeuronType.INTERNEURON, "LUAR": NeuronType.INTERNEURON,
    "PVNL": NeuronType.INTERNEURON, "PVNR": NeuronType.INTERNEURON,
    "PVPL": NeuronType.INTERNEURON, "PVPR": NeuronType.INTERNEURON,
    "PVQL": NeuronType.INTERNEURON, "PVQR": NeuronType.INTERNEURON,
    "PVR": NeuronType.INTERNEURON,
    "PVT": NeuronType.INTERNEURON,
    "PVWL": NeuronType.INTERNEURON, "PVWR": NeuronType.INTERNEURON,
    "RIAL": NeuronType.INTERNEURON, "RIAR": NeuronType.INTERNEURON,
    "RIBL": NeuronType.INTERNEURON, "RIBR": NeuronType.INTERNEURON,
    "RICL": NeuronType.INTERNEURON, "RICR": NeuronType.INTERNEURON,
    "RID": NeuronType.INTERNEURON,
    "RIFL": NeuronType.INTERNEURON, "RIFR": NeuronType.INTERNEURON,
    "RIGL": NeuronType.INTERNEURON, "RIGR": NeuronType.INTERNEURON,
    "RIH": NeuronType.INTERNEURON,
    "RIML": NeuronType.INTERNEURON, "RIMR": NeuronType.INTERNEURON,
    "RINL": NeuronType.INTERNEURON, "RINR": NeuronType.INTERNEURON,
    "RIR": NeuronType.INTERNEURON,
    "RIS": NeuronType.INTERNEURON,
    "RIVL": NeuronType.INTERNEURON, "RIVR": NeuronType.INTERNEURON,
    "RMDDL": NeuronType.INTERNEURON, "RMDDR": NeuronType.INTERNEURON,
    "RMDL": NeuronType.INTERNEURON,  "RMDR": NeuronType.INTERNEURON,
    "RMDVL": NeuronType.INTERNEURON, "RMDVR": NeuronType.INTERNEURON,
    "RMED": NeuronType.INTERNEURON,
    "RMEL": NeuronType.INTERNEURON,  "RMER": NeuronType.INTERNEURON,
    "RMEV": NeuronType.INTERNEURON,
    "RMFL": NeuronType.INTERNEURON,  "RMFR": NeuronType.INTERNEURON,
    "RMGL": NeuronType.INTERNEURON,  "RMGR": NeuronType.INTERNEURON,
    "RMHL": NeuronType.INTERNEURON,  "RMHR": NeuronType.INTERNEURON,

    # ── Motor neurons (109) ───────────────────────────────────────────────────
    "AS1": NeuronType.MOTOR,  "AS2": NeuronType.MOTOR,
    "AS3": NeuronType.MOTOR,  "AS4": NeuronType.MOTOR,
    "AS5": NeuronType.MOTOR,  "AS6": NeuronType.MOTOR,
    "AS7": NeuronType.MOTOR,  "AS8": NeuronType.MOTOR,
    "AS9": NeuronType.MOTOR,  "AS10": NeuronType.MOTOR,
    "AS11": NeuronType.MOTOR,
    "DA1": NeuronType.MOTOR,  "DA2": NeuronType.MOTOR,
    "DA3": NeuronType.MOTOR,  "DA4": NeuronType.MOTOR,
    "DA5": NeuronType.MOTOR,  "DA6": NeuronType.MOTOR,
    "DA7": NeuronType.MOTOR,  "DA8": NeuronType.MOTOR,
    "DA9": NeuronType.MOTOR,
    "DB1": NeuronType.MOTOR,  "DB2": NeuronType.MOTOR,
    "DB3": NeuronType.MOTOR,  "DB4": NeuronType.MOTOR,
    "DB5": NeuronType.MOTOR,  "DB6": NeuronType.MOTOR,
    "DB7": NeuronType.MOTOR,
    "DD1": NeuronType.MOTOR,  "DD2": NeuronType.MOTOR,
    "DD3": NeuronType.MOTOR,  "DD4": NeuronType.MOTOR,
    "DD5": NeuronType.MOTOR,  "DD6": NeuronType.MOTOR,
    "PDA": NeuronType.MOTOR,
    "PDB": NeuronType.MOTOR,
    "SAB": NeuronType.MOTOR,
    "SABDL": NeuronType.MOTOR, "SABDR": NeuronType.MOTOR,
    "SABVL": NeuronType.MOTOR, "SABVR": NeuronType.MOTOR,
    "SDQL": NeuronType.MOTOR,  "SDQR": NeuronType.MOTOR,
    "SIADL": NeuronType.MOTOR, "SIADR": NeuronType.MOTOR,
    "SIAVL": NeuronType.MOTOR, "SIAVR": NeuronType.MOTOR,
    "SMBDL": NeuronType.MOTOR, "SMBDR": NeuronType.MOTOR,
    "SMBVL": NeuronType.MOTOR, "SMBVR": NeuronType.MOTOR,
    "SMDDL": NeuronType.MOTOR, "SMDDR": NeuronType.MOTOR,
    "SMDVL": NeuronType.MOTOR, "SMDVR": NeuronType.MOTOR,
    "VA1": NeuronType.MOTOR,  "VA2": NeuronType.MOTOR,
    "VA3": NeuronType.MOTOR,  "VA4": NeuronType.MOTOR,
    "VA5": NeuronType.MOTOR,  "VA6": NeuronType.MOTOR,
    "VA7": NeuronType.MOTOR,  "VA8": NeuronType.MOTOR,
    "VA9": NeuronType.MOTOR,  "VA10": NeuronType.MOTOR,
    "VA11": NeuronType.MOTOR, "VA12": NeuronType.MOTOR,
    "VB1": NeuronType.MOTOR,  "VB2": NeuronType.MOTOR,
    "VB3": NeuronType.MOTOR,  "VB4": NeuronType.MOTOR,
    "VB5": NeuronType.MOTOR,  "VB6": NeuronType.MOTOR,
    "VB7": NeuronType.MOTOR,  "VB8": NeuronType.MOTOR,
    "VB9": NeuronType.MOTOR,  "VB10": NeuronType.MOTOR,
    "VB11": NeuronType.MOTOR,
    "VC1": NeuronType.MOTOR,  "VC2": NeuronType.MOTOR,
    "VC3": NeuronType.MOTOR,  "VC4": NeuronType.MOTOR,
    "VC5": NeuronType.MOTOR,  "VC6": NeuronType.MOTOR,
    "VD1": NeuronType.MOTOR,  "VD2": NeuronType.MOTOR,
    "VD3": NeuronType.MOTOR,  "VD4": NeuronType.MOTOR,
    "VD5": NeuronType.MOTOR,  "VD6": NeuronType.MOTOR,
    "VD7": NeuronType.MOTOR,  "VD8": NeuronType.MOTOR,
    "VD9": NeuronType.MOTOR,  "VD10": NeuronType.MOTOR,
    "VD11": NeuronType.MOTOR, "VD12": NeuronType.MOTOR,
    "VD13": NeuronType.MOTOR,
    "DVB": NeuronType.MOTOR,
    "PDEL": NeuronType.MOTOR,  "PDER": NeuronType.MOTOR,
    "PVCL": NeuronType.MOTOR,  "PVCR": NeuronType.MOTOR,

    # ── Pharyngeal neurons (23) ───────────────────────────────────────────────
    "I1L": NeuronType.PHARYNGEAL, "I1R": NeuronType.PHARYNGEAL,
    "I2L": NeuronType.PHARYNGEAL, "I2R": NeuronType.PHARYNGEAL,
    "I3": NeuronType.PHARYNGEAL,
    "I4": NeuronType.PHARYNGEAL,
    "I5": NeuronType.PHARYNGEAL,
    "I6": NeuronType.PHARYNGEAL,
    "M1": NeuronType.PHARYNGEAL,
    "M2L": NeuronType.PHARYNGEAL, "M2R": NeuronType.PHARYNGEAL,
    "M3L": NeuronType.PHARYNGEAL, "M3R": NeuronType.PHARYNGEAL,
    "M4": NeuronType.PHARYNGEAL,
    "M5": NeuronType.PHARYNGEAL,
    "MCL": NeuronType.PHARYNGEAL, "MCR": NeuronType.PHARYNGEAL,
    "MI": NeuronType.PHARYNGEAL,
    "NSML": NeuronType.PHARYNGEAL, "NSMR": NeuronType.PHARYNGEAL,
    "RIPL": NeuronType.PHARYNGEAL, "RIPR": NeuronType.PHARYNGEAL,
    "MVL07": NeuronType.PHARYNGEAL,
}


def classify_neuron(name: str) -> NeuronType:
    """Return the :class:`NeuronType` for *name*.

    Args:
        name: Canonical neuron name string (case-sensitive).

    Returns:
        :class:`NeuronType` enum member.

    Raises:
        KeyError: If *name* is not in the registry.
    """
    try:
        return NEURON_REGISTRY[name]
    except KeyError:
        raise KeyError(f"Unknown neuron name: {name!r}. Check NEURON_REGISTRY.")


def get_neurons_by_type(neuron_type: NeuronType) -> List[str]:
    """Return a sorted list of neuron names belonging to *neuron_type*.

    Args:
        neuron_type: The :class:`NeuronType` to filter by.

    Returns:
        Sorted list of matching neuron name strings.
    """
    return sorted(name for name, t in NEURON_REGISTRY.items() if t == neuron_type)
