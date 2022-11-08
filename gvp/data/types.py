from enum import Enum, auto
from typing import Iterable, Protocol

import numpy as np


class Atom(Protocol):
    def getCoords(self) -> np.ndarray:
        ...


class AtomGroup(Protocol):
    def iterAtoms(self) -> Iterable[Atom]:
        ...

    def getNames(self) -> Iterable[str]:
        ...

    def getResnames(self) -> Iterable[str]:
        ...

    def getResindices(self) -> np.ndarray:
        ...


class AtomTypes(Enum):
    N = 1
    CA = 2
    C = 3
    O = 4  # noqa: E741


class ResidueTypes(Enum):
    ALA = auto()
    ARG = auto()
    ASN = auto()
    ASP = auto()
    CYS = auto()
    GLN = auto()
    GLU = auto()
    GLY = auto()
    HIS = auto()
    ILE = auto()
    LEU = auto()
    LYS = auto()
    MET = auto()
    PHE = auto()
    PRO = auto()
    SER = auto()
    THR = auto()
    TRP = auto()
    TYR = auto()
    VAL = auto()
