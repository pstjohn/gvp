from enum import Enum, auto
from typing import List

from pydantic import BaseModel


class LookupEnum(Enum):
    """https://github.com/pydantic/pydantic/issues/598#issuecomment-503032706"""

    @classmethod
    def __get_validators__(cls):
        cls.lookup = {v: k.value for v, k in cls.__members__.items()}
        yield cls.validate

    @classmethod
    def validate(cls, v):
        try:
            return cls.lookup[v]
        except KeyError:
            raise ValueError("invalid value")


class AtomType(LookupEnum):
    N = 1
    CA = 2
    C = 3
    O = 4  # noqa: E741


class ResidueType(LookupEnum):
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


class Atom(BaseModel):
    pos: List[float]
    atom_type: AtomType
    residue_type: ResidueType
    residue_index: int

    class Config:
        use_enum_values = True


Protein = List[Atom]
