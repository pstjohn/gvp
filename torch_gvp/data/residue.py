from enum import Enum, EnumMeta, IntEnum, auto


class DefaultEnumMeta(EnumMeta):
    def __getitem__(cls, name: str):
        try:
            return super().__getitem__(name)  # type: ignore
        except KeyError:
            return cls.UNK  # type: ignore


class AtomType(IntEnum):
    N = 1
    CA = 2
    C = 3


class ResidueType(Enum, metaclass=DefaultEnumMeta):
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
    UNK = auto()
    MASK = auto()
