import gzip
import io
from pathlib import Path
from typing import Union

from biotite.structure import AtomArrayStack, filter_backbone
from biotite.structure.io.mmtf import MMTFFile, get_structure  # type: ignore

from gvp.data.types import Atom, Protein


def load_mmtf_file(filename: Union[str, Path]) -> AtomArrayStack:
    filename = Path(filename)
    assert ".mmtf" in filename.suffixes, "must be passed an mmtf file"
    open_fn = gzip.open if ".gz" in filename.suffixes else open
    with open_fn(filename, "rb") as f:
        return load_bytes(f.read(), compressed=False)


def load_bytes(data: bytes, compressed=True) -> AtomArrayStack:
    if compressed:
        data = gzip.decompress(data)
    return get_structure(MMTFFile.read(io.BytesIO(data)))


def as_protein(structure: AtomArrayStack) -> Protein:
    protein = []
    for item in structure:
        for atom in item[filter_backbone(item)]:  # type: ignore
            protein.append(
                Atom(
                    pos=atom.coord.tolist(),
                    atom_type=atom.atom_name,
                    residue_type=atom.res_name,
                    residue_index=atom.res_id,
                )
            )

    return protein
