import gzip
import io
from pathlib import Path
from typing import Union

import torch  # type: ignore
import torch_geometric.data
from biotite.structure import AtomArray, AtomArrayStack, filter_backbone
from biotite.structure.io.mmtf import MMTFFile, get_structure  # type: ignore

from gvp.data.types import AtomType, ResidueType


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


def convert_to_pyg(atom_stack: AtomArrayStack) -> torch_geometric.data.Data:
    """Converts an AtomGroup to an initial PyG Data object with fields

    pos: [num_atoms, 3], float
        the cartesian coordinates of each atom in the AtomGroup
    atom_types: [num_atoms,], int
        An array mapping each atom to its corresponding N, Ca, C, O type (as int)
    residue_types: [num_atoms,], int
        An array mapping the atom's residue to an integer representation of its amino
        acid
    residue_indices: [num_atoms,], int
        An array mapping each atom to it's corresponding amino acid's position in the
        overall protein chain

    Parameters
    ----------
    atoms : prody.atomic.atomgroup.AtomGroup
        A prody representation of the protein backbone, likely from `parse_pdb`

    Returns
    -------
    torch_geometric.data.Data
        A pyg graph data object
    """

    # For proteins where we have an NMR ensemble, we just take the first chain
    atoms: AtomArray = atom_stack[0]  # type: ignore
    assert atoms.coord is not None  # for typing, not sure we'd expect this to be None

    is_backbone = filter_backbone(atoms)
    pos = torch.tensor(atoms.coord[is_backbone], dtype=torch.float32)

    atom_types = torch.tensor(
        [AtomType[name].value for name in atoms._annot["atom_name"][is_backbone]],
        dtype=torch.int32,
    )

    residue_types = torch.tensor(
        [ResidueType[name].value for name in atoms._annot["res_name"][is_backbone]],
        dtype=torch.int32,
    )

    residue_indices = torch.tensor(
        atoms._annot["res_id"][is_backbone], dtype=torch.int32
    )

    return torch_geometric.data.Data(
        pos=pos,
        atom_types=atom_types,
        residue_types=residue_types,
        residue_indices=residue_indices,
    )


# def as_protein(structure: AtomArrayStack) -> Protein:
#     protein = []
#     for item in structure:
#         for atom in item[filter_backbone(item)]:  # type: ignore
#             protein.append(
#                 Atom(
#                     pos=atom.coord.tolist(),
#                     atom_type=atom.atom_name,
#                     residue_type=atom.res_name,
#                     residue_index=atom.res_id,
#                 )
#             )

#     return protein
