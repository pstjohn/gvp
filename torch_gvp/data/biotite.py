import gzip
import io
from pathlib import Path
from typing import Union

import torch
import torch_geometric.data
from biotite.structure import (
    AtomArray,
    AtomArrayStack,
    check_res_id_continuity,
    filter_backbone,
)
from biotite.structure.io.mmtf import MMTFFile, get_structure  # type: ignore

from torch_gvp.data.residue import AtomType, ResidueType


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


class ResidueData(torch_geometric.data.Data):
    """We need to provide a custom data object so we can define how the `residue_index`
    value is incremented. Rather than being offset by the total number of nodes, this
    should be offset by the total number of residues (nodes // 3, assuming 3 backbone
    atoms per residue).
    """

    def __inc__(self, key, value, *args, **kwargs):
        if key == "residue_index":
            return self.num_nodes // 3  # type: ignore
        return super().__inc__(key, value, *args, **kwargs)


def convert_to_pyg(atom_stack: AtomArrayStack) -> torch_geometric.data.Data:
    """Converts an AtomGroup to an initial PyG Data object with fields

    pos: [num_atoms, 3], float
        the cartesian coordinates of each atom in the AtomGroup
    atom_type: [num_atoms,], int
        An array mapping each atom to its corresponding N, Ca, C, O type (as int)
    residue_type: [num_atoms,], int
        An array mapping the atom's residue to an integer representation of its amino
        acid
    residue_index: [num_atoms,], int
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

    atom_type = torch.tensor(
        [AtomType[name].value for name in atoms._annot["atom_name"][is_backbone]],
        dtype=torch.int64,
    )

    assert is_backbone.sum() % 3 == 0, "number of backbone atoms must be divisible by 3"
    num_residues = is_backbone.sum() // 3
    assert torch.all(
        atom_type == (1 + torch.arange(3)).repeat(num_residues)
    ), "residue backbone atoms were not in the expected elemental order"

    residue_type = torch.tensor(
        [ResidueType[name].value for name in atoms._annot["res_name"][is_backbone]],
        dtype=torch.int64,
    )

    residue_index = torch.arange(0, num_residues, dtype=torch.int64).repeat_interleave(
        3
    )

    residue_discont = torch.as_tensor(
        check_res_id_continuity(atoms[is_backbone]), dtype=torch.int64
    )

    return ResidueData(
        pos=pos,
        atom_type=atom_type,
        residue_type=residue_type,
        residue_index=residue_index,
        residue_discont=residue_discont,
    )
