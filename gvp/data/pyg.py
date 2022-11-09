import torch
import torch_geometric.data

from gvp.data.types import Protein


def convert_to_pyg(atoms: Protein) -> torch_geometric.data.Data:
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

    pos = torch.tensor([atom.pos for atom in atoms], dtype=torch.float32)
    atom_types = torch.tensor([atom.atom_type for atom in atoms], dtype=torch.int32)
    residue_types = torch.tensor(
        [atom.residue_type for atom in atoms], dtype=torch.int32
    )
    residue_indices = torch.tensor(
        [atom.residue_index for atom in atoms], dtype=torch.int32
    )

    return torch_geometric.data.Data(
        pos=pos,
        atom_types=atom_types,
        residue_types=residue_types,
        residue_indices=residue_indices,
    )
