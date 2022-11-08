from typing import Callable, Tuple

import numpy as np
import prody
import torch
import torch_geometric.data

from gvp.data.types import AtomGroup, AtomTypes, ResidueTypes

prody.confProDy(verbosity="error")


def parse_pdb(filename: str) -> prody.atomic.AtomGroup:
    """Parse the backbone protein structure with ProDy"""
    pdb = prody.parsePDB(filename, subset="bb")
    assert type(pdb) is prody.atomic.AtomGroup, f"Issues parsing {filename}"
    return pdb


def get_dihedral_features(residue: prody.atomic.residue.Residue) -> np.ndarray:
    """Embed dihedral features following the scheme from Generative Models for
    Graph-Based Protein Design by John Ingraham, Vikas Garg, Regina Barzilay and Tommi
    Jaakkola, NeurIPS 2019.

    Not positive we want to use this?
    """

    def try_dihedral(
        residue: prody.atomic.residue.Residue, angle_fn: Callable
    ) -> Tuple[float, float]:
        try:
            angle = angle_fn(residue)
            return np.sin(angle), np.cos(angle)
        except ValueError:
            return 0.0, 0.0

    angles = [prody.calcPhi, prody.calcPsi, prody.calcOmega]
    return np.array([val for angle in angles for val in try_dihedral(residue, angle)])


def pdb_to_graph(atoms: AtomGroup) -> torch_geometric.data.Data:
    """An opinionated conversion of a protein backbone structure to a graph.



    Parameters
    ----------
    atoms : prody.atomic.atomgroup.AtomGroup
        A prody representation of the protein backbone, likely from `parse_pdb`

    Returns
    -------
    torch_geometric.data.Data
        A pyg graph data object containing the completed graph
    """

    pos = torch.tensor(np.stack([atom.getCoords() for atom in atoms.iterAtoms()]))
    atom_types = torch.tensor(
        [AtomTypes[atom].value for atom in atoms.getNames()], dtype=torch.int32
    )
    residue_types = torch.tensor(
        [ResidueTypes[atom].value for atom in atoms.getResnames()], dtype=torch.int32
    )
    residue_indices = torch.tensor(
        atoms.getResindices() + 1, dtype=torch.int32
    )  # offset by one to start at 1

    return torch_geometric.data.Data(
        residue_types=residue_types,
        residue_indices=residue_indices,
        atom_types=atom_types,
        pos=pos,
    )
