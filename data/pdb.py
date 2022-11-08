from typing import Callable, Tuple

import numpy as np
import prody
import torch_geometric.data

prody.confProDy(verbosity="error")


def parse_pdb(filename: str) -> prody.atomic.AtomGroup:
    """Parse the backbone protein structure with ProDy"""
    pdb = prody.parsePDB(filename, subset="bb")
    assert type(pdb) is prody.atomic.AtomGroup, f"Issues parsing {filename}"
    return pdb


def get_dihedral_features(residue: prody.atomic.residue.Residue) -> np.ndarray:
    """Embed dihedral features following the scheme from Generative Models for
    Graph-Based Protein Design by John Ingraham, Vikas Garg, Regina Barzilay and Tommi
    Jaakkola, NeurIPS 2019."""

    def try_dihedral(
        residue: prody.atomic.residue.Residue, angle_fn: Callable
    ) -> Tuple[float, float]:
        try:
            angle = angle_fn(residue)
            return np.sin(angle), np.cos(angle)
        except AttributeError:
            return 0.0, 0.0

    angles = [prody.calcPhi, prody.calcPsi, prody.calcOmega]
    return np.array([val for angle in angles for val in try_dihedral(residue, angle)])


def pdb_to_graph(atoms: prody.atomic.atomgroup.AtomGroup) -> torch_geometric.data.Data:

    coords = np.stack([residue.getCoords() for residue in atoms.iterResidues()])
    dihedrals = np.stack(
        [get_dihedral_features(residue) for residue in atoms.iterResidues()]
    )

    return torch_geometric.data.Data()
