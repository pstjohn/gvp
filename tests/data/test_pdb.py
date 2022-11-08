from pathlib import Path

import numpy as np
import pytest

from gvp.data.pdb import get_dihedral_features, parse_pdb


@pytest.fixture
def pdb():
    pdb_file = Path(Path(__file__).parent, "5zck.pdb.gz").absolute().as_posix()
    return parse_pdb(pdb_file)


@pytest.fixture
def residue(pdb):
    return pdb["A", 1]


def test_parse(pdb):
    assert pdb.numResidues() == 4
    assert pdb.numAtoms() == 4 * pdb.numResidues()


def test_get_dihedral_features(residue):
    dihedrals = get_dihedral_features(residue)
    assert len(dihedrals) == 6
    assert dihedrals.max() < 1.0
    assert np.isfinite(dihedrals).all()
