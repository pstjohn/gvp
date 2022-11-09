from pathlib import Path

import pytest

from gvp.data.biotite import as_protein, load_mmtf_file
from gvp.data.pyg import convert_to_pyg


@pytest.fixture
def biotite_stack():
    filename = Path(Path(__file__).parent, "4HHB.mmtf.gz").absolute()
    return load_mmtf_file(filename)


@pytest.fixture
def protein(biotite_stack):
    return as_protein(biotite_stack)


def test_convert_to_pyg(protein):
    data = convert_to_pyg(protein)
    assert data.pos.shape[1] == 3
    assert data.atom_types.shape[0] == data.pos.shape[0]
    assert data.atom_types.max() == 3
    assert data.residue_types.max() <= 20
