from pathlib import Path

import pytest

from torch_gvp.data.biotite import convert_to_pyg, load_mmtf_file


@pytest.fixture
def biotite_stack():
    filename = Path(Path(__file__).parent, "4HHB.mmtf.gz").absolute()
    return load_mmtf_file(filename)


def test_convert_to_pyg(biotite_stack):
    data = convert_to_pyg(biotite_stack)
    assert data.pos.shape[1] == 3
    assert data.atom_type.shape[0] == data.pos.shape[0]
    assert data.atom_type.max() == 3
    assert data.residue_type.max() <= 20
