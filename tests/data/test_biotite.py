def test_convert_to_pyg(prot_data):
    assert prot_data.pos.shape[1] == 3
    assert prot_data.atom_type.shape[0] == prot_data.pos.shape[0]
    assert prot_data.atom_type.max() == 3
    assert prot_data.residue_type.max() <= 20
    assert "residue_index" in prot_data
    assert "residue_indices" not in prot_data
