from gvp.data.residue import ResidueType


def test_residue():

    assert ResidueType["ALA"].value == 1
    assert ResidueType["VAL"].value == 20
    assert ResidueType["xxx"].value == ResidueType["UNK"].value
