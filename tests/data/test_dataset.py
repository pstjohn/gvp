# import pytest
from torch_geometric.loader import DataLoader


# @pytest.mark.skip
def test_rcdb_dataset(rcsb_dataset):
    loader = DataLoader(rcsb_dataset, batch_size=2)
    assert len(loader) == 3
    for item in loader:
        assert "edge_v" in item
