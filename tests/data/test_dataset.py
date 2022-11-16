# import pytest
import torch
from torch_geometric.loader import DataLoader


# @pytest.mark.skip
def test_rcdb_dataset(rcsb_dataset):
    loader = DataLoader(rcsb_dataset, batch_size=2, shuffle=False)
    # assert len(loader) == 1
    for item in loader:
        assert "edge_v" in item

        # Ensure that 'residue_index' is being appropriately offset
        # by the dataloader batching, such that we can aggregate on
        # this term to reduce representations to the residue level.
        assert (
            torch.unique(item.residue_index).shape[0]
            == item.residue_index.shape[0] // 3
        )

        assert item.residue_index.max() == item.num_nodes // 3 - 1
