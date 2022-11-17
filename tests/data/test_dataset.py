# import pytest
import shutil
from pathlib import Path

import pytest
import torch
from torch_geometric.loader import DataLoader

from torch_gvp.data.rcsb_dataset import RCSBDataset, size_filter
from torch_gvp.data.transforms import create_gvp_transformer_stack


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


def test_dask(tmp_path_factory):
    pytest.importorskip("dask.dataframe")

    fn: Path = tmp_path_factory.mktemp("data_dask")
    Path(fn, "raw").mkdir(exist_ok=True)
    shutil.copy(
        Path(Path(__file__).parent.parent, "sample_tiny.parquet"),
        Path(fn, "raw", "data.parquet"),
    )

    RCSBDataset(
        fn.as_posix(),
        transform=create_gvp_transformer_stack(jitter=0.02, residue_mask_prob=0.35),
        pre_filter=size_filter,
        num_processes=None,
    )
