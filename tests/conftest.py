import shutil
from pathlib import Path

import pytest
import torch
from scipy.spatial.transform import Rotation
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

from torch_gvp.data.biotite import convert_to_pyg, load_mmtf_file
from torch_gvp.data.rcsb_dataset import RCSBDataset, size_filter
from torch_gvp.data.transforms import create_gvp_transformer_stack


@pytest.fixture(scope="session")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="function")
def rotation(device):
    return torch.as_tensor(
        Rotation.random().as_matrix(), dtype=torch.float32, device=device
    )


@pytest.fixture
def prot_data() -> Data:
    filename = Path(Path(__file__).parent, "data/4HHB.mmtf.gz").absolute()
    return convert_to_pyg(load_mmtf_file(filename))


@pytest.fixture(scope="session")
def data_path(tmp_path_factory):
    fn: Path = tmp_path_factory.mktemp("data")
    Path(fn, "raw").mkdir(exist_ok=True)
    shutil.copy(
        Path(Path(__file__).parent, "sample_tiny.parquet"),
        Path(fn, "raw", "data.parquet"),
    )
    yield fn.as_posix()


@pytest.fixture(scope="session")
def rcsb_dataset(data_path: str) -> Dataset:
    return RCSBDataset(
        data_path,
        transform=create_gvp_transformer_stack(jitter=0.02, residue_mask_prob=0.35),
        pre_filter=size_filter,
        num_processes=1,
    )


@pytest.fixture(scope="session")
def rcsb_loader(rcsb_dataset: Dataset) -> DataLoader:
    return DataLoader(rcsb_dataset, batch_size=2)
