import glob
import logging
import os.path as osp
from functools import partial
from pathlib import Path
from typing import Callable, List, Optional, Union

import pandas as pd
import torch
from torch_geometric.data import Data, Dataset, download_url
from tqdm import tqdm

from torch_gvp.data.biotite import convert_to_pyg, load_bytes

try:
    import dask.dataframe as dd
except ImportError:
    dd = None

logger = logging.getLogger(__name__)


def process_item(
    row: pd.Series,
    processed_dir: Union[str, Path],
    pre_filter: Optional[Callable] = None,
    pre_transform: Optional[Callable] = None,
) -> int:
    """Filter and process a database item, saving it to `{name}.pt`

    Parameters
    ----------
    data : bytes
        A compressed bytes array of the protein structure
    name : str
        A pdb database entry for the corresponding protein
    processed_dir: str | Path
        Where to save the resulting file
    pre_filter : Optional[Callable], optional
        An optional `torch_geometric` prefilter function, by default None
    pre_transform : Optional[Callable], optional
        An optional `torch_geometric` transformation, by default None

    """
    name, protein_data = row["_1"], row["_2"]

    try:
        data = convert_to_pyg(load_bytes(protein_data, compressed=True))
    except Exception as ex:
        logger.warning(f"Issue processing {name}: {str(ex)}")
        return 0

    if pre_filter is not None and not pre_filter(data):
        return 0

    if pre_transform is not None:
        data = pre_transform(data)

    torch.save(data, Path(processed_dir, f"prot-{name}.pt"))
    return 1


def size_filter(data: Data, max_num_nodes: int = 10000, min_num_nodes: int = 1) -> bool:
    if data.num_nodes:
        return min_num_nodes <= data.num_nodes <= max_num_nodes
    else:
        return False


class RCSBDataset(Dataset):
    def __init__(
        self,
        root: Union[str, None],
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        filename: str = "data.parquet",
        num_processes: Optional[int] = None,
    ):
        """A dataset to load protein structures from MMTF data collated from RCSB.

        Parameters
        ----------
        root : Union[str, None]
            The dataset root directory. The raw data, if provided, should live
            at root/filename, otherwise a sample file will be downloaded
        transform : Optional[Callable], optional
            A post-processing transform, by default None
        pre_transform : Optional[Callable], optional
            A pre-processing transform, saved in the processed directory
            by default None
        pre_filter : Optional[Callable], optional
            A pre-filter to remove calculations before saving, by default None
        filename : str, optional
            The filename of the raw data, by default "data.parquet"
        num_processes : Optional[int], optional
            Number of parallel processes to use for multiprocessing, by default None
        """
        self._raw_filename = filename
        self._files: Optional[List] = None
        self.num_processes = num_processes
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def processed_file_names(self) -> List[str]:
        return [
            f"prot-{id}.pt" for id in ["5KJG", "1VEX", "1U5M", "2KHR", "1Q09"]
        ]  # tiny proteins

    @property
    def files(self):
        if self._files is None:
            self._files = glob.glob(osp.join(self.processed_dir, "prot-*.pt"))
        return self._files

    @property
    def raw_file_names(self):
        return self._raw_filename

    def len(self):
        return len(self.files)

    def get(self, idx: int) -> Data:
        return torch.load(self.files[idx])

    def download(self):
        url = (
            "https://github.com/pstjohn/torch_gvp/releases/download/v0.0.1/"
            "sample_tiny.parquet"
        )

        download_url(url, self.raw_dir, log=True, filename=self._raw_filename)

    def process(self):

        existing_files = glob.glob(osp.join(self.processed_dir, "prot-*.pt"))
        existing_ids = pd.Series(existing_files, dtype=str).str.extract(
            ".*prot-(.*).pt$"
        )[0]

        map_fn = partial(
            process_item,
            processed_dir=self.processed_dir,
            pre_filter=self.pre_filter,
            pre_transform=self.pre_transform,
        )

        if dd is not None and self.num_processes != 1:
            df = dd.read_parquet(  # type: ignore
                Path(self.raw_dir, self._raw_filename), split_row_groups=True
            )
            df = df[~df["_1"].isin(existing_ids)]
            df.map_partitions(
                lambda x: x.apply(map_fn, axis=1), meta=pd.Series(dtype="float64")
            ).compute()

        else:
            tqdm.pandas()
            df = pd.read_parquet(Path(self.raw_dir, self._raw_filename))
            df = df[~df["_1"].isin(existing_ids)]
            df.progress_apply(map_fn, axis=1)
