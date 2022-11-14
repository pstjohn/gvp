import glob
import multiprocessing
import os.path as osp
from functools import partial
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import pandas as pd
import torch
from torch_geometric.data import Data, Dataset, download_url
from tqdm import tqdm

from torch_gvp.data.biotite import convert_to_pyg, load_bytes


def process_item(
    args: Tuple[str, bytes],
    processed_dir: Union[str, Path],
    pre_filter: Optional[Callable] = None,
    pre_transform: Optional[Callable] = None,
) -> None:
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
    name, protein_data = args
    data = convert_to_pyg(load_bytes(protein_data, compressed=True))

    if pre_filter is not None and not pre_filter(data):
        return

    if pre_transform is not None:
        data = pre_transform(data)

    torch.save(data, Path(processed_dir, f"prot-{name}.pt"))


def size_filter(data: Data, max_num_nodes: int = 10000) -> bool:
    if data.num_nodes:
        return data.num_nodes < max_num_nodes
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
        return ["prot-2XN4.pt", "prot-1ESE.pt"]  # head and tail proteins

    @property
    def files(self):
        if self._files is None:
            self._files = glob.glob(osp.join(self.processed_dir, "prot*.pt"))
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
            "https://github.com/pstjohn/gvp/releases/download/v0.0.1/"
            "small_sample.parquet"
        )

        download_url(url, self.raw_dir, filename=self._raw_filename)

    def process(self):

        df = pd.read_parquet(Path(self.raw_dir, self._raw_filename))

        map_fn = partial(
            process_item,
            processed_dir=self.processed_dir,
            pre_filter=self.pre_filter,
            pre_transform=self.pre_transform,
        )

        with multiprocessing.Pool(self.num_processes) as pool:
            for item in tqdm(pool.imap_unordered(map_fn, df.values), total=len(df)):
                continue
