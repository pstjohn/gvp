from dask.distributed import Client

from torch_gvp.data.rcsb_dataset import RCSBDataset, size_filter

if __name__ == "__main__":
    client = Client()
    print(client.dashboard_link, flush=True)
    dataset = RCSBDataset(
        "/projects/robustmicrob/pstjohn/rcsb/", pre_filter=size_filter
    )
