import logging
from datetime import datetime
from pathlib import Path
from time import time

import torch
import torch_geometric
import torch_geometric.transforms
import torchmetrics
from torch.utils.data import random_split
from torch.utils.tensorboard.writer import SummaryWriter
from torch_geometric.loader import DataLoader, DynamicBatchSampler
from torchmetrics.text.perplexity import Perplexity

from torch_gvp.data.rcsb_dataset import RCSBDataset, size_filter
from torch_gvp.data.transforms import (
    BaseTransform,
    EdgeSplit,
    NodeOrientation,
    ResidueMask,
)
from torch_gvp.models.res_gvp import ResidueGVP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("__name__")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.info("creating dataset object")


class PinMemory(BaseTransform):
    def __call__(self, data):
        return data.pin_memory()


pre_transform = torch_geometric.transforms.Compose(
    [
        # torch_geometric.transforms.ToDevice(device),  # type: ignore
        torch_geometric.transforms.RadiusGraph(
            r=10.0, loop=False, max_num_neighbors=32
        ),
    ]
)

post_transform = torch_geometric.transforms.Compose(
    [
        PinMemory(),
        torch_geometric.transforms.ToDevice(device, non_blocking=True),  # type: ignore
        torch_geometric.transforms.RandomJitter(0.02),
        torch_geometric.transforms.Cartesian(),
        torch_geometric.transforms.Distance(norm=False),
        NodeOrientation(),
        EdgeSplit(),
        ResidueMask(mask_prob=0.4, random_token_prob=0.15),
    ]
)

dataset = RCSBDataset(
    "/projects/robustmicrob/pstjohn/rcsb/sample",
    pre_transform=pre_transform,
    transform=post_transform,
    pre_filter=size_filter,
    num_processes=1,
)

logging.info("creating model")
model = ResidueGVP(
    node_dims=(128, 16),
    edge_dims=(32, 1),
    n_atom_conv=1,
    n_res_conv=3,
    conv_n_message=3,
    conv_n_feedforward=2,
    drop_rate=0.1,
    vector_gate=True,
).to(device)

valid_size = 25
train_size = len(dataset) - valid_size

logging.info("Splitting dataset")
train_dataset, valid_dataset = random_split(
    dataset, [train_size, valid_size], generator=torch.Generator().manual_seed(42)
)

logging.info("Creating dataloader")

max_num_nodes = 30000
train_sampler = DynamicBatchSampler(
    train_dataset, max_num=max_num_nodes, mode="node", shuffle=True  # type: ignore
)
train_loader = DataLoader(
    train_dataset, batch_sampler=train_sampler,  # type: ignore
)

valid_sampler = DynamicBatchSampler(
    valid_dataset, max_num=max_num_nodes, mode="node", shuffle=False  # type: ignore
)
valid_loader = DataLoader(
    valid_dataset, batch_sampler=valid_sampler,  # type: ignore
)

logging.info("Creating optimizer, loss, and accuracy objects")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
criteron = torch.nn.CrossEntropyLoss()
acc_metric = torchmetrics.Accuracy().to(device)
perplexity_metric = Perplexity(ignore_index=20).to(device)

writer = SummaryWriter(
    log_dir=Path(
        "/projects/robustmicrob/pstjohn/logs",
        "residueGVP",
        datetime.now().strftime("%Y/%m/%d, %H:%M"),
    )
)


def train(starting_step: int) -> int:
    model.train()
    current_step = 0
    for i, batch in enumerate(train_loader):

        start_time = time()
        current_step = starting_step + i

        logging.debug(f"Entering batch {i}")
        logging.debug(f"{batch.num_nodes = }")
        logging.debug(f"{batch.num_edges = }")

        optimizer.zero_grad()

        s_out = model(
            batch.atom_type,
            batch.residue_type,
            batch.node_v,
            batch.edge_s,
            batch.edge_v,
            batch.edge_index,
            batch.residue_index,
        )

        residue_logits = s_out[batch.residue_mask]
        logging.debug(f"{residue_logits.shape = }")
        logging.debug(f"{batch.true_residue_type.shape = }")

        # Need to account for python's off-by-one enum class assignments
        loss = criteron(residue_logits, batch.true_residue_type - 1)
        logging.info(f"Loss (batch {current_step}): {loss.item():.3f}")
        writer.add_scalar("Loss/train", loss, current_step)

        loss.backward()
        optimizer.step()

        acc = acc_metric(residue_logits, batch.true_residue_type - 1)
        logging.info(f"Accuracy (batch {current_step}): {acc:.3f}")
        writer.add_scalar("Accuracy/train", acc, current_step)

        perplexity = perplexity_metric(
            residue_logits.unsqueeze(0), batch.true_residue_type.unsqueeze(0) - 1
        )
        logging.info(f"Perplexity (batch {current_step}): {perplexity:.3f}")
        writer.add_scalar("Perplexity/train", perplexity, current_step)

        logging.debug(f"Finishing batch {current_step}")

        logging.info(f"Time / node: {(time() - start_time) / batch.num_nodes}")

    acc = acc_metric.compute()
    logging.info(f"Epoch accuracy: {acc:.3f}")

    perplexity = perplexity_metric.compute()
    logging.info(f"Epoch perplexity: {perplexity:.3f}")

    acc_metric.reset()
    perplexity_metric.reset()

    return current_step


def valid(step: int):

    logging.debug("Entering validation")
    model.eval()
    for batch in valid_loader:
        batch.to(device)

        with torch.no_grad():
            s_out = model(
                batch.atom_type,
                batch.residue_type,
                batch.node_v,
                batch.edge_s,
                batch.edge_v,
                batch.edge_index,
                batch.residue_index,
            )

            residue_logits = s_out[batch.residue_mask]
            acc = acc_metric(residue_logits, batch.true_residue_type - 1)
            perplexity = perplexity_metric(
                residue_logits.unsqueeze(0), batch.true_residue_type.unsqueeze(0) - 1
            )
            logging.debug(f"Validation batch accuracy: {acc:.3f}")

    acc = acc_metric.compute()
    writer.add_scalar("Accuracy/valid", acc, step)
    logging.info(f"Validation accuracy: {acc:.3f}")

    perplexity = perplexity_metric.compute()
    writer.add_scalar("Perplexity/valid", perplexity, step)
    logging.info(f"Validation perplexity: {perplexity:.3f}")

    acc_metric.reset()
    perplexity_metric.reset()

    logging.debug("Leaving validation")


if __name__ == "__main__":
    step = 0
    for epoch in range(100):
        step = train(step) + 1
        valid(step)

        torch.save(
            model.state_dict(),
            Path("/projects/robustmicrob/pstjohn/checkpoints", "model.pt"),
        )
