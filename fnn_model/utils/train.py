from typing import Union
from torch_geometric.loader import DataLoader
import torch
import torch.optim as opt
from torch.nn.modules.loss import _Loss

def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: Union[opt.AdamW, opt.SGD, opt.Adam],
    loss_fn: _Loss,
    device: torch.device,
    ) -> float:

    model.train()
    loss_all = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(data), data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()

    return loss_all / len(train_loader.dataset)
