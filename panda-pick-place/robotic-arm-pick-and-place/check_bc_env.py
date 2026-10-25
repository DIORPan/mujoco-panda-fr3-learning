"""Check CPU training tools with synthetic data; no expert data or checkpoints."""

import os
import sys

import numpy as np
import torch
from torch import nn


def main():
    torch.manual_seed(0)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    device = torch.device("cpu")

    # Exercise the actual NumPy <-> PyTorch bridge required by our NPZ dataset.
    array = np.random.default_rng(0).normal(size=(16, 25)).astype(np.float32)
    inputs = torch.from_numpy(array).to(device)
    network = nn.Sequential(
        nn.Linear(25, 128), nn.ReLU(),
        nn.Linear(128, 128), nn.ReLU(),
        nn.Linear(128, 8),
    ).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

    before = network[0].weight.detach().clone()
    outputs = network(inputs)
    loss = nn.functional.mse_loss(outputs, torch.zeros_like(outputs))
    optimizer.zero_grad()
    loss.backward()
    if not torch.isfinite(loss).item():
        raise RuntimeError("Non-finite loss in synthetic check.")
    for parameter in network.parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all().item():
            raise RuntimeError("Missing or non-finite gradients.")
    optimizer.step()
    if torch.equal(before, network[0].weight):
        raise RuntimeError("Optimizer did not update the network weights.")
    if outputs.detach().numpy().shape != (16, 8):
        raise RuntimeError("Unexpected output array shape.")

    print(f"python = {sys.version.split()[0]}")
    print(f"torch = {torch.__version__}")
    print(f"numpy = {np.__version__}")
    print(f"device = {inputs.device}")
    print(f"CUDA build = {torch.version.cuda}")
    print(f"input shape = {tuple(inputs.shape)}")
    print(f"output shape = {tuple(outputs.shape)}")
    print("NumPy bridge, forward, backward, optimizer: PASS")
    print("Synthetic check only. No expert data used and no policy saved.")


if __name__ == "__main__":
    main()
