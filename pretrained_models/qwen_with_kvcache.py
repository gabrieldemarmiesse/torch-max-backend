import torch
from torch_max_backend import max_backend


def simple_bug(x):
    # x is (3, 3)
    mask = torch.triu(torch.ones(3, 3, device=x.device, dtype=torch.bool), diagonal=1)
    return x.masked_fill(mask, -1.0)


# Compile and run
compiled_fn = torch.compile(simple_bug, backend=max_backend)
x = torch.randn(3, 3, device="cuda")
output = compiled_fn(x)
