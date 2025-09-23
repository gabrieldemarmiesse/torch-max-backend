import torch
from torch_max_backend import max_backend


def minimal_bug(x):
    mask = torch.triu(torch.ones(3, 3, device=x.device, dtype=torch.bool), diagonal=1)
    return x.masked_fill(mask, -1.0)


compiled_fn = torch.compile(minimal_bug, backend=max_backend)
x = torch.randn(3, 3, device="cuda")
output = compiled_fn(x)
