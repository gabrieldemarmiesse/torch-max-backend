import os

import torch
from torch_max_backend import max_backend
from torch._dynamo import mark_dynamic

os.environ["TORCH_MAX_BACKEND_PROFILE"] = "1"
os.environ["TORCH_MAX_BACKEND_VERBOSE"] = "1"
os.environ["TORCH_MAX_BACKEND_DEBUG_GRAPH"] = "1"


def _create_masks(seq_len):
    ones = torch.ones((seq_len, seq_len), dtype=torch.bool, device="cuda")
    mask_global = torch.triu(ones, diagonal=1)

    far_past = torch.triu(ones, diagonal=512).T
    mask_local = mask_global | far_past
    return mask_local


def model(input_ids):
    # Forward pass
    _, seq_len = input_ids.shape
    _ = input_ids + 1
    return _create_masks(seq_len)


model = torch.compile(model, backend=max_backend)

input_token_ids_tensor = torch.tensor([5 for _ in range(19)], device="cuda").unsqueeze(
    0
)

with torch.no_grad():
    mark_dynamic(input_token_ids_tensor, 1)
    _ = model(input_token_ids_tensor)[:, -1]
    raise RuntimeError("Debugging")
