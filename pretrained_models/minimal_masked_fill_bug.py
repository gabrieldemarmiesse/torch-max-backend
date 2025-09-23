import torch
import torch.nn as nn
from torch_max_backend import max_backend


class MinimalModel(nn.Module):
    def forward(self, x):
        # Create a mask
        mask = torch.triu(
            torch.ones(3, 3, device=x.device, dtype=torch.bool), diagonal=1
        )
        # Use masked_fill
        return x.masked_fill(mask, -torch.inf)


model = MinimalModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Compile with max_backend
model = torch.compile(model, backend=max_backend, fullgraph=True)

# Test input
x = torch.randn(3, 3, device=device)

with torch.no_grad():
    output = model(x)
    print("Forward pass completed")
