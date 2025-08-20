import torch
from torchvision.models import densenet121
from torch_max_backend import max_backend, get_accelerators

device = "cuda" if len(list(get_accelerators())) >= 2 else "cpu"


model = densenet121(pretrained=True).to(torch.float32).to(device)

model = torch.compile(model, backend=max_backend, fullgraph=True)

image_url = "https://raw.githubusercontent.com/jigsawpieces/dog-api-images/refs/heads/main/boxer/n02108089_10229.jpg"
