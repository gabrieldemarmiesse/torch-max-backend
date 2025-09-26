from pathlib import Path
import torch
from torch_max_backend import max_backend, make_torch_op_from_mojo
import os
import requests
import io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

os.environ["TORCH_MAX_BACKEND_VERBOSE"] = "1"


def allocate_outputs_grayscale(pic: torch.Tensor) -> torch.Tensor:
    return pic.new_empty(pic.shape[:-1], dtype=torch.float32)


my_torch_grayscale = make_torch_op_from_mojo(
    Path(__file__).parent / "dummy_mojo_kernels",
    "grayscale",
    allocate_outputs_grayscale,
)


def simple_graph(img: torch.Tensor) -> torch.Tensor:
    img = img - 1
    img = my_torch_grayscale(img)
    img = img + 1
    return img


simple_graph_compiled = torch.compile(simple_graph, backend=max_backend)

img_url = "https://docs.modular.com/images/artwork/pytorch-custom-operators.jpg"

some_image = Image.open(io.BytesIO(requests.get(img_url).content)).convert("RGB")
img = torch.from_numpy(np.array(some_image)).to("cuda")

x_eager = simple_graph(img)
x_compiled = simple_graph_compiled(img)
torch.testing.assert_close(x_eager, x_compiled)
print("Results match for simple_graph")
explanation = torch._dynamo.explain(simple_graph_compiled, img)
print("Number of graph breaks:", explanation.graph_break_count)

plt.imshow(x_compiled.cpu().numpy().astype(np.uint8), cmap="gray")
plt.show()
