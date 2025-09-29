import os

import torch
from torch._dynamo import mark_dynamic

from torch_max_backend import max_backend, register_max_devices

os.environ["TORCH_MAX_BACKEND_VERBOSE"] = "1"

register_max_devices()


def simple_func(img: torch.Tensor) -> torch.Tensor:
    img = img - 1
    img = img + img * 10
    img = img + 10
    return img


simple_func_compiled = torch.compile(simple_func, backend=max_backend)

array = torch.rand(2, 2).to("max_device")
mark_dynamic(array, 0)
mark_dynamic(array, 1)

x_eager = simple_func(array).cpu()
print("value after simple func with max_device eager mode:", x_eager)
x_compiled = simple_func_compiled(array).cpu()
print("value after simple func with max_device torch.compile:", x_compiled)

torch.testing.assert_close(x_eager, x_compiled)
print("Results match for simple_graph")
