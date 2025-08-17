# Torch's MAX Backend

Simply use [`torch.compile`](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html), but with Modular's MAX backend.

## Installation

```bash
pip install torch-max-backend
```

## Quick Start

### Basic Usage

```python
from torch_max_backend import max_backend
import torch

# Compile your model with MAX backend
model = YourModel()
compiled_model = torch.compile(model, backend=max_backend)

# Use normally - now accelerated by MAX
output = compiled_model(input_tensor)
```

### Simple Function Example

```python
import torch
from torch_max_backend import max_backend

@torch.compile(backend=max_backend)
def simple_math(x, y):
    return x + y * 2

# Usage
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])
print(simple_math(a, b))  # Accelerated execution
```

### Training

Training works as expected 

```python
from torch_max_backend import max_backend
import torch
import torch.nn
import torch.optim
import torch.nn.functional as F

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)

    def forward(self, x):
        return self.linear(x)

device = "cuda"
model = MyModel().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

@torch.compile(backend=max_backend)
def train_step(x, y):
    model.train()
    optimizer.zero_grad()
    output = model(x)
    loss = F.mse_loss(output, y)
    loss.backward()
    optimizer.step()
    return loss

a = torch.randn(5, 3).to(device)
b = torch.randn(5, 2).to(device)

print(train_step(a, b).cpu().detach().numpy())
```

### Device Selection

Note that currently the MAX backend does not support some older nvidia/amd gpus.
So you'll need to ask MAX first if your GPU is supported before 
using the gpu.

```python
from torch_max_backend import get_accelerators

# Check available accelerators
# The CPU is necessarily included in the list of accelerators
accelerators = get_accelerators()
device = "cuda" if len(list(accelerators)) >= 2 else "cpu"
model = model.to(device)
```


## Supported Operations

The backend currently supports operations defined in [`aten_functions.py`](https://github.com/gabrieldemarmiesse/max-torch-backend/blob/main/torch_max_backend/aten_functions.py). You can view the mapping dictionary by importing `MAPPING_TORCH_ATEN_TO_MAX`.

## Extending the Backend

You can add support for new PyTorch operations without cloning the repository by creating custom mappings:

```python
from torch_max_backend import MAPPING_TORCH_ATEN_TO_MAX
from torch.ops import aten
import max_ops

# Example: Add support for a new operation
def my_custom_tanh(x):
    return max_ops.tanh(x)

# Register the operation
MAPPING_TORCH_ATEN_TO_MAX[aten.tanh] = my_custom_tanh

# Now you can use it with torch.compile
import torch
from torch_max_backend import max_backend

@torch.compile(backend=max_backend)
def my_function(x):
    return torch.tanh(x)  # Will now use your custom implementation
```

This approach allows you to:
- Add missing operations your models need
- Override existing implementations with optimized versions
- Prototype new MAX operations before contributing them back

## Performance Tips

### Dynamic Shapes
For variable input sizes, mark dynamic dimensions to avoid recompiling:

```python
from torch._dynamo import mark_dynamic

mark_dynamic(input_tensor, 0)  # batch dimension
mark_dynamic(input_tensor, 1)  # sequence length
```

If you don't do so, Pytorch will compile a second time when it sees a different shape, which can be costly.
You can find more information about dynamic shapes in the [PyTorch documentation](https://docs.pytorch.org/docs/stable/torch.compiler_dynamic_shapes.html).

### Compilation Strategy
- Use `fullgraph=True` when possible for better optimization. You'll get an error message if 
  pytorch has to trigger a graph break, making it easy to fix.

### Debugging
You can get various information with the following environement variables:
* `TORCH_MAX_BACKEND_PROFILE=1` to get various information about timing (time to compile, time to run, ...)
* `TORCH_MAX_BACKEND_VERBOSE=1` to display the graph(s) made by pytorch and various other information.
* `TORCH_MAX_BACKEND_BEARTYPE=0` to disable type checking. By default, everything in the package is type-checked at runtime. But it may lead to errors when actually the code is valid (and the type hint is wrong). You can try disabling the type-checking then to see if the bug goes away. Feel free to open a bug report in any case! Type errors should never happen and are a sign of an internal bug.

## Contributing
### Testing

```bash
# Run all tests (the first time is slow, chaching kicks in after)
uv run pytest -v -n 2 --forked

# Lint and format
uvx pre-commit run --all-files
# Or install the pre-commit hook
uvx pre-commit install
```

### Aten ops
The list of aten ops can be found in [`ressources/aten_ops.txt`](ressources/aten_ops.txt). Sadly 
the documentation for those ops is non-existant. Reverse-ingeneering is the only thing that works.
