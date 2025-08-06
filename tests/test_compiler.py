import pytest
import torch
from collections.abc import Callable
from max_torch_backend import my_compiler


def check_functions_are_equivalent(
    fn: Callable, device: str, inputs: list[torch.Tensor]
):
    fn_compiled = torch.compile(backend=my_compiler)(fn)

    inputs_on_device = [input_tensor.to(device) for input_tensor in inputs]

    output_original = fn(*inputs_on_device)
    output_compiled = fn_compiled(*inputs_on_device)

    assert type(output_original) == type(output_compiled)

    if isinstance(output_original, torch.Tensor):
        output_original = [output_original]
        output_compiled = [output_compiled]

    for original, compiled in zip(output_original, output_compiled):
        assert torch.allclose(original, compiled, rtol=1e-5)
        assert original.device == compiled.device


def test_basic_addition(device: str):
    def fn(x, y):
        return x + y

    a = torch.randn(3)
    b = torch.randn(3)

    check_functions_are_equivalent(fn, device, [a, b])
