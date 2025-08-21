import torch
from collections.abc import Callable
from torch_max_backend import max_backend


def check_functions_are_equivalent(
    fn: Callable,
    device: str | None,
    inputs: list[torch.Tensor],
    fn_compiled: Callable | None = None,
    rtol=5e-2,
    atol=5e-3,
):
    fn_compiled = fn_compiled or torch.compile(backend=max_backend)(fn)
    if device is not None:
        inputs = [input_tensor.to(device) for input_tensor in inputs]

    # We use the compiled first because compiled never changes
    # the input tensors, while the original function might.
    output_compiled = fn_compiled(*inputs)
    output_original = fn(*inputs)

    assert type(output_original) == type(output_compiled)

    if isinstance(output_original, torch.Tensor):
        output_original = [output_original]
        output_compiled = [output_compiled]

    for i, (original, compiled) in enumerate(zip(output_original, output_compiled)):
        assert original.shape == compiled.shape, f"Issue with output {i}"
        assert original.device == compiled.device, f"Issue with output {i}"
        assert original.dtype == compiled.dtype, f"Issue with output {i}"
        assert torch.allclose(original, compiled, rtol=rtol, atol=atol), (
            f"Issue with output {i}"
        )
