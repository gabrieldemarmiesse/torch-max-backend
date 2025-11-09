import inspect
import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from torch_max_backend import max_backend


def scaled_dot_product_flash_attention_cpu(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = False,
    *,
    scale: float | None = None,
):
    """
    Pure PyTorch implementation of scaled dot product attention for CPU.
    This mimics the behavior of aten::_scaled_dot_product_flash_attention.

    Args:
        query: Query tensor of shape [batch, num_heads, seq_len, head_dim]
        key: Key tensor of shape [batch, num_heads, seq_len, head_dim]
        value: Value tensor of shape [batch, num_heads, seq_len, head_dim]
        dropout_p: Dropout probability (not implemented in this CPU version)
        is_causal: Whether to apply causal masking
        return_debug_mask: Whether to return debug mask (not implemented)
        scale: Scaling factor for attention scores

    Returns:
        Tuple matching PyTorch's flash attention signature:
        (output, logsumexp, cum_seq_q, cum_seq_k, max_q, max_k, rng_state, unused, debug_attn_mask)
        Only the first element (output) is properly implemented.
    """
    # Calculate scale if not provided
    if scale is None:
        head_dim = query.shape[-1]
        scale = 1.0 / math.sqrt(float(head_dim))

    # Convert scale to tensor to handle different dtypes properly
    scale_tensor = torch.tensor(scale, dtype=query.dtype, device=query.device)

    # Compute attention scores: (batch, num_heads, seq_len, seq_len)
    attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale_tensor

    # Apply causal mask if requested
    if is_causal:
        batch_size, num_heads, seq_len, _ = attn_scores.shape
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=query.device),
            diagonal=1,
        )
        attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))

    # Apply softmax to get attention weights
    attn_weights = F.softmax(attn_scores, dim=-1)

    # Apply dropout if specified (skip for CPU fallback to keep it simple)
    if dropout_p > 0.0:
        # Note: In a real implementation, this would need training mode handling
        pass

    # Compute attention output
    attn_output = torch.matmul(attn_weights, value)

    # Compute logsumexp for stability (used in training)
    logsumexp = torch.logsumexp(attn_scores, dim=-1)

    # Create placeholder tensors for the remaining return values
    # These are used for training and optimization in the real implementation
    batch_size, num_heads, seq_len, head_dim = query.shape

    # cum_seq_q and cum_seq_k are for variable sequence length support
    cum_seq_q = (
        torch.arange(1, seq_len + 1, dtype=torch.int32, device=query.device)
        .unsqueeze(0)
        .expand(batch_size, -1)
    )
    cum_seq_k = (
        torch.arange(1, seq_len + 1, dtype=torch.int32, device=query.device)
        .unsqueeze(0)
        .expand(batch_size, -1)
    )

    # max_q and max_k are symbolic integers for sequence lengths
    max_q = seq_len
    max_k = seq_len

    # rng_state is for dropout reproducibility
    rng_state = torch.empty(2, dtype=torch.int64, device=query.device)

    # unused tensor
    unused = torch.empty(0, dtype=query.dtype, device=query.device)

    # debug_attn_mask if requested
    if return_debug_mask:
        debug_attn_mask = attn_weights
    else:
        debug_attn_mask = torch.empty(0, dtype=query.dtype, device=query.device)

    # Return tuple matching PyTorch's flash attention signature
    return (
        attn_output,
        logsumexp,
        cum_seq_q,
        cum_seq_k,
        max_q,
        max_k,
        rng_state,
        unused,
        debug_attn_mask,
    )


# Register the CPU implementation as a fallback for flash attention
def register_cpu_flash_attention_fallback():
    """Register the CPU flash attention implementation using torch.library."""
    torch.library.impl(
        "aten::_scaled_dot_product_flash_attention",
        "CPU",
        scaled_dot_product_flash_attention_cpu,
    )


def check_functions_are_equivalent(
    fn: Callable,
    device: str | None,
    inputs: list[torch.Tensor],
    fn_compiled: Callable | None = None,
    rtol=None,
    atol=None,
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
        torch.testing.assert_close(original, compiled, rtol=rtol, atol=atol)


@dataclass
class Conf:
    device: str
    compile: bool

    def __str__(self) -> str:
        if self.compile:
            word = "compiled"
        else:
            word = "eager"
        return f"{self.device}, {word}"


def to_device(tensors: list[torch.Tensor], device: str) -> list[torch.Tensor]:
    return [torch.clone(tensor).to(device) for tensor in tensors]


def check_outputs(
    fn: Callable, conf: Conf, inputs: list[torch.Tensor], *, rtol=None, atol=None
):
    # We compare to eager cpu execution
    # We first check if the function has a device argument
    has_device_arg = "device" in inspect.signature(fn).parameters
    inputs_cpu = to_device(inputs, "cpu")
    if has_device_arg:
        outputs_eager_cpu = fn(*inputs_cpu, device="cpu")
    else:
        outputs_eager_cpu = fn(*inputs_cpu)

    if conf.compile:
        fn_to_run = torch.compile(fn, backend=max_backend)
    else:
        fn_to_run = fn

    inputs_on_device = to_device(inputs, conf.device)
    if has_device_arg:
        outputs_conf = fn_to_run(*inputs_on_device, device=conf.device)
    else:
        outputs_conf = fn_to_run(*inputs_on_device)

    # Now we compare outputs
    if isinstance(outputs_eager_cpu, torch.Tensor):
        outputs_eager_cpu = [outputs_eager_cpu]
        outputs_conf = [outputs_conf]

    for i, (output_eager_cpu, output_conf) in enumerate(
        zip(outputs_eager_cpu, outputs_conf)
    ):
        expected_device = torch.device(conf.device)
        if not (output_conf.device == expected_device):
            raise AssertionError(
                f"Issue with output {i}, expected device {repr(expected_device)} but got {repr(output_conf.device)}"
            )
        assert output_eager_cpu.shape == output_conf.shape, f"Issue with output {i}"
        assert output_eager_cpu.dtype == output_conf.dtype, f"Issue with output {i}"
        # move to cpu for comparison
        output_conf_cpu = output_conf.to("cpu")
        torch.testing.assert_close(
            output_eager_cpu, output_conf_cpu, rtol=rtol, atol=atol
        )
