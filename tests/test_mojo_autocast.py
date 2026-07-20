"""AutocastPrivateUse1 policy tests for the eager Mojo device."""

import pytest
import torch

from torch_mojo_backend import get_accelerators, register_mojo_devices

pytestmark = pytest.mark.xdist_group(name="group1")


@pytest.fixture(autouse=True)
def setup_mojo_device():
    register_mojo_devices()


@pytest.fixture
def mojo_h100():
    accelerators = list(get_accelerators())
    if not accelerators:
        pytest.skip("MAX has no accelerator")
    accelerator = accelerators[0]
    if accelerator.api != "cuda" or accelerator.architecture_name != "sm_90a":
        pytest.skip("BF16 FA4 autocast validation requires an H100")
    return "mojo:0"


def test_mojo_autocast_fallback_and_required_policies_are_registered() -> None:
    assert torch._C._dispatch_has_backend_fallback(
        torch._C.DispatchKey.AutocastPrivateUse1
    )
    for name in (
        "addmm",
        "bmm",
        "linear",
        "matmul",
        "mm",
        "scaled_dot_product_attention",
        "_scaled_dot_product_flash_attention",
        "layer_norm",
        "native_layer_norm",
        "nll_loss",
        "nll_loss_forward",
    ):
        assert torch._C._dispatch_has_kernel_for_dispatch_key(
            f"aten::{name}", "AutocastPrivateUse1"
        ), name


def test_mojo_bf16_autocast_linear_loss_and_grad_dtypes(mojo_h100) -> None:
    generator = torch.Generator().manual_seed(20260718)
    host_input = torch.randn(8, 16, generator=generator)
    host_weight = torch.randn(11, 16, generator=generator)
    host_bias = torch.randn(11, generator=generator)
    target = torch.randint(0, 11, (8,), generator=generator)
    input = host_input.to(mojo_h100).requires_grad_()
    weight = host_weight.to(mojo_h100).requires_grad_()
    bias = host_bias.to(mojo_h100).requires_grad_()

    with torch.amp.autocast("mojo", dtype=torch.bfloat16):
        assert torch.is_autocast_enabled("mojo")
        logits = torch.nn.functional.linear(input, weight, bias)
        loss = torch.nn.functional.cross_entropy(logits, target.to(mojo_h100))

    assert logits.dtype == torch.bfloat16
    assert loss.dtype == torch.float32
    loss.backward()
    for tensor in (input, weight, bias):
        assert tensor.grad is not None
        assert tensor.grad.dtype == torch.float32
        assert torch.isfinite(tensor.grad.cpu()).all()


def test_mojo_bf16_autocast_layernorm_and_fa4_dtypes(mojo_h100) -> None:
    generator = torch.Generator().manual_seed(20260718)
    layer_input = torch.randn(4, 64, generator=generator).to(mojo_h100)
    layer_weight = torch.randn(64, generator=generator).to(mojo_h100)
    layer_bias = torch.randn(64, generator=generator).to(mojo_h100)
    q, k, v = [
        torch.randn(1, 4, 128, 64, generator=generator).to(mojo_h100).requires_grad_()
        for _ in range(3)
    ]

    with torch.amp.autocast("mojo", dtype=torch.bfloat16):
        normalized = torch.nn.functional.layer_norm(
            layer_input, (64,), layer_weight, layer_bias
        )
        attended = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=True
        )

    assert normalized.dtype == torch.float32
    assert attended.dtype == torch.bfloat16
    attended.backward(torch.ones(attended.shape, dtype=torch.bfloat16).to(mojo_h100))
    for tensor in (q, k, v):
        assert tensor.grad is not None
        assert tensor.grad.dtype == torch.float32
        assert torch.isfinite(tensor.grad.cpu()).all()
