"""Host-only contracts for the future fused H100 BF16 cross entropy."""

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from torch_mojo_backend.eager_kernels import aten_fast
from torch_mojo_backend.mojo_device import mojo_device_aten_ops
from torch_mojo_backend.mojo_device import mojo_device_autograd as autograd


def _device(*, architecture: str = "sm_90a", device_id: int = 0):
    return SimpleNamespace(
        label="gpu", api="cuda", architecture_name=architecture, id=device_id
    )


def _tensor(
    name: str,
    shape,
    *,
    dtype=None,
    device=None,
    ptr: int = 100,
    contiguous: bool = True,
    offset: int = 0,
    requires_grad: bool = False,
):
    shape = tuple(shape)
    return SimpleNamespace(
        name=name,
        _shape=shape,
        _strides=aten_fast._row_major_strides(shape),
        _offset=offset,
        _dtype=dtype or aten_fast.DType.bfloat16,
        _device=device or _device(),
        _ptr=ptr,
        _itemsize=(dtype or aten_fast.DType.bfloat16).size_in_bytes,
        _numel=math.prod(shape),
        _is_contiguous=contiguous,
        _holder=object(),
        requires_grad=requires_grad,
    )


def _eligible_inputs(*, requires_grad: bool = False):
    device = _device()
    logits = _tensor(
        "logits",
        (7, 65),
        device=device,
        ptr=1042,
        offset=21,
        requires_grad=requires_grad,
    )
    target = _tensor(
        "target", (7,), dtype=aten_fast.DType.int64, device=device, ptr=2080, offset=10
    )
    return logits, target


def _mock_mojo_autocast(monkeypatch: pytest.MonkeyPatch, enabled: bool) -> None:
    monkeypatch.setattr(
        aten_fast.torch,
        "is_autocast_enabled",
        lambda device_type=None: enabled if device_type == "mojo" else False,
    )


def test_bf16_cross_entropy_optional_module_is_registered() -> None:
    from torch_mojo_backend import eager_kernels

    assert "bf16_cross_entropy_ops" in eager_kernels._MOJO_MODULES


def test_bf16_cross_entropy_missing_sources_do_no_device_work(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from torch_mojo_backend import eager_kernels

    logits, target = _eligible_inputs()
    _mock_mojo_autocast(monkeypatch, True)
    monkeypatch.setattr(aten_fast, "_t", lambda tensor: tensor)
    monkeypatch.delitem(eager_kernels.__dict__, "bf16_cross_entropy_ops", raising=False)
    monkeypatch.setattr(
        aten_fast,
        "_BF16_CROSS_ENTROPY_SOURCE_PATHS",
        (tmp_path / "missing_bridge.mojo", tmp_path / "missing_kernel.mojo"),
    )
    monkeypatch.setattr(aten_fast, "_BF16_CROSS_ENTROPY_IMPORT_FAILED", False)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("missing fused sources reached allocation or launch")

    monkeypatch.setattr(aten_fast, "_alloc", forbidden)
    assert not aten_fast.bf16_cross_entropy_supported(logits, target)
    assert (
        aten_fast.fast_bf16_cross_entropy_forward(logits, target)
        is aten_fast.NOT_HANDLED
    )


@pytest.mark.parametrize(
    "invalid",
    [
        "logits_dtype",
        "target_dtype",
        "device",
        "architecture",
        "logits_layout",
        "target_layout",
        "logits_rank",
        "target_shape",
        "empty_rows",
        "weight",
        "reduction",
        "ignore_index",
        "label_smoothing",
    ],
)
def test_bf16_cross_entropy_rejects_metadata_before_bridge_or_allocation(
    monkeypatch: pytest.MonkeyPatch, invalid: str
) -> None:
    logits, target = _eligible_inputs()
    _mock_mojo_autocast(monkeypatch, True)
    kwargs = {}
    if invalid == "logits_dtype":
        logits._dtype = aten_fast.DType.float32
    elif invalid == "target_dtype":
        target._dtype = aten_fast.DType.int32
    elif invalid == "device":
        target._device = _device(device_id=1)
    elif invalid == "architecture":
        logits._device.architecture_name = "sm_80"
    elif invalid == "logits_layout":
        logits._is_contiguous = False
    elif invalid == "target_layout":
        target._is_contiguous = False
    elif invalid == "logits_rank":
        logits._shape = (1, 7, 65)
    elif invalid == "target_shape":
        target._shape = (6,)
    elif invalid == "empty_rows":
        logits._shape = (0, 65)
        target._shape = (0,)
    elif invalid == "weight":
        kwargs["weight"] = object()
    elif invalid == "reduction":
        kwargs["reduction"] = 2
    elif invalid == "ignore_index":
        kwargs["ignore_index"] = True
    elif invalid == "label_smoothing":
        kwargs["label_smoothing"] = 0.1

    monkeypatch.setattr(aten_fast, "_t", lambda tensor: tensor)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("invalid metadata reached bridge/device work")

    monkeypatch.setattr(aten_fast, "_resolve_bf16_cross_entropy_bridge", forbidden)
    monkeypatch.setattr(aten_fast, "_alloc", forbidden)
    assert not aten_fast.bf16_cross_entropy_supported(logits, target, **kwargs)
    assert (
        aten_fast.fast_bf16_cross_entropy_forward(logits, target, **kwargs)
        is aten_fast.NOT_HANDLED
    )


def test_bf16_cross_entropy_forward_bridge_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logits, target = _eligible_inputs()
    _mock_mojo_autocast(monkeypatch, True)
    launches = []
    allocations = []
    next_ptr = iter((3001, 3002, 3003, 3004, 3005))

    def forward(*args):
        launches.append(args)

    bridges = {"Bf16CrossEntropyForward": forward, "Bf16CrossEntropyBackward": object()}

    def alloc(shape, dtype, device):
        tensor = _tensor(
            f"allocation_{len(allocations)}",
            shape,
            dtype=dtype,
            device=device,
            ptr=next(next_ptr),
        )
        allocations.append((tuple(shape), dtype, device, tensor))
        return tensor

    monkeypatch.setattr(aten_fast, "_t", lambda tensor: tensor)
    monkeypatch.setattr(aten_fast, "_resolve_bf16_cross_entropy_bridge", bridges.get)
    monkeypatch.setattr(aten_fast, "_alloc", alloc)
    monkeypatch.setattr(aten_fast, "_ctx_ptr", lambda device: 9000 + device.id)

    loss, row_max, row_logsum, total_weight = aten_fast.fast_bf16_cross_entropy_forward(
        logits, target, ignore_index=-1, require_backward=True
    )

    assert [(shape, dtype) for shape, dtype, _device, _tensor in allocations] == [
        ((), aten_fast.DType.float32),
        ((), aten_fast.DType.float32),
        ((7,), aten_fast.DType.float32),
        ((7,), aten_fast.DType.float32),
        ((7,), aten_fast.DType.float32),
    ]
    assert loss is allocations[0][3]
    assert total_weight is allocations[1][3]
    assert row_max is allocations[2][3]
    assert row_logsum is allocations[3][3]
    assert launches == [(3001, 3002, 3003, 3004, 3005, 1042, 2080, 7, 65, -1, 9000)]


def test_bf16_cross_entropy_requires_backward_bridge_before_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logits, target = _eligible_inputs(requires_grad=True)
    _mock_mojo_autocast(monkeypatch, True)
    monkeypatch.setattr(aten_fast, "_t", lambda tensor: tensor)
    monkeypatch.setattr(
        aten_fast,
        "_resolve_bf16_cross_entropy_bridge",
        lambda name: object() if name == "Bf16CrossEntropyForward" else None,
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("a forward-only bridge allocated autograd state")

    monkeypatch.setattr(aten_fast, "_alloc", forbidden)
    assert not aten_fast.bf16_cross_entropy_supported(
        logits, target, require_backward=True
    )
    assert (
        aten_fast.fast_bf16_cross_entropy_forward(logits, target, require_backward=True)
        is aten_fast.NOT_HANDLED
    )


def test_bf16_cross_entropy_without_autocast_rejects_before_bridge_or_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logits, target = _eligible_inputs()
    _mock_mojo_autocast(monkeypatch, False)
    monkeypatch.setattr(aten_fast, "_t", lambda tensor: tensor)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("non-autocast BF16 reached the FP32-loss fused route")

    monkeypatch.setattr(aten_fast, "_resolve_bf16_cross_entropy_bridge", forbidden)
    monkeypatch.setattr(aten_fast, "_alloc", forbidden)
    assert not aten_fast.bf16_cross_entropy_supported(logits, target)
    assert (
        aten_fast.fast_bf16_cross_entropy_forward(logits, target)
        is aten_fast.NOT_HANDLED
    )


def test_bf16_cross_entropy_backward_bridge_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logits, target = _eligible_inputs(requires_grad=True)
    grad_output = _tensor(
        "grad_output",
        (),
        dtype=aten_fast.DType.float32,
        device=logits._device,
        ptr=4001,
    )
    row_max = _tensor(
        "row_max", (7,), dtype=aten_fast.DType.float32, device=logits._device, ptr=4002
    )
    row_logsum = _tensor(
        "row_logsum",
        (7,),
        dtype=aten_fast.DType.float32,
        device=logits._device,
        ptr=4003,
    )
    total_weight = _tensor(
        "total_weight",
        (),
        dtype=aten_fast.DType.float32,
        device=logits._device,
        ptr=4004,
    )
    launches = []
    allocations = []

    monkeypatch.setattr(aten_fast, "_t", lambda tensor: tensor)
    monkeypatch.setattr(
        aten_fast,
        "_resolve_bf16_cross_entropy_bridge",
        lambda name: (
            (lambda *args: launches.append(args))
            if name == "Bf16CrossEntropyBackward"
            else None
        ),
    )

    def alloc(shape, dtype, device):
        allocations.append((tuple(shape), dtype, device))
        return _tensor("grad_input", shape, dtype=dtype, device=device, ptr=5001)

    monkeypatch.setattr(aten_fast, "_alloc", alloc)
    monkeypatch.setattr(aten_fast, "_ctx_ptr", lambda device: 7000 + device.id)

    grad_input = aten_fast.fast_bf16_cross_entropy_backward(
        grad_output, logits, target, row_max, row_logsum, total_weight, -1
    )

    assert grad_input._dtype == aten_fast.DType.bfloat16
    assert allocations == [((7, 65), aten_fast.DType.bfloat16, logits._device)]
    assert launches == [(5001, 4001, 1042, 2080, 4002, 4003, 4004, 7, 65, -1, 7000)]


@pytest.mark.parametrize(
    "invalid", ["grad", "row_max", "row_logsum", "total_weight", "device"]
)
def test_bf16_cross_entropy_backward_rejects_before_bridge_or_allocation(
    monkeypatch: pytest.MonkeyPatch, invalid: str
) -> None:
    logits, target = _eligible_inputs(requires_grad=True)
    grad_output = _tensor(
        "grad", (), dtype=aten_fast.DType.float32, device=logits._device
    )
    row_max = _tensor(
        "row_max", (7,), dtype=aten_fast.DType.float32, device=logits._device
    )
    row_logsum = _tensor(
        "row_logsum", (7,), dtype=aten_fast.DType.float32, device=logits._device
    )
    total_weight = _tensor(
        "weight", (), dtype=aten_fast.DType.float32, device=logits._device
    )
    if invalid == "grad":
        grad_output._dtype = aten_fast.DType.bfloat16
    elif invalid == "row_max":
        row_max._shape = (6,)
    elif invalid == "row_logsum":
        row_logsum._shape = (6,)
    elif invalid == "total_weight":
        total_weight._numel = 2
    elif invalid == "device":
        row_max._device = _device(device_id=1)

    monkeypatch.setattr(aten_fast, "_t", lambda tensor: tensor)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("invalid backward metadata reached device work")

    monkeypatch.setattr(aten_fast, "_resolve_bf16_cross_entropy_bridge", forbidden)
    monkeypatch.setattr(aten_fast, "_alloc", forbidden)
    assert (
        aten_fast.fast_bf16_cross_entropy_backward(
            grad_output, logits, target, row_max, row_logsum, total_weight, -1
        )
        is aten_fast.NOT_HANDLED
    )


def test_privateuse1_cross_entropy_uses_fused_result_or_exact_decomposition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    loss = object()
    logits, target = object(), object()
    fast = SimpleNamespace(
        NOT_HANDLED=sentinel,
        fast_bf16_cross_entropy_forward=lambda *_args: (
            loss,
            object(),
            object(),
            object(),
        ),
    )
    monkeypatch.setattr(mojo_device_aten_ops, "_fast", lambda: fast)
    assert mojo_device_aten_ops.mojo_device_cross_entropy_loss(logits, target) is loss

    calls = []
    fast.fast_bf16_cross_entropy_forward = lambda *_args: sentinel
    monkeypatch.setattr(
        mojo_device_aten_ops,
        "decompose_cross_entropy_loss",
        lambda *args: calls.append(args) or "composite",
    )
    assert (
        mojo_device_aten_ops.mojo_device_cross_entropy_loss(
            logits, target, None, 2, -1, 0.2
        )
        == "composite"
    )
    assert calls == [(logits, target, None, 2, -1, 0.2)]


def test_autograd_router_falls_back_and_requires_both_fused_bridges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logits, target = _eligible_inputs(requires_grad=True)
    calls = []
    fake_fast = SimpleNamespace(
        bf16_cross_entropy_supported=lambda *_args, **kwargs: (
            calls.append(kwargs) or False
        )
    )
    monkeypatch.setattr(autograd, "_fast", lambda: fake_fast)
    monkeypatch.setattr(
        autograd,
        "decompose_cross_entropy_loss",
        lambda *args: calls.append(args) or "composite",
    )
    assert autograd._cross_entropy_loss_autograd(logits, target) == "composite"
    assert calls[0] == {"require_backward": True}
    assert calls[1] == (logits, target, None, 1, -100, 0.0)

    fake_fast.bf16_cross_entropy_supported = lambda *_args, **_kwargs: True
    monkeypatch.setattr(
        autograd._Bf16CrossEntropyAutograd,
        "apply",
        lambda *args: calls.append(args) or "fused-autograd",
    )
    assert autograd._cross_entropy_loss_autograd(logits, target) == "fused-autograd"


def test_autograd_router_no_grad_does_not_require_backward_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logits, target = _eligible_inputs(requires_grad=True)
    loss = object()
    calls = []
    fake_fast = SimpleNamespace(
        NOT_HANDLED=object(),
        bf16_cross_entropy_supported=lambda *_args, **kwargs: (
            calls.append(("supported", kwargs)) or True
        ),
        fast_bf16_cross_entropy_forward=lambda *_args, **kwargs: (
            calls.append(("forward", kwargs)) or (loss, object(), object(), object())
        ),
    )
    monkeypatch.setattr(autograd, "_fast", lambda: fake_fast)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("no_grad inference constructed an autograd function")

    monkeypatch.setattr(autograd._Bf16CrossEntropyAutograd, "apply", forbidden)
    with torch.no_grad():
        assert autograd._cross_entropy_loss_autograd(logits, target) is loss
    assert calls == [("supported", {"require_backward": False}), ("forward", {})]


def test_fused_autograd_saves_only_persistent_kernel_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logits, target = _eligible_inputs(requires_grad=True)
    loss = _tensor("loss", (), dtype=aten_fast.DType.float32, device=logits._device)
    row_max = _tensor(
        "row_max", (7,), dtype=aten_fast.DType.float32, device=logits._device
    )
    row_logsum = _tensor(
        "row_logsum", (7,), dtype=aten_fast.DType.float32, device=logits._device
    )
    total_weight = _tensor(
        "total_weight", (), dtype=aten_fast.DType.float32, device=logits._device
    )
    grad_output = _tensor(
        "grad_output", (), dtype=aten_fast.DType.float32, device=logits._device
    )
    grad_input = _tensor(
        "grad_input", (7, 65), dtype=aten_fast.DType.bfloat16, device=logits._device
    )
    sentinel = object()
    calls = []
    fake_fast = SimpleNamespace(
        NOT_HANDLED=sentinel,
        fast_bf16_cross_entropy_forward=lambda *args, **kwargs: (
            calls.append((args, kwargs)) or (loss, row_max, row_logsum, total_weight)
        ),
        fast_bf16_cross_entropy_backward=lambda *args: calls.append(args) or grad_input,
    )

    class Context:
        def save_for_backward(self, *tensors):
            self.saved_tensors = tensors

        def set_materialize_grads(self, value):
            self.materialize_grads = value

    ctx = Context()
    monkeypatch.setattr(autograd, "_fast", lambda: fake_fast)
    monkeypatch.setattr(autograd, "_SavedMojoPayload", lambda tensor: tensor.name)

    actual_loss = autograd._Bf16CrossEntropyAutograd.forward(
        ctx, logits, target, None, 1, -1, 0.0
    )
    assert actual_loss is loss
    assert ctx.saved_tensors == (logits, target, row_max, row_logsum, total_weight)
    assert ctx.saved_payloads == (
        "logits",
        "target",
        "row_max",
        "row_logsum",
        "total_weight",
    )
    assert ctx.materialize_grads is False
    assert calls[0][1] == {"require_backward": True}

    monkeypatch.setattr(
        autograd,
        "_restore_saved_mojo_tensors",
        lambda actual_ctx: actual_ctx.saved_tensors,
    )
    gradients = autograd._Bf16CrossEntropyAutograd.backward(ctx, grad_output)
    assert gradients == (grad_input, None, None, None, None, None)
    assert calls[1] == (
        grad_output,
        logits,
        target,
        row_max,
        row_logsum,
        total_weight,
        -1,
    )


def test_cross_entropy_privateuse1_and_autograd_registrations() -> None:
    from torch_mojo_backend import register_mojo_devices

    register_mojo_devices()
    assert torch._C._dispatch_has_kernel_for_dispatch_key(
        "aten::cross_entropy_loss", "PrivateUse1"
    )
    assert torch._C._dispatch_has_kernel_for_dispatch_key(
        "aten::cross_entropy_loss", "AutogradPrivateUse1"
    )


def test_cross_entropy_decompose_preserves_cpu_autograd() -> None:
    from torch_mojo_backend.mojo_device.cross_entropy import (
        decompose_cross_entropy_loss,
    )

    logits = torch.randn(5, 11, dtype=torch.bfloat16, requires_grad=True)
    target = torch.tensor([1, 3, -1, 7, 2])
    loss = decompose_cross_entropy_loss(logits, target, None, 1, -1, 0.0)
    loss.backward()
    assert loss.dtype == torch.bfloat16
    assert loss.grad_fn is not None
    assert logits.grad is not None
    assert logits.grad.dtype == torch.bfloat16
    assert torch.isfinite(logits.grad).all()


def test_privateuse1_no_grad_route_observes_live_mojo_autocast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Autocast fallthrough must leave its TLS flag visible to PrivateUse1."""
    from torch_mojo_backend import register_mojo_devices

    register_mojo_devices()
    logits, target = _eligible_inputs()
    allocations = []
    bridges = {
        "Bf16CrossEntropyForward": lambda *_args: None,
        "Bf16CrossEntropyBackward": lambda *_args: None,
    }

    def alloc(shape, dtype, device):
        result = _tensor(
            f"allocation_{len(allocations)}",
            shape,
            dtype=dtype,
            device=device,
            ptr=6000 + len(allocations),
        )
        allocations.append(result)
        return result

    monkeypatch.setattr(aten_fast, "_t", lambda tensor: tensor)
    monkeypatch.setattr(aten_fast, "_resolve_bf16_cross_entropy_bridge", bridges.get)
    monkeypatch.setattr(aten_fast, "_alloc", alloc)
    monkeypatch.setattr(aten_fast, "_ctx_ptr", lambda _device: 7777)
    monkeypatch.setattr(mojo_device_aten_ops, "_fast", lambda: aten_fast)
    monkeypatch.setattr(
        mojo_device_aten_ops,
        "decompose_cross_entropy_loss",
        lambda *_args: "bf16-composite",
    )

    assert not torch.is_autocast_enabled("mojo")
    assert (
        mojo_device_aten_ops.mojo_device_cross_entropy_loss(logits, target)
        == "bf16-composite"
    )
    assert allocations == []

    with torch.amp.autocast("mojo", dtype=torch.bfloat16):
        assert torch.is_autocast_enabled("mojo")
        loss = mojo_device_aten_ops.mojo_device_cross_entropy_loss(logits, target)
    assert loss._dtype == aten_fast.DType.float32
    assert loss is allocations[0]


def test_missing_fused_sources_keep_real_mojo_composite_autograd(
    mojo_gpu_available: bool,
) -> None:
    """Regression for direct registrations shadowing the composite body.

    This deliberately exercises the real H100 constituent path only while the
    optional fused sources are absent.  Host-only test runs may select the
    other tests in this file without launching this GPU regression.
    """
    from torch_mojo_backend import eager_kernels, get_accelerators
    from torch_mojo_backend import register_mojo_devices

    if not mojo_gpu_available:
        pytest.skip("MAX has no GPU")
    accelerator = list(get_accelerators())[0]
    if accelerator.api != "cuda" or accelerator.architecture_name != "sm_90a":
        pytest.skip("the BF16 cross-entropy regression requires an H100")
    if any(path.is_file() for path in aten_fast._BF16_CROSS_ENTROPY_SOURCE_PATHS):
        pytest.skip("the fused BF16 cross-entropy source is installed")

    register_mojo_devices()
    generator = torch.Generator().manual_seed(20260719)
    host_logits = torch.randn(7, 65, generator=generator).to(torch.bfloat16)
    target = torch.tensor([1, 7, -1, 13, 64, 2, -1])

    reference = host_logits.clone().requires_grad_()
    reference_log_probs = torch.nn.functional.log_softmax(reference, dim=-1)
    reference_loss = torch.nn.functional.nll_loss(
        reference_log_probs.float(), target, ignore_index=-1
    )
    reference_loss.backward()

    calls = {"forward": 0, "backward": 0}
    original_forward = eager_kernels.loss_ops.NllLossForwardF32
    original_backward = eager_kernels.loss_ops.NllLossBackwardF32

    def forward(*args):
        calls["forward"] += 1
        return original_forward(*args)

    def backward(*args):
        calls["backward"] += 1
        return original_backward(*args)

    eager_kernels.loss_ops.NllLossForwardF32 = forward
    eager_kernels.loss_ops.NllLossBackwardF32 = backward
    try:
        actual = host_logits.to("mojo:0").requires_grad_()
        with torch.amp.autocast("mojo", dtype=torch.bfloat16):
            actual_loss = torch.nn.functional.cross_entropy(
                actual, target.to("mojo:0"), ignore_index=-1
            )
        actual_loss.backward()
    finally:
        eager_kernels.loss_ops.NllLossForwardF32 = original_forward
        eager_kernels.loss_ops.NllLossBackwardF32 = original_backward

    assert calls == {"forward": 1, "backward": 1}
    assert actual_loss.dtype == torch.float32
    assert actual.grad is not None
    assert actual.grad.dtype == torch.bfloat16
    torch.testing.assert_close(actual_loss.cpu(), reference_loss, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(actual.grad.cpu(), reference.grad, atol=4e-3, rtol=4e-3)


def _source_present_h100_or_skip(mojo_gpu_available: bool) -> str:
    """Return the real H100 device only once both optional sources exist."""
    if not all(path.is_file() for path in aten_fast._BF16_CROSS_ENTROPY_SOURCE_PATHS):
        pytest.skip("the fused BF16 cross-entropy source is not installed")
    if not mojo_gpu_available:
        pytest.skip("MAX has no GPU")

    from torch_mojo_backend import get_accelerators, register_mojo_devices

    accelerator = list(get_accelerators())[0]
    if accelerator.api != "cuda" or accelerator.architecture_name != "sm_90a":
        pytest.skip("the fused BF16 cross-entropy path requires an H100")
    register_mojo_devices()
    return "mojo:0"


def _cuda_amp_bf16_cross_entropy_oracle(
    logits: torch.Tensor, target: torch.Tensor, *, ignore_index: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU oracle for the fused path's CUDA BF16 AMP rounding boundaries."""
    logits_f32 = logits.float()
    row_max = logits_f32.amax(dim=-1)
    row_logsum = torch.exp(logits_f32 - row_max[:, None]).sum(dim=-1).log()
    log_probs_bf16 = ((logits_f32 - row_max[:, None]) - row_logsum[:, None]).to(
        torch.bfloat16
    )

    valid = target != ignore_index
    valid_rows = valid.nonzero(as_tuple=False).flatten()
    if valid_rows.numel() == 0:
        loss = torch.tensor(float("nan"), dtype=torch.float32)
    else:
        valid_targets = target[valid]
        loss = -log_probs_bf16[valid, valid_targets].float().sum()
        loss /= float(valid_rows.numel())

    # CUDA's BF16 NLL backward rounds this scale before log-softmax backward.
    # Keep the zero NLL rows in the composite instead of overwriting them after
    # it: exp(NaN) * +0 is NaN, including when every target is ignored.
    nll_grad = torch.zeros_like(logits_f32)
    if valid_rows.numel() != 0:
        scale = torch.tensor(-1.0 / float(valid_rows.numel()), dtype=torch.float32).to(
            torch.bfloat16
        )
        nll_grad[valid_rows, target[valid]] = scale.float()
    probability = torch.exp(log_probs_bf16.float())
    grad_f32 = nll_grad - probability * nll_grad.sum(dim=-1, keepdim=True)
    return loss, grad_f32.to(torch.bfloat16)


def test_cuda_amp_oracle_preserves_all_ignored_nonfinite_composite() -> None:
    logits = torch.tensor(
        [[float("nan"), float("inf"), float("-inf")], [0.0, 0.0, 0.0]],
        dtype=torch.bfloat16,
    )
    loss, grad = _cuda_amp_bf16_cross_entropy_oracle(
        logits, torch.tensor([-1, -1]), ignore_index=-1
    )

    assert torch.isnan(loss)
    assert torch.isnan(grad[0]).all()
    assert torch.equal(grad[1], torch.zeros(3, dtype=torch.bfloat16))
    assert not torch.signbit(grad[1]).any()

    mixed_loss, mixed_grad = _cuda_amp_bf16_cross_entropy_oracle(
        logits, torch.tensor([-1, 1]), ignore_index=-1
    )
    assert mixed_loss == torch.tensor(1.1015625, dtype=torch.float32)
    torch.testing.assert_close(
        mixed_grad,
        torch.tensor(
            [
                [float("nan"), float("nan"), float("nan")],
                [0.33203125, -0.66796875, 0.33203125],
            ],
            dtype=torch.bfloat16,
        ),
        atol=0.0,
        rtol=0.0,
        equal_nan=True,
    )


def test_source_present_fused_cross_entropy_forward_backward(
    mojo_gpu_available: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the real pointer bridge, saved row state, and BF16 backward."""
    mojo_h100 = _source_present_h100_or_skip(mojo_gpu_available)

    from torch_mojo_backend import eager_kernels

    module = eager_kernels.bf16_cross_entropy_ops
    original_forward = module.Bf16CrossEntropyForward
    original_backward = module.Bf16CrossEntropyBackward
    calls = {"forward": 0, "backward": 0}

    def forward(*args):
        calls["forward"] += 1
        return original_forward(*args)

    def backward(*args):
        calls["backward"] += 1
        return original_backward(*args)

    monkeypatch.setattr(module, "Bf16CrossEntropyForward", forward)
    monkeypatch.setattr(module, "Bf16CrossEntropyBackward", backward)

    generator = torch.Generator().manual_seed(20260719)
    host_logits = (torch.randn(19, 65, generator=generator) * 2.0).to(torch.bfloat16)
    target = torch.tensor(
        [1, 7, -1, 13, 64, 2, 5, 8, -1, 34, 21, 0, 63, 4, 17, -1, 9, 41, 3]
    )
    expected_loss, expected_grad = _cuda_amp_bf16_cross_entropy_oracle(
        host_logits, target, ignore_index=-1
    )

    actual = host_logits.to(mojo_h100).requires_grad_()
    actual_target = target.to(mojo_h100)
    with torch.amp.autocast("mojo", dtype=torch.bfloat16):
        assert aten_fast.bf16_cross_entropy_supported(
            actual, actual_target, ignore_index=-1, require_backward=True
        )
        actual_loss = torch.nn.functional.cross_entropy(
            actual, actual_target, ignore_index=-1
        )
    actual_loss.backward()

    assert calls == {"forward": 1, "backward": 1}
    assert actual_loss.dtype == torch.float32
    assert actual.grad is not None
    assert actual.grad.dtype == torch.bfloat16
    torch.testing.assert_close(actual_loss.cpu(), expected_loss, atol=1e-3, rtol=1e-4)
    torch.testing.assert_close(actual.grad.cpu(), expected_grad, atol=4e-3, rtol=0.0)


def test_source_present_fused_cross_entropy_rejects_base_mutation_of_logits_view(
    mojo_gpu_available: bool,
) -> None:
    """The nanoGPT public view must share the saved logits version counter."""
    mojo_h100 = _source_present_h100_or_skip(mojo_gpu_available)
    generator = torch.Generator().manual_seed(20260720)
    base = (
        torch.randn(3, 7, 65, generator=generator)
        .to(torch.bfloat16)
        .to(mojo_h100)
        .requires_grad_()
    )
    logits = base.view(-1, 65)
    target = (
        torch.arange(logits.shape[0], dtype=torch.int64).remainder(65).to(mojo_h100)
    )

    with torch.amp.autocast("mojo", dtype=torch.bfloat16):
        assert aten_fast.bf16_cross_entropy_supported(
            logits, target, require_backward=True
        )
        loss = torch.nn.functional.cross_entropy(logits, target)

    version = logits._version
    with torch.no_grad():
        base.add_(1.0)
    assert logits._version == version + 1
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        loss.backward()


def test_source_present_fused_cross_entropy_matches_cuda_nonfinite_semantics(
    mojo_gpu_available: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ignored and targeted nonfinite rows preserve CUDA BF16 arithmetic."""
    mojo_h100 = _source_present_h100_or_skip(mojo_gpu_available)

    from torch_mojo_backend import eager_kernels

    module = eager_kernels.bf16_cross_entropy_ops
    original_forward = module.Bf16CrossEntropyForward
    original_backward = module.Bf16CrossEntropyBackward
    calls = {"forward": 0, "backward": 0}

    def forward(*args):
        calls["forward"] += 1
        return original_forward(*args)

    def backward(*args):
        calls["backward"] += 1
        return original_backward(*args)

    monkeypatch.setattr(module, "Bf16CrossEntropyForward", forward)
    monkeypatch.setattr(module, "Bf16CrossEntropyBackward", backward)
    cases = (
        (
            torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.bfloat16),
            torch.tensor([-1]),
            float("nan"),
            torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.bfloat16),
        ),
        (
            torch.tensor(
                [[float("nan"), float("inf"), float("-inf")]], dtype=torch.bfloat16
            ),
            torch.tensor([-1]),
            float("nan"),
            torch.tensor(
                [[float("nan"), float("nan"), float("nan")]], dtype=torch.bfloat16
            ),
        ),
        (
            torch.tensor(
                [[float("nan"), float("inf"), float("-inf")], [0.0, 0.0, 0.0]],
                dtype=torch.bfloat16,
            ),
            torch.tensor([-1, 1]),
            1.1015625,
            torch.tensor(
                [
                    [float("nan"), float("nan"), float("nan")],
                    [0.33203125, -0.66796875, 0.33203125],
                ],
                dtype=torch.bfloat16,
            ),
        ),
        (
            torch.tensor([[float("-inf"), 0.0, 1.0]], dtype=torch.bfloat16),
            torch.tensor([0]),
            float("inf"),
            torch.tensor([[-1.0, 0.26953125, 0.73046875]], dtype=torch.bfloat16),
        ),
    )

    for host_logits, host_target, expected_loss, expected_grad in cases:
        actual = host_logits.to(mojo_h100).requires_grad_()
        target = host_target.to(mojo_h100)
        with torch.amp.autocast("mojo", dtype=torch.bfloat16):
            assert aten_fast.bf16_cross_entropy_supported(
                actual, target, ignore_index=-1, require_backward=True
            )
            loss = torch.nn.functional.cross_entropy(actual, target, ignore_index=-1)
        loss.backward()

        assert loss.dtype == torch.float32
        assert actual.grad is not None
        assert actual.grad.dtype == torch.bfloat16
        torch.testing.assert_close(
            loss.cpu(),
            torch.tensor(expected_loss, dtype=torch.float32),
            atol=0.0,
            rtol=0.0,
            equal_nan=True,
        )
        actual_grad_cpu = actual.grad.cpu()
        torch.testing.assert_close(
            actual_grad_cpu, expected_grad, atol=0.0, rtol=0.0, equal_nan=True
        )
        if torch.isfinite(host_logits).all() and bool((host_target == -1).all()):
            assert not torch.signbit(actual_grad_cpu).any()

    assert calls == {"forward": len(cases), "backward": len(cases)}
