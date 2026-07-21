"""Tests for the Mojo-extension fast path used by mojo eager mode."""

import weakref

import pytest
import torch

from torch_mojo_backend import get_accelerators, register_mojo_devices
from torch_mojo_backend.flags import fast_eager_enabled

pytestmark = pytest.mark.xdist_group(name="group1")


@pytest.fixture(autouse=True)
def setup_max_device():
    register_mojo_devices()


BINARY_OPS = [torch.add, torch.sub, torch.mul, torch.div, torch.maximum, torch.minimum]
UNARY_OPS = [torch.relu, torch.exp]


@pytest.mark.parametrize("op", BINARY_OPS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_fast_binary_ops_match_cpu(mojo_device, op, dtype):
    x = torch.randn(33, 65).to(dtype)
    y = torch.randn(33, 65).to(dtype) + 1.5  # avoid div-by-~0
    result = op(x.to(mojo_device), y.to(mojo_device))
    torch.testing.assert_close(result.cpu(), op(x, y))


@pytest.mark.parametrize("op", UNARY_OPS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_fast_unary_ops_match_cpu(mojo_device, op, dtype):
    x = torch.randn(33, 65).to(dtype)
    result = op(x.to(mojo_device))
    torch.testing.assert_close(result.cpu(), op(x))


def test_fast_log1p_preserves_small_values(mojo_device):
    x = torch.tensor([1e-10, -1e-10, 1e-8, -1e-8, 1e-6, -1e-6])
    result = torch.log1p(x.to(mojo_device)).cpu()
    torch.testing.assert_close(result, torch.log1p(x), rtol=2e-6, atol=0)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_fast_binary_int_dtypes(mojo_device, dtype):
    x = torch.arange(100, dtype=dtype)
    y = torch.arange(100, dtype=dtype) * 3
    result = (x.to(mojo_device) + y.to(mojo_device)).cpu()
    torch.testing.assert_close(result, x + y)


def test_fast_path_is_used(mojo_device):
    """The eligible case must go through the Mojo kernel, not the fallback.

    Tensor-tensor adds route through the shared spec op, or through the
    Apple flat kernel selected during Metal device registration."""
    if not fast_eager_enabled():
        pytest.skip("fast eager path disabled")
    from torch_mojo_backend import eager_kernels

    calls = []
    x = torch.randn(8, 8).to(mojo_device)
    y = torch.randn(8, 8).to(mojo_device)
    if x._device.api == "metal":
        module = eager_kernels.elementwise_ops
        name = "Add"
    else:
        module = eager_kernels.logic_ops
        name = "AddSpec"
    original = getattr(module, name)

    def spy(*args):
        calls.append(args)
        return original(*args)

    setattr(module, name, spy)
    try:
        _ = x + y
    finally:
        setattr(module, name, original)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "module_name,spec_name,fn",
    [
        ("logic_ops", "SubSpec", lambda x, y: x - y),
        ("logic_ops", "EqSpec", lambda x, y: x == y),
        ("elementwise_ops", "SigmoidSpec", lambda x, y: torch.sigmoid(x)),
        ("elementwise_ops", "MulScalarSpec", lambda x, y: x * 2.0),
        ("reduction_ops", "SumSpec", lambda x, y: x.sum(-1)),
        ("nn_ops", "SoftmaxSpec", lambda x, y: torch.softmax(x, -1)),
        ("matmul_ops", "MatmulSpec", lambda x, y: x @ y),
    ],
)
def test_spec_path_is_used(mojo_device, module_name, spec_name, fn):
    """One representative op per converted family must route through its
    spec entry (whole prologue in one Mojo call), not the classic chain."""
    if not fast_eager_enabled():
        pytest.skip("fast eager path disabled")
    from torch_mojo_backend import eager_kernels

    module = getattr(eager_kernels, module_name)
    calls = []
    original = getattr(module, spec_name)

    def spy(*args):
        calls.append(args)
        return original(*args)

    setattr(module, spec_name, spy)
    try:
        x = torch.randn(8, 8).to(mojo_device)
        y = torch.randn(8, 8).to(mojo_device)
        _ = fn(x, y)
    finally:
        setattr(module, spec_name, original)
    assert len(calls) == 1


def test_fallback_broadcast(mojo_device):
    x = torch.randn(16, 16)
    y = torch.randn(16)
    result = (x.to(mojo_device) + y.to(mojo_device)).cpu()
    torch.testing.assert_close(result, x + y)


def test_fallback_scalar_other(mojo_device):
    x = torch.randn(16, 16)
    result = (x.to(mojo_device) + 2.5).cpu()
    torch.testing.assert_close(result, x + 2.5)


def test_fallback_alpha(mojo_device):
    x = torch.randn(16, 16)
    y = torch.randn(16, 16)
    result = torch.add(x.to(mojo_device), y.to(mojo_device), alpha=2.0).cpu()
    torch.testing.assert_close(result, torch.add(x, y, alpha=2.0))


def test_fallback_int_div(mojo_device):
    x = torch.arange(1, 65, dtype=torch.int32)
    y = torch.full((64,), 4, dtype=torch.int32)
    result = (x.to(mojo_device) / y.to(mojo_device)).cpu()
    # check_dtype=False: the graph-based fallback path promotes int div to
    # float64 where torch gives float32 — a pre-existing deviation
    # (reproduces with TORCH_MOJO_BACKEND_FAST_EAGER=0).
    torch.testing.assert_close(result, x / y, check_dtype=False)


@pytest.mark.parametrize("shape", [(0,), (1,), (7,), (0, 5)])
def test_edge_case_shapes(mojo_device, shape):
    x = torch.randn(*shape)
    y = torch.randn(*shape)
    result = (x.to(mojo_device) + y.to(mojo_device)).cpu()
    torch.testing.assert_close(result, x + y)


def test_chained_fast_ops(mojo_device):
    """Outputs of fast ops must be valid inputs to further fast ops."""
    x = torch.randn(32, 32)
    y = torch.randn(32, 32)
    device_result = x.to(mojo_device)
    for _ in range(5):
        device_result = torch.relu(
            device_result * y.to(mojo_device) + y.to(mojo_device)
        )
    expected = x
    for _ in range(5):
        expected = torch.relu(expected * y + y)
    torch.testing.assert_close(device_result.cpu(), expected)


@pytest.fixture
def mojo_gpu(mojo_gpu_available: bool):
    """GPU mojo device only — for ops whose fast path is GPU-gated."""
    if not mojo_gpu_available:
        pytest.skip("You do not have a GPU supported by MAX")
    return "mojo:0"


def test_fast_view_family(mojo_device):
    x = torch.randn(2, 6, 768)
    xd = x.to(mojo_device)
    torch.testing.assert_close(xd.view(-1, 768).cpu(), x.view(-1, 768))
    torch.testing.assert_close(xd.reshape(12, 768).cpu(), x.reshape(12, 768))
    torch.testing.assert_close(xd.unsqueeze(0).cpu(), x.unsqueeze(0))


def test_fast_view_aliases_storage(mojo_device):
    """The fast view must alias, matching torch.Tensor.view semantics."""
    x = torch.zeros(4, 4).to(mojo_device)
    v = x.view(16)
    x += torch.ones(4, 4).to(mojo_device)
    torch.testing.assert_close(v.cpu(), torch.ones(16))


@pytest.mark.parametrize("dims", [(0, 1), (1, 2), (-1, -2)])
def test_fast_transpose(mojo_device, dims):
    x = torch.randn(2, 3, 4)
    result = x.to(mojo_device).transpose(*dims).contiguous().cpu()
    torch.testing.assert_close(result, x.transpose(*dims).contiguous())


def test_fast_t(mojo_device):
    x = torch.randn(50, 30)
    torch.testing.assert_close(
        x.to(mojo_device).t().contiguous().cpu(), x.t().contiguous()
    )


@pytest.mark.parametrize("split_size,dim", [(768, 2), (2, 0), ([1, 2, 3], 1)])
def test_fast_split(mojo_device, split_size, dim):
    x = torch.randn(4, 6, 2304)
    dev_parts = x.to(mojo_device).split(split_size, dim=dim)
    for dev_part, ref_part in zip(dev_parts, x.split(split_size, dim=dim)):
        torch.testing.assert_close(dev_part.cpu(), ref_part)


def test_fast_cat_skips_legacy_empty(mojo_device):
    empty = torch.empty(0)
    x = torch.randn(1, 12, 6, 64)
    result = torch.cat([empty.to(mojo_device), x.to(mojo_device)], dim=-2)
    torch.testing.assert_close(result.cpu(), torch.cat([empty, x], dim=-2))


def test_fast_batch_norm_inference(mojo_device):
    x = torch.randn(2, 64, 14, 14)
    bn = torch.nn.BatchNorm2d(64).eval()
    bn.running_mean.normal_()
    bn.running_var.uniform_(0.5, 2.0)
    bn_dev = torch.nn.BatchNorm2d(64).eval()
    bn_dev.load_state_dict(bn.state_dict())
    bn_dev = bn_dev.to(mojo_device)
    with torch.no_grad():
        torch.testing.assert_close(
            bn_dev(x.to(mojo_device)).cpu(), bn(x), atol=1e-5, rtol=1e-5
        )


def test_fast_layer_norm(mojo_device):
    x = torch.randn(2, 6, 768)
    ln = torch.nn.LayerNorm(768).eval()
    with torch.no_grad():
        ln.weight.normal_()
        ln.bias.normal_()
    ln_dev = torch.nn.LayerNorm(768).eval()
    ln_dev.load_state_dict(ln.state_dict())
    ln_dev = ln_dev.to(mojo_device)
    with torch.no_grad():
        torch.testing.assert_close(
            ln_dev(x.to(mojo_device)).cpu(), ln(x), atol=1e-5, rtol=1e-5
        )


def test_fast_native_layer_norm_stats(mojo_device):
    x = torch.randn(1, 6, 768).to(mojo_device)
    w = torch.ones(768).to(mojo_device)
    b = torch.zeros(768).to(mojo_device)
    out, mean, rstd = torch.native_layer_norm(x, [768], w, b, 1e-5)
    ref_out, ref_mean, ref_rstd = torch.native_layer_norm(
        x.cpu(), [768], w.cpu(), b.cpu(), 1e-5
    )
    torch.testing.assert_close(out.cpu(), ref_out, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(mean.cpu(), ref_mean, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(rstd.cpu(), ref_rstd, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("keepdim", [True, False])
def test_fast_mean_trailing_dims(mojo_device, keepdim):
    x = torch.randn(1, 512, 7, 7)
    result = x.to(mojo_device).mean([-1, -2], keepdim=keepdim).cpu()
    torch.testing.assert_close(result, x.mean([-1, -2], keepdim=keepdim))


def test_fast_reduction_library_tier(mojo_device):
    """Huge-col reductions route to the stdlib reduction library (GPU: rows <=
    128 and cols >= 2**20; MAX-CPU: always) — exercise that tier, which no other
    test reaches. Integer-valued floats keep every f32 partial sum exact, so the
    comparisons are bit-exact instead of tolerance-based."""
    # rows == 1: full-reduction layout (the two-phase GPU tier's main case).
    x = torch.randint(-4, 5, (1, 2**20 + 7)).float()
    xd = x.to(mojo_device)
    torch.testing.assert_close(xd.sum(-1).cpu(), x.sum(-1))
    torch.testing.assert_close(xd.amax(-1).cpu(), x.amax(-1))
    torch.testing.assert_close(torch.any(xd, -1).cpu(), torch.any(x, -1))

    # rows == 128 (gate boundary): per-row outputs must land in the right rows.
    y = torch.randint(-4, 5, (128, 2**20)).float()
    y[5] = 0.0  # give any() a False row
    yd = y.to(mojo_device)
    torch.testing.assert_close(yd.sum(-1).cpu(), y.sum(-1))
    torch.testing.assert_close(yd.amax(-1).cpu(), y.amax(-1))
    torch.testing.assert_close(torch.any(yd, -1).cpu(), torch.any(y, -1))


def test_fast_anyall_nan_is_truthy(mojo_device):
    """torch treats NaN as truthy in any/all. Cover both dispatch tiers: the
    small shape uses the block kernel on GPU (and the library on MAX-CPU), the
    huge shape uses the library tier everywhere."""
    small_any = torch.zeros(2, 100)
    small_any[0, 0] = float("nan")
    small_all = torch.full((2, 100), float("nan"))
    huge_any = torch.zeros(1, 2**20 + 7)
    huge_any[0, 12345] = float("nan")
    huge_all = torch.ones(1, 2**20 + 7)
    huge_all[0, 999] = float("nan")
    for x in (small_any, huge_any):
        got = torch.any(x.to(mojo_device), -1).cpu()
        torch.testing.assert_close(got, torch.any(x, -1))
    for x in (small_all, huge_all):
        got = torch.all(x.to(mojo_device), -1).cpu()
        torch.testing.assert_close(got, torch.all(x, -1))


def test_fast_max_pool2d(mojo_device):
    x = torch.randn(1, 64, 32, 32)
    result = torch.nn.functional.max_pool2d(x.to(mojo_device), 3, 2, 1).cpu()
    torch.testing.assert_close(result, torch.nn.functional.max_pool2d(x, 3, 2, 1))


def test_fast_max_pool2d_indices(mojo_device):
    x = torch.randn(1, 8, 16, 16)
    dev_vals, dev_idx = torch.nn.functional.max_pool2d(
        x.to(mojo_device), 2, 2, return_indices=True
    )
    ref_vals, ref_idx = torch.nn.functional.max_pool2d(x, 2, 2, return_indices=True)
    torch.testing.assert_close(dev_vals.cpu(), ref_vals)
    torch.testing.assert_close(dev_idx.cpu(), ref_idx)


def test_fast_embedding(mojo_device):
    weight = torch.randn(100, 32)
    idx = torch.randint(0, 100, (2, 5))
    result = torch.nn.functional.embedding(idx.to(mojo_device), weight.to(mojo_device))
    torch.testing.assert_close(result.cpu(), torch.nn.functional.embedding(idx, weight))


def test_fast_scalar_elementwise(mojo_device):
    x = torch.randn(2, 6, 3072)
    xd = x.to(mojo_device)
    torch.testing.assert_close((xd * 0.5).cpu(), x * 0.5)
    torch.testing.assert_close((xd + 1.0).cpu(), x + 1.0)
    torch.testing.assert_close((xd**3.0).cpu(), x**3.0, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(torch.tanh(xd).cpu(), torch.tanh(x))


def test_fast_gpu_portability_kernels(mojo_device):
    # These closures previously captured Float64 or instantiated host-only
    # code for the GPU target, which is rejected by Metal and gfx942.
    base = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    device_base = base.to(mojo_device)
    device_base.t().fill_(2.5)
    torch.testing.assert_close(device_base.cpu(), torch.full_like(base, 2.5))

    index = torch.tensor([[0, 2, 1], [1, 0, 2]])
    source = torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    scattered = (
        torch.zeros_like(source)
        .to(mojo_device)
        .scatter(1, index.to(mojo_device), source.to(mojo_device))
    )
    torch.testing.assert_close(
        scattered.cpu(), torch.zeros_like(source).scatter(1, index, source)
    )

    a = torch.randn(2, 3)
    b = torch.randn(2, 3)
    c = torch.randn(2, 3)
    result = torch.addcmul(
        a.to(mojo_device), b.to(mojo_device), c.to(mojo_device), value=0.125
    )
    torch.testing.assert_close(result.cpu(), torch.addcmul(a, b, c, value=0.125))


def test_fast_add_scalar_int(mojo_device):
    x = torch.arange(6)
    torch.testing.assert_close((x.to(mojo_device) + 3).cpu(), x + 3)


def test_fast_add_inplace(mojo_device):
    x = torch.randn(4, 4)
    y = torch.randn(4, 4)
    xd = x.clone().to(mojo_device)
    xd += y.to(mojo_device)
    torch.testing.assert_close(xd.cpu(), x + y)


def test_fast_all_and_item(mojo_device):
    ones = torch.ones(1, 6, dtype=torch.bool).to(mojo_device)
    assert bool(ones.all().item()) is True
    mixed = torch.tensor([[True, False, True]]).to(mojo_device)
    assert bool(mixed.all().item()) is False


def test_fast_arange(mojo_device):
    torch.testing.assert_close(
        torch.arange(6, device=mojo_device).cpu(), torch.arange(6)
    )
    torch.testing.assert_close(
        torch.arange(2, 20, 3, device=mojo_device).cpu(), torch.arange(2, 20, 3)
    )


def test_fast_arange_uses_device_accumulator(mojo_device):
    args = (16_777_217.0, 16_777_227.0, 1.0)
    result = torch.arange(*args, dtype=torch.float32, device=mojo_device).cpu()
    cpu_index = len(list(get_accelerators())) - 1
    if mojo_device == f"mojo:{cpu_index}":
        # PyTorch's CPU kernel specifies a float64 accumulator for float32.
        # Build that scalar reference explicitly: arm64's vectorized kernel
        # has platform-specific intermediate rounding at this boundary.
        expected = torch.tensor(
            [args[0] + i * args[2] for i in range(10)], dtype=torch.float32
        )
    elif torch.cuda.is_available():
        expected = torch.arange(*args, dtype=torch.float32, device="cuda").cpu()
    elif torch.backends.mps.is_available():
        expected = torch.arange(*args, dtype=torch.float32, device="mps").cpu()
    else:
        pytest.skip("no native GPU reference for MAX accelerator")
    assert torch.equal(result, expected)


def test_fast_cast(mojo_device):
    x = torch.randint(0, 3, (1, 6))
    torch.testing.assert_close(x.to(mojo_device).to(torch.bool).cpu(), x.to(torch.bool))
    f = torch.randn(3, 4)
    torch.testing.assert_close(
        f.to(mojo_device).to(torch.float16).cpu(), f.to(torch.float16)
    )


def test_fast_float64_factories_fill_scatter_and_arange(mojo_gpu):
    if list(get_accelerators())[0].api == "metal":
        pytest.skip("Metal does not support float64 kernels")

    ones = torch.ones(5, dtype=torch.float64, device=mojo_gpu)
    torch.testing.assert_close(ones.cpu(), torch.ones(5, dtype=torch.float64))

    values = torch.arange(5, dtype=torch.float64).to(mojo_gpu)
    values.fill_(2.5)
    torch.testing.assert_close(values.cpu(), torch.full((5,), 2.5, dtype=torch.float64))

    base = torch.zeros(5, dtype=torch.float64).to(mojo_gpu)
    index = torch.tensor([1, 3], dtype=torch.int64).to(mojo_gpu)
    source = torch.tensor([4.0, 7.0], dtype=torch.float64).to(mojo_gpu)
    scattered = base.scatter(0, index, source).cpu()
    torch.testing.assert_close(
        scattered, torch.tensor([0.0, 4.0, 0.0, 7.0, 0.0], dtype=torch.float64)
    )

    result = torch.arange(0.0, 2.0, 0.25, dtype=torch.float64, device=mojo_gpu)
    torch.testing.assert_close(
        result.cpu(), torch.arange(0.0, 2.0, 0.25, dtype=torch.float64)
    )


# ---- GPU-only fast paths (matmul / conv / attention via MAX kernel library)


def test_fast_mm_addmm(mojo_gpu):
    a = torch.randn(6, 768)
    b = torch.randn(768, 2304)
    bias = torch.randn(2304)
    dev = torch.addmm(bias.to(mojo_gpu), a.to(mojo_gpu), b.to(mojo_gpu)).cpu()
    # TF32-level tolerance: the MAX matmul kernels (same as graph mode) use
    # tensor cores for float32.
    torch.testing.assert_close(dev, torch.addmm(bias, a, b), atol=5e-2, rtol=5e-2)
    dev = (a.to(mojo_gpu) @ b.to(mojo_gpu)).cpu()
    torch.testing.assert_close(dev, a @ b, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize(
    "in_features,out_features", [(768, 2304), (4096, 1024), (992, 3001), (768, 50257)]
)
def test_fast_linear_gfx942_dynamic_mfma(mojo_gpu, in_features, out_features):
    if list(get_accelerators())[0].architecture_name != "gfx942":
        pytest.skip("the dynamic MFMA kernels target gfx942")

    x = torch.randn(256, in_features)
    weight = torch.randn(out_features, in_features)
    bias = torch.randn(out_features)
    dev = torch.nn.functional.linear(
        x.to(mojo_gpu), weight.to(mojo_gpu), bias.to(mojo_gpu)
    ).cpu()
    ref = torch.nn.functional.linear(x, weight, bias)
    torch.testing.assert_close(dev, ref, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize(
    "in_features,out_features", [(768, 768), (1024, 4096), (4096, 1024), (992, 3001)]
)
def test_fast_addmm_gfx942_dynamic_mfma(mojo_gpu, in_features, out_features):
    if list(get_accelerators())[0].architecture_name != "gfx942":
        pytest.skip("the dynamic MFMA kernels target gfx942")

    x = torch.randn(256, in_features)
    weight = torch.randn(in_features, out_features)
    bias = torch.randn(out_features)
    dev_x = x.to(mojo_gpu)
    dev_weight = weight.to(mojo_gpu)
    dev_bias = bias.to(mojo_gpu)
    # Queue repeated launches before synchronizing. This catches invalid
    # tile schedules whose shared-memory race is hidden by a single launch.
    dev_outputs = [torch.addmm(dev_bias, dev_x, dev_weight) for _ in range(3)]
    dev = [output.cpu() for output in dev_outputs]
    ref = torch.addmm(bias, x, weight)
    for actual in dev:
        torch.testing.assert_close(actual, ref, atol=5e-2, rtol=5e-2)
    assert torch.equal(dev[0], dev[1])
    assert torch.equal(dev[0], dev[2])


@pytest.mark.parametrize("batch", [64, 257, 512])
def test_fast_addmm_gfx942_dynamic_batch_mfma(mojo_gpu, batch):
    if list(get_accelerators())[0].architecture_name != "gfx942":
        pytest.skip("the dynamic MFMA kernels target gfx942")

    # A K-dominant projection selects that shape regime without embedding
    # these dimensions in the kernel. The non-tile-aligned M covers its edge.
    x = torch.randn(batch, 4096)
    weight = torch.randn(4096, 1024)
    bias = torch.randn(1024)
    dev_x = x.to(mojo_gpu)
    dev_weight = weight.to(mojo_gpu)
    dev_bias = bias.to(mojo_gpu)
    outputs = [torch.addmm(dev_bias, dev_x, dev_weight) for _ in range(3)]
    actual = [output.cpu() for output in outputs]
    ref = torch.addmm(bias, x, weight)
    for output in actual:
        torch.testing.assert_close(output, ref, atol=5e-2, rtol=5e-2)
    assert torch.equal(actual[0], actual[1])
    assert torch.equal(actual[0], actual[2])


def test_fast_addmm_gfx942_unaligned_k(mojo_gpu):
    if list(get_accelerators())[0].architecture_name != "gfx942":
        pytest.skip("the dynamic MFMA dispatch targets gfx942")

    # K values outside the MFMA tile-alignment regime retain the general
    # dynamic GEMM path; this is a regime fallback, not a model-shape gate.
    x = torch.randn(65, 1000)
    weight = torch.randn(1000, 257)
    bias = torch.randn(257)
    actual = torch.addmm(bias.to(mojo_gpu), x.to(mojo_gpu), weight.to(mojo_gpu)).cpu()
    torch.testing.assert_close(
        actual, torch.addmm(bias, x, weight), atol=5e-2, rtol=5e-2
    )


def test_fast_gpt2_decode_attention_with_strided_kv(mojo_gpu):
    batch, heads, seq_len, capacity, head_dim = 4, 12, 8, 16, 64
    query = torch.randn(batch, heads, 1, head_dim)
    key_storage = torch.randn(batch, heads, capacity, head_dim)
    value_storage = torch.randn(batch, heads, capacity, head_dim)
    key = key_storage[:, :, :seq_len, :]
    value = value_storage[:, :, :seq_len, :]

    dev_key_storage = key_storage.to(mojo_gpu)
    dev_value_storage = value_storage.to(mojo_gpu)
    dev_key = dev_key_storage[:, :, :seq_len, :]
    dev_value = dev_value_storage[:, :, :seq_len, :]
    actual = torch.nn.functional.scaled_dot_product_attention(
        query.to(mojo_gpu), dev_key, dev_value
    ).cpu()
    ref = torch.nn.functional.scaled_dot_product_attention(query, key, value)
    torch.testing.assert_close(actual, ref, atol=2e-4, rtol=2e-4)


def test_fast_gpt2_logits_argmax(mojo_gpu):
    if list(get_accelerators())[0].architecture_name != "gfx942":
        pytest.skip("the GPT-2 argmax specialization targets gfx942")

    logits = torch.randn(256, 50257)
    logits[:, 123] = 100.0
    actual = torch.argmax(logits.to(mojo_gpu), dim=-1).cpu()
    torch.testing.assert_close(actual, torch.argmax(logits, dim=-1))


def test_fast_bmm(mojo_gpu):
    a = torch.randn(12, 6, 64)
    b = torch.randn(12, 64, 6)
    dev = torch.bmm(a.to(mojo_gpu), b.to(mojo_gpu)).cpu()
    torch.testing.assert_close(dev, torch.bmm(a, b), atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_fast_mm_degenerate_dims(mojo_device, dtype):
    # n == 1 used to segfault the CPU library-matmul route (gemv special
    # case without a DeviceContext); m == 1 / k == 1 pinned as regression
    # guards for the library's other special-case routes.
    for m, k, n in [(37, 129, 1), (1, 129, 64), (64, 1, 33), (1, 129, 1)]:
        a = torch.randn(m, k).to(dtype)
        b = torch.randn(k, n).to(dtype)
        dev = torch.mm(a.to(mojo_device), b.to(mojo_device)).cpu()
        ref = (a.float() @ b.float()).to(dtype)
        torch.testing.assert_close(dev, ref, atol=5e-2, rtol=5e-2)
    # batched n == 1 shares the same path
    a3 = torch.randn(4, 8, 129).to(dtype)
    b3 = torch.randn(4, 129, 1).to(dtype)
    dev3 = torch.bmm(a3.to(mojo_device), b3.to(mojo_device)).cpu()
    ref3 = torch.bmm(a3.float(), b3.float()).to(dtype)
    torch.testing.assert_close(dev3, ref3, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_fast_mm_aligned_single_row(mojo_gpu, dtype):
    # This aligned shape selects GEVM on AMD.  Keep both the plain and bias
    # paths covered because GPT-2 decode uses the latter.
    a = torch.randn(1, 128).to(dtype)
    b = torch.randn(128, 64).to(dtype)
    bias = torch.randn(64).to(dtype)

    dev = torch.mm(a.to(mojo_gpu), b.to(mojo_gpu)).cpu()
    ref = (a.float() @ b.float()).to(dtype)
    torch.testing.assert_close(dev, ref, atol=5e-2, rtol=5e-2)

    dev_bias = torch.addmm(bias.to(mojo_gpu), a.to(mojo_gpu), b.to(mojo_gpu)).cpu()
    ref_bias = (a.float() @ b.float() + bias.float()).to(dtype)
    torch.testing.assert_close(dev_bias, ref_bias, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_fast_linear_single_token(mojo_device, dtype):
    # m == 1 with bias: the decode-step GEMV route (on GPU this is
    # modular's gemv_gpu — GEMV_SPLIT_K for f16/bf16 aligned-k — plus the
    # row-broadcast bias epilogue).
    x = torch.randn(1, 768).to(dtype)
    w = torch.randn(96, 768).to(dtype)
    b = torch.randn(96).to(dtype)
    dev = torch.nn.functional.linear(
        x.to(mojo_device), w.to(mojo_device), b.to(mojo_device)
    ).cpu()
    ref = (x.float() @ w.float().t() + b.float()).to(dtype)
    torch.testing.assert_close(dev, ref, atol=5e-2, rtol=5e-2)


def test_fast_linear_skinny_m_large_output(mojo_gpu):
    # GPT-2's batch-32 lm_head takes Apple's 32-row simdgroup-matrix path.
    # Other GPUs retain the skinny-M C-transpose path.
    x = torch.randn(32, 1, 768)
    w = torch.randn(8192, 768)
    dev = torch.nn.functional.linear(x.to(mojo_gpu), w.to(mojo_gpu)).cpu()
    ref = x @ w.t()
    torch.testing.assert_close(dev, ref, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize(
    ("in_features", "out_features"), [(768, 2304), (768, 768), (768, 3072), (3072, 768)]
)
def test_fast_addmm_gpt2_batch32(mojo_gpu, in_features, out_features):
    x = torch.randn(32, in_features)
    w = torch.randn(in_features, out_features)
    bias = torch.randn(out_features)
    dev = torch.addmm(bias.to(mojo_gpu), x.to(mojo_gpu), w.to(mojo_gpu)).cpu()
    ref = torch.addmm(bias, x, w)
    torch.testing.assert_close(dev, ref, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_fast_linear_out_features_one(mojo_device, dtype):
    # out_features == 1 -> transposed-B GEMM with n == 1, plus bias.
    x = torch.randn(37, 129).to(dtype)
    w = torch.randn(1, 129).to(dtype)
    b = torch.randn(1).to(dtype)
    dev = torch.nn.functional.linear(
        x.to(mojo_device), w.to(mojo_device), b.to(mojo_device)
    ).cpu()
    ref = (x.float() @ w.float().t() + b.float()).to(dtype)
    torch.testing.assert_close(dev, ref, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize(
    "in_c,out_c,k,stride,padding,dilation,groups",
    [
        (3, 64, 7, 2, 3, 1, 1),
        (64, 64, 3, 1, 1, 1, 1),
        (64, 128, 1, 2, 0, 1, 1),
        (8, 12, 3, 1, 1, 2, 1),
        (8, 12, 3, 1, 1, 1, 2),
    ],
)
def test_fast_conv2d(mojo_gpu, in_c, out_c, k, stride, padding, dilation, groups):
    x = torch.randn(1, in_c, 32, 32)
    w = torch.randn(out_c, in_c // groups, k, k)
    b = torch.randn(out_c)
    dev = torch.nn.functional.conv2d(
        x.to(mojo_gpu),
        w.to(mojo_gpu),
        b.to(mojo_gpu),
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    ).cpu()
    ref = torch.nn.functional.conv2d(
        x, w, b, stride=stride, padding=padding, dilation=dilation, groups=groups
    )
    torch.testing.assert_close(dev, ref, atol=5e-2, rtol=5e-2)


def test_fast_conv2d_batched_falls_back_correctly(mojo_gpu):
    x = torch.randn(3, 8, 16, 16)
    w = torch.randn(12, 8, 3, 3)
    dev = torch.nn.functional.conv2d(x.to(mojo_gpu), w.to(mojo_gpu), padding=1).cpu()
    ref = torch.nn.functional.conv2d(x, w, padding=1)
    torch.testing.assert_close(dev, ref, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("is_causal", [True, False])
# kv_len <= 32 exercises the library softmax's warp kernel, kv_len=64 the
# online/block kernel.
@pytest.mark.parametrize("kv_len", [6, 10, 64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_fast_sdpa(mojo_gpu, is_causal, kv_len, dtype):
    q = torch.randn(1, 12, 6, 64, dtype=dtype)
    k = torch.randn(1, 12, kv_len, 64, dtype=dtype)
    v = torch.randn(1, 12, kv_len, 64, dtype=dtype)
    dev = torch.nn.functional.scaled_dot_product_attention(
        q.to(mojo_gpu), k.to(mojo_gpu), v.to(mojo_gpu), is_causal=is_causal
    ).cpu()
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
    torch.testing.assert_close(dev, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_fast_sdpa_decode(mojo_gpu, dtype):
    # q_len == 1 selects the fused decode kernel used by GPT-2 generation.
    q = torch.randn(4, 12, 1, 64, dtype=dtype)
    k = torch.randn(4, 12, 128, 64, dtype=dtype)
    v = torch.randn(4, 12, 128, 64, dtype=dtype)
    dev = torch.nn.functional.scaled_dot_product_attention(
        q.to(mojo_gpu), k.to(mojo_gpu), v.to(mojo_gpu)
    ).cpu()
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    torch.testing.assert_close(dev, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("with_bias", [False, True])
def test_fast_linear_training_backward(mojo_gpu, with_bias):
    generator = torch.Generator().manual_seed(20260718)
    x = torch.randn(2, 16, 32, generator=generator)
    weight = torch.randn(64, 32, generator=generator)
    bias = torch.randn(64, generator=generator) if with_bias else None
    grad_output = torch.randn(2, 16, 64, generator=generator)

    reference_inputs = [x.clone().requires_grad_(), weight.clone().requires_grad_()]
    reference_bias = bias.clone().requires_grad_() if bias is not None else None
    torch.nn.functional.linear(*reference_inputs, reference_bias).backward(grad_output)

    mojo_inputs = [tensor.to(mojo_gpu).requires_grad_() for tensor in (x, weight)]
    mojo_bias = bias.to(mojo_gpu).requires_grad_() if bias is not None else None
    torch.nn.functional.linear(*mojo_inputs, mojo_bias).backward(
        grad_output.to(mojo_gpu)
    )

    for actual, expected in zip(mojo_inputs, reference_inputs, strict=True):
        assert actual.grad is not None
        torch.testing.assert_close(
            actual.grad.cpu(), expected.grad, atol=2e-4, rtol=2e-4
        )
    if mojo_bias is not None:
        assert mojo_bias.grad is not None
        torch.testing.assert_close(
            mojo_bias.grad.cpu(), reference_bias.grad, atol=2e-4, rtol=2e-4
        )


def test_fast_log_softmax_training_backward(mojo_gpu):
    """Autograd must retain a valid Mojo payload for the saved forward output."""
    generator = torch.Generator().manual_seed(20260718)
    x = torch.randn(32, 65, generator=generator)
    grad_output = torch.randn(32, 65, generator=generator)

    reference = x.clone().requires_grad_()
    torch.nn.functional.log_softmax(reference, dim=-1).backward(grad_output)

    actual = x.to(mojo_gpu).requires_grad_()
    torch.nn.functional.log_softmax(actual, dim=-1).backward(grad_output.to(mojo_gpu))

    assert actual.grad is not None
    torch.testing.assert_close(actual.grad.cpu(), reference.grad, atol=2e-5, rtol=2e-5)


def test_fast_log_softmax_uses_saved_tensor_hooks(mojo_gpu):
    generator = torch.Generator().manual_seed(20260718)
    x = torch.randn(8, 17, generator=generator)
    grad_output = torch.randn(8, 17, generator=generator)
    reference = x.clone().requires_grad_()
    torch.nn.functional.log_softmax(reference, dim=-1).backward(grad_output)
    hook_calls = []

    def pack(tensor):
        hook_calls.append(("pack", tensor.device.type))
        return tensor.cpu()

    def unpack(tensor):
        hook_calls.append(("unpack", tensor.device.type))
        return tensor

    actual = x.to(mojo_gpu).requires_grad_()
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        torch.nn.functional.log_softmax(actual, dim=-1).backward(
            grad_output.to(mojo_gpu)
        )

    assert hook_calls == [("pack", "mojo"), ("unpack", "cpu")]
    torch.testing.assert_close(actual.grad.cpu(), reference.grad, atol=2e-5, rtol=2e-5)


def test_fast_log_softmax_backward_rejects_mutated_saved_output(mojo_gpu):
    output = torch.nn.functional.log_softmax(
        torch.randn(8, 17).to(mojo_gpu).requires_grad_(), dim=-1
    )
    with torch.no_grad():
        output.add_(torch.ones_like(output))

    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        output.backward(torch.randn(8, 17).to(mojo_gpu))


def test_fast_log_softmax_does_not_retain_python_output_cycle(mojo_gpu):
    input = torch.randn(8, 17).to(mojo_gpu).requires_grad_()
    output = torch.nn.functional.log_softmax(input, dim=-1)
    output_ref = weakref.ref(output)

    del output

    assert output_ref() is None
