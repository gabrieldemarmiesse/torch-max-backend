"""Tests for the Mojo-extension fast path used by max_device eager mode."""

import pytest
import torch

from torch_max_backend import register_max_devices
from torch_max_backend.flags import fast_eager_enabled

pytestmark = pytest.mark.xdist_group(name="group1")


@pytest.fixture(autouse=True)
def setup_max_device():
    register_max_devices()


BINARY_OPS = [torch.add, torch.sub, torch.mul, torch.div, torch.maximum, torch.minimum]
UNARY_OPS = [torch.relu, torch.exp]


@pytest.mark.parametrize("op", BINARY_OPS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_fast_binary_ops_match_cpu(max_device, op, dtype):
    x = torch.randn(33, 65).to(dtype)
    y = torch.randn(33, 65).to(dtype) + 1.5  # avoid div-by-~0
    result = op(x.to(max_device), y.to(max_device))
    torch.testing.assert_close(result.cpu(), op(x, y))


@pytest.mark.parametrize("op", UNARY_OPS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_fast_unary_ops_match_cpu(max_device, op, dtype):
    x = torch.randn(33, 65).to(dtype)
    result = op(x.to(max_device))
    torch.testing.assert_close(result.cpu(), op(x))


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_fast_binary_int_dtypes(max_device, dtype):
    x = torch.arange(100, dtype=dtype)
    y = torch.arange(100, dtype=dtype) * 3
    result = (x.to(max_device) + y.to(max_device)).cpu()
    torch.testing.assert_close(result, x + y)


def test_fast_path_is_used(max_device):
    """The eligible case must go through the Mojo kernel, not the fallback."""
    if not fast_eager_enabled():
        pytest.skip("fast eager path disabled")
    from torch_max_backend import eager_kernels

    calls = []
    original = eager_kernels.elementwise_ops.Add

    def spy(*args):
        calls.append(args)
        return original(*args)

    eager_kernels.elementwise_ops.Add = spy
    try:
        x = torch.randn(8, 8).to(max_device)
        y = torch.randn(8, 8).to(max_device)
        _ = x + y
    finally:
        eager_kernels.elementwise_ops.Add = original
    assert len(calls) == 1


def test_fallback_broadcast(max_device):
    x = torch.randn(16, 16)
    y = torch.randn(16)
    result = (x.to(max_device) + y.to(max_device)).cpu()
    torch.testing.assert_close(result, x + y)


def test_fallback_scalar_other(max_device):
    x = torch.randn(16, 16)
    result = (x.to(max_device) + 2.5).cpu()
    torch.testing.assert_close(result, x + 2.5)


def test_fallback_alpha(max_device):
    x = torch.randn(16, 16)
    y = torch.randn(16, 16)
    result = torch.add(x.to(max_device), y.to(max_device), alpha=2.0).cpu()
    torch.testing.assert_close(result, torch.add(x, y, alpha=2.0))


def test_fallback_int_div(max_device):
    x = torch.arange(1, 65, dtype=torch.int32)
    y = torch.full((64,), 4, dtype=torch.int32)
    result = (x.to(max_device) / y.to(max_device)).cpu()
    # check_dtype=False: the graph-based fallback path promotes int div to
    # float64 where torch gives float32 — a pre-existing deviation
    # (reproduces with TORCH_MAX_BACKEND_FAST_EAGER=0).
    torch.testing.assert_close(result, x / y, check_dtype=False)


@pytest.mark.parametrize("shape", [(0,), (1,), (7,), (0, 5)])
def test_edge_case_shapes(max_device, shape):
    x = torch.randn(*shape)
    y = torch.randn(*shape)
    result = (x.to(max_device) + y.to(max_device)).cpu()
    torch.testing.assert_close(result, x + y)


def test_chained_fast_ops(max_device):
    """Outputs of fast ops must be valid inputs to further fast ops."""
    x = torch.randn(32, 32)
    y = torch.randn(32, 32)
    device_result = x.to(max_device)
    for _ in range(5):
        device_result = torch.relu(device_result * y.to(max_device) + y.to(max_device))
    expected = x
    for _ in range(5):
        expected = torch.relu(expected * y + y)
    torch.testing.assert_close(device_result.cpu(), expected)


@pytest.fixture
def max_gpu(max_gpu_available: bool):
    """GPU max_device only — for ops whose fast path is GPU-gated."""
    if not max_gpu_available:
        pytest.skip("You do not have a GPU supported by Max")
    return "max_device:0"


def test_fast_view_family(max_device):
    x = torch.randn(2, 6, 768)
    xd = x.to(max_device)
    torch.testing.assert_close(xd.view(-1, 768).cpu(), x.view(-1, 768))
    torch.testing.assert_close(xd.reshape(12, 768).cpu(), x.reshape(12, 768))
    torch.testing.assert_close(xd.unsqueeze(0).cpu(), x.unsqueeze(0))


def test_fast_view_aliases_storage(max_device):
    """The fast view must alias, matching torch.Tensor.view semantics."""
    x = torch.zeros(4, 4).to(max_device)
    v = x.view(16)
    x += torch.ones(4, 4).to(max_device)
    torch.testing.assert_close(v.cpu(), torch.ones(16))


@pytest.mark.parametrize("dims", [(0, 1), (1, 2), (-1, -2)])
def test_fast_transpose(max_device, dims):
    x = torch.randn(2, 3, 4)
    result = x.to(max_device).transpose(*dims).contiguous().cpu()
    torch.testing.assert_close(result, x.transpose(*dims).contiguous())


def test_fast_t(max_device):
    x = torch.randn(50, 30)
    torch.testing.assert_close(
        x.to(max_device).t().contiguous().cpu(), x.t().contiguous()
    )


@pytest.mark.parametrize("split_size,dim", [(768, 2), (2, 0), ([1, 2, 3], 1)])
def test_fast_split(max_device, split_size, dim):
    x = torch.randn(4, 6, 2304)
    dev_parts = x.to(max_device).split(split_size, dim=dim)
    for dev_part, ref_part in zip(dev_parts, x.split(split_size, dim=dim)):
        torch.testing.assert_close(dev_part.cpu(), ref_part)


def test_fast_cat_skips_legacy_empty(max_device):
    empty = torch.empty(0)
    x = torch.randn(1, 12, 6, 64)
    result = torch.cat([empty.to(max_device), x.to(max_device)], dim=-2)
    torch.testing.assert_close(result.cpu(), torch.cat([empty, x], dim=-2))


def test_fast_batch_norm_inference(max_device):
    x = torch.randn(2, 64, 14, 14)
    bn = torch.nn.BatchNorm2d(64).eval()
    bn.running_mean.normal_()
    bn.running_var.uniform_(0.5, 2.0)
    bn_dev = torch.nn.BatchNorm2d(64).eval()
    bn_dev.load_state_dict(bn.state_dict())
    bn_dev = bn_dev.to(max_device)
    with torch.no_grad():
        torch.testing.assert_close(
            bn_dev(x.to(max_device)).cpu(), bn(x), atol=1e-5, rtol=1e-5
        )


def test_fast_layer_norm(max_device):
    x = torch.randn(2, 6, 768)
    ln = torch.nn.LayerNorm(768).eval()
    with torch.no_grad():
        ln.weight.normal_()
        ln.bias.normal_()
    ln_dev = torch.nn.LayerNorm(768).eval()
    ln_dev.load_state_dict(ln.state_dict())
    ln_dev = ln_dev.to(max_device)
    with torch.no_grad():
        torch.testing.assert_close(
            ln_dev(x.to(max_device)).cpu(), ln(x), atol=1e-5, rtol=1e-5
        )


def test_fast_native_layer_norm_stats(max_device):
    x = torch.randn(1, 6, 768).to(max_device)
    w = torch.ones(768).to(max_device)
    b = torch.zeros(768).to(max_device)
    out, mean, rstd = torch.native_layer_norm(x, [768], w, b, 1e-5)
    ref_out, ref_mean, ref_rstd = torch.native_layer_norm(
        x.cpu(), [768], w.cpu(), b.cpu(), 1e-5
    )
    torch.testing.assert_close(out.cpu(), ref_out, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(mean.cpu(), ref_mean, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(rstd.cpu(), ref_rstd, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("keepdim", [True, False])
def test_fast_mean_trailing_dims(max_device, keepdim):
    x = torch.randn(1, 512, 7, 7)
    result = x.to(max_device).mean([-1, -2], keepdim=keepdim).cpu()
    torch.testing.assert_close(result, x.mean([-1, -2], keepdim=keepdim))


def test_fast_reduction_library_tier(max_device):
    """Huge-col reductions route to the stdlib reduction library (GPU: rows <=
    128 and cols >= 2**20; MAX-CPU: always) — exercise that tier, which no other
    test reaches. Integer-valued floats keep every f32 partial sum exact, so the
    comparisons are bit-exact instead of tolerance-based."""
    # rows == 1: full-reduction layout (the two-phase GPU tier's main case).
    x = torch.randint(-4, 5, (1, 2**20 + 7)).float()
    xd = x.to(max_device)
    torch.testing.assert_close(xd.sum(-1).cpu(), x.sum(-1))
    torch.testing.assert_close(xd.amax(-1).cpu(), x.amax(-1))
    torch.testing.assert_close(torch.any(xd, -1).cpu(), torch.any(x, -1))

    # rows == 128 (gate boundary): per-row outputs must land in the right rows.
    y = torch.randint(-4, 5, (128, 2**20)).float()
    y[5] = 0.0  # give any() a False row
    yd = y.to(max_device)
    torch.testing.assert_close(yd.sum(-1).cpu(), y.sum(-1))
    torch.testing.assert_close(yd.amax(-1).cpu(), y.amax(-1))
    torch.testing.assert_close(torch.any(yd, -1).cpu(), torch.any(y, -1))


def test_fast_anyall_nan_is_truthy(max_device):
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
        got = torch.any(x.to(max_device), -1).cpu()
        torch.testing.assert_close(got, torch.any(x, -1))
    for x in (small_all, huge_all):
        got = torch.all(x.to(max_device), -1).cpu()
        torch.testing.assert_close(got, torch.all(x, -1))


def test_fast_max_pool2d(max_device):
    x = torch.randn(1, 64, 32, 32)
    result = torch.nn.functional.max_pool2d(x.to(max_device), 3, 2, 1).cpu()
    torch.testing.assert_close(result, torch.nn.functional.max_pool2d(x, 3, 2, 1))


def test_fast_max_pool2d_indices(max_device):
    x = torch.randn(1, 8, 16, 16)
    dev_vals, dev_idx = torch.nn.functional.max_pool2d(
        x.to(max_device), 2, 2, return_indices=True
    )
    ref_vals, ref_idx = torch.nn.functional.max_pool2d(x, 2, 2, return_indices=True)
    torch.testing.assert_close(dev_vals.cpu(), ref_vals)
    torch.testing.assert_close(dev_idx.cpu(), ref_idx)


def test_fast_embedding(max_device):
    weight = torch.randn(100, 32)
    idx = torch.randint(0, 100, (2, 5))
    result = torch.nn.functional.embedding(idx.to(max_device), weight.to(max_device))
    torch.testing.assert_close(result.cpu(), torch.nn.functional.embedding(idx, weight))


def test_fast_scalar_elementwise(max_device):
    x = torch.randn(2, 6, 3072)
    xd = x.to(max_device)
    torch.testing.assert_close((xd * 0.5).cpu(), x * 0.5)
    torch.testing.assert_close((xd + 1.0).cpu(), x + 1.0)
    torch.testing.assert_close((xd**3.0).cpu(), x**3.0, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(torch.tanh(xd).cpu(), torch.tanh(x))


def test_fast_add_scalar_int(max_device):
    x = torch.arange(6)
    torch.testing.assert_close((x.to(max_device) + 3).cpu(), x + 3)


def test_fast_add_inplace(max_device):
    x = torch.randn(4, 4)
    y = torch.randn(4, 4)
    xd = x.clone().to(max_device)
    xd += y.to(max_device)
    torch.testing.assert_close(xd.cpu(), x + y)


def test_fast_all_and_item(max_device):
    ones = torch.ones(1, 6, dtype=torch.bool).to(max_device)
    assert bool(ones.all().item()) is True
    mixed = torch.tensor([[True, False, True]]).to(max_device)
    assert bool(mixed.all().item()) is False


def test_fast_arange(max_device):
    torch.testing.assert_close(
        torch.arange(6, device=max_device).cpu(), torch.arange(6)
    )
    torch.testing.assert_close(
        torch.arange(2, 20, 3, device=max_device).cpu(), torch.arange(2, 20, 3)
    )


def test_fast_cast(max_device):
    x = torch.randint(0, 3, (1, 6))
    torch.testing.assert_close(x.to(max_device).to(torch.bool).cpu(), x.to(torch.bool))
    f = torch.randn(3, 4)
    torch.testing.assert_close(
        f.to(max_device).to(torch.float16).cpu(), f.to(torch.float16)
    )


# ---- GPU-only fast paths (matmul / conv / attention via MAX kernel library)


def test_fast_mm_addmm(max_gpu):
    a = torch.randn(6, 768)
    b = torch.randn(768, 2304)
    bias = torch.randn(2304)
    dev = torch.addmm(bias.to(max_gpu), a.to(max_gpu), b.to(max_gpu)).cpu()
    # TF32-level tolerance: the MAX matmul kernels (same as graph mode) use
    # tensor cores for float32.
    torch.testing.assert_close(dev, torch.addmm(bias, a, b), atol=5e-2, rtol=5e-2)
    dev = (a.to(max_gpu) @ b.to(max_gpu)).cpu()
    torch.testing.assert_close(dev, a @ b, atol=5e-2, rtol=5e-2)


def test_fast_bmm(max_gpu):
    a = torch.randn(12, 6, 64)
    b = torch.randn(12, 64, 6)
    dev = torch.bmm(a.to(max_gpu), b.to(max_gpu)).cpu()
    torch.testing.assert_close(dev, torch.bmm(a, b), atol=1e-2, rtol=1e-2)


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
def test_fast_conv2d(max_gpu, in_c, out_c, k, stride, padding, dilation, groups):
    x = torch.randn(1, in_c, 32, 32)
    w = torch.randn(out_c, in_c // groups, k, k)
    b = torch.randn(out_c)
    dev = torch.nn.functional.conv2d(
        x.to(max_gpu),
        w.to(max_gpu),
        b.to(max_gpu),
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    ).cpu()
    ref = torch.nn.functional.conv2d(
        x, w, b, stride=stride, padding=padding, dilation=dilation, groups=groups
    )
    torch.testing.assert_close(dev, ref, atol=5e-2, rtol=5e-2)


def test_fast_conv2d_batched_falls_back_correctly(max_gpu):
    x = torch.randn(3, 8, 16, 16)
    w = torch.randn(12, 8, 3, 3)
    dev = torch.nn.functional.conv2d(x.to(max_gpu), w.to(max_gpu), padding=1).cpu()
    ref = torch.nn.functional.conv2d(x, w, padding=1)
    torch.testing.assert_close(dev, ref, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("is_causal", [True, False])
# kv_len <= 32 exercises the library softmax's warp kernel, kv_len=64 the
# online/block kernel.
@pytest.mark.parametrize("kv_len", [6, 10, 64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_fast_sdpa(max_gpu, is_causal, kv_len, dtype):
    q = torch.randn(1, 12, 6, 64, dtype=dtype)
    k = torch.randn(1, 12, kv_len, 64, dtype=dtype)
    v = torch.randn(1, 12, kv_len, 64, dtype=dtype)
    dev = torch.nn.functional.scaled_dot_product_attention(
        q.to(max_gpu), k.to(max_gpu), v.to(max_gpu), is_causal=is_causal
    ).cpu()
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
    torch.testing.assert_close(dev, ref, atol=1e-2, rtol=1e-2)
