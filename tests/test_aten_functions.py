import torch
import pytest
from torch_max_backend.testing import check_functions_are_equivalent
from torch.ops import aten
from torch._dynamo.exc import BackendCompilerFailed
from torch._dynamo import mark_dynamic


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_scaled_dot_product_flash_attention_basic(cuda_device: str, dtype: torch.dtype):
    """Test _scaled_dot_product_flash_attention basic functionality"""

    def fn(q, k, v):
        return torch.ops.aten._scaled_dot_product_flash_attention(
            q, k, v, dropout_p=0.0, is_causal=False, return_debug_mask=False
        )[0]  # For the moment we support only training

    batch_size, num_heads, seq_len, head_dim = 2, 4, 8, 16
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype)

    check_functions_are_equivalent(fn, cuda_device, [q, k, v])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_scaled_dot_product_flash_attention_with_causal(cuda_device: str, dtype: str):
    """Test _scaled_dot_product_flash_attention with causal masking"""

    def fn(q, k, v):
        return torch.ops.aten._scaled_dot_product_flash_attention(
            q, k, v, dropout_p=0.0, is_causal=True, return_debug_mask=False
        )[0]  # For the moment we support only training

    batch_size, num_heads, seq_len, head_dim = 1, 2, 4, 8
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype)

    check_functions_are_equivalent(fn, cuda_device, [q, k, v])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_scaled_dot_product_flash_attention_with_scale(cuda_device: str, dtype):
    """Test _scaled_dot_product_flash_attention with custom scale"""

    def fn(q, k, v):
        return torch.ops.aten._scaled_dot_product_flash_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=False,
            return_debug_mask=False,
            scale=0.125,
        )[0]  # For the moment we support only training

    batch_size, num_heads, seq_len, head_dim = 1, 1, 4, 8
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype)

    check_functions_are_equivalent(fn, cuda_device, [q, k, v])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_native_batch_norm_legit_no_training_basic(device: str, dtype: torch.dtype):
    """Test basic batch normalization inference with different dtypes"""
    if device == "cuda":
        pytest.xfail("_native_batch_norm_legit_no_training not working on gpus yet")

    def fn(input_tensor, weight, bias, running_mean, running_var):
        return aten._native_batch_norm_legit_no_training.default(
            input_tensor, weight, bias, running_mean, running_var, 0.1, 1e-5
        )

    # Create test tensors
    batch_size, channels, height, width = 2, 3, 4, 4
    input_tensor = torch.randn(
        batch_size, channels, height, width, dtype=dtype, device=device
    )
    weight = torch.randn(channels, dtype=dtype, device=device)
    bias = torch.randn(channels, dtype=dtype, device=device)
    running_mean = torch.randn(channels, dtype=dtype, device=device)
    running_var = torch.abs(torch.randn(channels, dtype=dtype, device=device)) + 1e-5

    check_functions_are_equivalent(
        fn, device, [input_tensor, weight, bias, running_mean, running_var]
    )


@pytest.mark.parametrize("channels", [1, 4, 16])
def test_native_batch_norm_legit_no_training_different_channels(
    device: str, channels: int
):
    """Test batch norm with different numbers of channels"""
    if device == "cuda":
        pytest.xfail("_native_batch_norm_legit_no_training not working on gpus yet")

    def fn(input_tensor, weight, bias, running_mean, running_var):
        return aten._native_batch_norm_legit_no_training.default(
            input_tensor, weight, bias, running_mean, running_var, 0.1, 1e-5
        )

    # Create test tensors with varying channel dimensions
    batch_size, height, width = 2, 8, 8
    input_tensor = torch.randn(batch_size, channels, height, width, device=device)
    weight = torch.randn(channels, device=device)
    bias = torch.randn(channels, device=device)
    running_mean = torch.randn(channels, device=device)
    running_var = torch.abs(torch.randn(channels, device=device)) + 1e-5

    # Test that compilation works and outputs match
    check_functions_are_equivalent(
        fn, device, [input_tensor, weight, bias, running_mean, running_var]
    )


def test_native_batch_norm_legit_no_training_none_weight_bias(device: str):
    """Test batch norm with None weight and bias"""
    if device == "cuda":
        pytest.xfail("_native_batch_norm_legit_no_training not working on gpus yet")

    def fn(input_tensor, running_mean, running_var):
        return aten._native_batch_norm_legit_no_training.default(
            input_tensor, None, None, running_mean, running_var, 0.1, 1e-5
        )

    # Create test tensors
    batch_size, channels, height, width = 2, 3, 4, 4
    input_tensor = torch.randn(batch_size, channels, height, width, device=device)
    running_mean = torch.randn(channels, device=device)
    running_var = torch.abs(torch.randn(channels, device=device)) + 1e-5

    # Test that compilation works and outputs match
    check_functions_are_equivalent(
        fn, device, [input_tensor, running_mean, running_var]
    )


@pytest.mark.parametrize("eps", [1e-5, 1e-3])
def test_native_batch_norm_legit_no_training_different_eps(device: str, eps: float):
    """Test batch norm with different epsilon values"""
    if device == "cuda":
        pytest.xfail("_native_batch_norm_legit_no_training not working on gpus yet")

    def fn(input_tensor, weight, bias, running_mean, running_var):
        return aten._native_batch_norm_legit_no_training.default(
            input_tensor, weight, bias, running_mean, running_var, 0.1, eps
        )

    # Create test tensors
    batch_size, channels, height, width = 2, 3, 4, 4
    input_tensor = torch.randn(batch_size, channels, height, width, device=device)
    weight = torch.randn(channels, device=device)
    bias = torch.randn(channels, device=device)
    running_mean = torch.randn(channels, device=device)
    running_var = torch.abs(torch.randn(channels, device=device)) + eps * 10

    # Test that compilation works and outputs match
    check_functions_are_equivalent(
        fn, device, [input_tensor, weight, bias, running_mean, running_var]
    )


def test_native_batch_norm_legit_no_training_2d_input(device: str):
    """Test batch norm with 2D input (N, C)"""
    if device == "cuda":
        pytest.xfail("_native_batch_norm_legit_no_training not working on gpus yet")

    def fn(input_tensor, weight, bias, running_mean, running_var):
        return aten._native_batch_norm_legit_no_training.default(
            input_tensor, weight, bias, running_mean, running_var, 0.1, 1e-5
        )

    # Create 2D test tensors (batch_size, channels)
    batch_size, channels = 10, 5
    input_tensor = torch.randn(batch_size, channels, device=device)
    weight = torch.randn(channels, device=device)
    bias = torch.randn(channels, device=device)
    running_mean = torch.randn(channels, device=device)
    running_var = torch.abs(torch.randn(channels, device=device)) + 1e-5

    # Test that compilation works and outputs match
    check_functions_are_equivalent(
        fn, device, [input_tensor, weight, bias, running_mean, running_var]
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_aten_amax_all_dims(device: str, dtype: torch.dtype):
    """Test aten_amax with default empty dim list (reduces over all dimensions)"""
    # Skip float16 on CPU as MAX doesn't support f16 on CPU
    if device == "cpu" and dtype == torch.float16:
        pytest.skip("float16 not supported on CPU in MAX")

    def fn(x):
        return aten.amax(x)

    # Test with different shapes
    x = torch.randn(3, 4, 5, dtype=dtype, device=device)
    check_functions_are_equivalent(fn, device, [x])

    # Test with 1D tensor
    x1d = torch.randn(10, dtype=dtype, device=device)
    check_functions_are_equivalent(fn, device, [x1d])


@pytest.mark.parametrize(
    "dtype", [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]
)
def test_aten_bitwise_not(device: str, dtype: torch.dtype):
    def fn(x):
        return aten.bitwise_not(x)

    # Create test tensors
    x = torch.randint(0, 10, (3, 4), dtype=dtype, device=device)

    check_functions_are_equivalent(fn, device, [x])


def test_aten_bitwise_not_bool(device: str):
    dtype = torch.bool

    def fn(x):
        return aten.bitwise_not(x)

    # Create test tensors
    x = torch.randint(0, 2, (3, 4), dtype=dtype, device=device)

    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize(
    "dtype", [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]
)
def test_aten_bitwise_and_scalar(device: str, dtype: torch.dtype):
    def fn(x):
        return aten.bitwise_and(x, 6)

    # Create test tensors
    x = torch.randint(0, 10, (3, 4), dtype=dtype, device=device)

    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("bool_value", [True, False])
def test_aten_bitwise_and_scalar_bool(device: str, bool_value: bool):
    dtype = torch.bool

    def fn(x):
        return aten.bitwise_and(x, bool_value)

    # Create test tensors
    x = torch.randint(0, 2, (3, 4), dtype=dtype, device=device)

    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize(
    "dtype", [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]
)
def test_aten_bitwise_and(device: str, dtype: torch.dtype):
    def fn(x, y):
        return aten.bitwise_and(x, y)

    # Create test tensors
    x = torch.randint(0, 10, (3, 4), dtype=dtype, device=device)
    y = torch.randint(0, 10, (3, 4), dtype=dtype, device=device)

    check_functions_are_equivalent(fn, device, [x, y])


def test_aten_bitwise_and_bool(device: str):
    dtype = torch.bool

    def fn(x, y):
        return aten.bitwise_and(x, y)

    # Create test tensors
    x = torch.randint(0, 2, (3, 4), dtype=dtype, device=device)
    y = torch.randint(0, 2, (3, 4), dtype=dtype, device=device)

    check_functions_are_equivalent(fn, device, [x, y])


def test_aten_bitwise_and_broadcasting(device: str):
    def fn(x, y):
        return aten.bitwise_and(x, y)

    # Create test tensors with broadcasting shapes
    x = torch.randint(0, 10, (3, 4, 5), dtype=torch.int32)
    y = torch.randint(0, 10, (5,), dtype=torch.int32)

    check_functions_are_equivalent(fn, device, [x, y])


def test_aten_bitwise_and_broadcasting_ones(device: str):
    def fn(x, y):
        return aten.bitwise_and(x, y)

    # Create test tensors with broadcasting shapes
    x = torch.randint(0, 100, (3, 1, 5), dtype=torch.int32)
    y = torch.randint(0, 100, (1, 4, 5), dtype=torch.int32)

    check_functions_are_equivalent(fn, device, [x, y])


def test_aten_bitwise_and_broadcasting_ones_pad(device: str):
    def fn(x, y):
        return aten.bitwise_and(x, y)

    # Create test tensors with broadcasting shapes
    x = torch.randint(0, 100, (8, 3, 1, 5), dtype=torch.int32)
    y = torch.randint(0, 100, (1, 4, 5), dtype=torch.int32)

    check_functions_are_equivalent(fn, device, [x, y])


def test_aten_bitwise_and_broadcasting_ones_pad_dynamic_dim(device: str):
    def fn(x, y):
        return aten.bitwise_and(x, y)

    # Create test tensors with broadcasting shapes
    x = torch.randint(0, 100, (8, 3, 1, 5), dtype=torch.int32, device=device)
    mark_dynamic(x, 0)
    mark_dynamic(x, 1)
    y = torch.randint(0, 100, (1, 4, 5), dtype=torch.int32, device=device)
    mark_dynamic(y, 1)

    check_functions_are_equivalent(fn, None, [x, y])


# Tests for bitwise_or operations
@pytest.mark.parametrize(
    "dtype", [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]
)
def test_aten_bitwise_or_scalar(device: str, dtype: torch.dtype):
    def fn(x):
        return aten.bitwise_or(x, 6)

    # Create test tensors
    x = torch.randint(0, 10, (3, 4), dtype=dtype, device=device)

    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("bool_value", [True, False])
def test_aten_bitwise_or_scalar_bool(device: str, bool_value: bool):
    dtype = torch.bool

    def fn(x):
        return aten.bitwise_or(x, bool_value)

    # Create test tensors
    x = torch.randint(0, 2, (3, 4), dtype=dtype, device=device)

    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize(
    "dtype", [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]
)
def test_aten_bitwise_or(device: str, dtype: torch.dtype):
    def fn(x, y):
        return aten.bitwise_or(x, y)

    # Create test tensors
    x = torch.randint(0, 10, (3, 4), dtype=dtype, device=device)
    y = torch.randint(0, 10, (3, 4), dtype=dtype, device=device)

    check_functions_are_equivalent(fn, device, [x, y])


def test_aten_bitwise_or_bool(device: str):
    dtype = torch.bool

    def fn(x, y):
        return aten.bitwise_or(x, y)

    # Create test tensors
    x = torch.randint(0, 2, (3, 4), dtype=dtype, device=device)
    y = torch.randint(0, 2, (3, 4), dtype=dtype, device=device)

    check_functions_are_equivalent(fn, device, [x, y])


def test_aten_bitwise_or_broadcasting(device: str):
    def fn(x, y):
        return aten.bitwise_or(x, y)

    # Create test tensors with broadcasting shapes
    x = torch.randint(0, 10, (3, 4, 5), dtype=torch.int32)
    y = torch.randint(0, 10, (4, 5), dtype=torch.int32)

    check_functions_are_equivalent(fn, device, [x, y])


# Tests for bitwise_xor operations
@pytest.mark.parametrize(
    "dtype", [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]
)
def test_aten_bitwise_xor_scalar(device: str, dtype: torch.dtype):
    def fn(x):
        return aten.bitwise_xor(x, 6)

    # Create test tensors
    x = torch.randint(0, 10, (3, 4), dtype=dtype, device=device)

    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("bool_value", [True, False])
def test_aten_bitwise_xor_scalar_bool(device: str, bool_value: bool):
    dtype = torch.bool

    def fn(x):
        return aten.bitwise_xor(x, bool_value)

    # Create test tensors
    x = torch.randint(0, 2, (3, 4), dtype=dtype, device=device)

    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize(
    "dtype", [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]
)
def test_aten_bitwise_xor(device: str, dtype: torch.dtype):
    def fn(x, y):
        return aten.bitwise_xor(x, y)

    # Create test tensors
    x = torch.randint(0, 10, (3, 4), dtype=dtype, device=device)
    y = torch.randint(0, 10, (3, 4), dtype=dtype, device=device)

    check_functions_are_equivalent(fn, device, [x, y])


def test_aten_bitwise_xor_bool(device: str):
    dtype = torch.bool

    def fn(x, y):
        return aten.bitwise_xor(x, y)

    # Create test tensors
    x = torch.randint(0, 2, (3, 4), dtype=dtype, device=device)
    y = torch.randint(0, 2, (3, 4), dtype=dtype, device=device)

    check_functions_are_equivalent(fn, device, [x, y])


def test_aten_bitwise_xor_broadcasting(device: str):
    def fn(x, y):
        return aten.bitwise_xor(x, y)

    # Create test tensors with broadcasting shapes
    x = torch.randint(0, 10, (3, 4, 5), dtype=torch.int32)
    y = torch.randint(0, 10, (4, 5), dtype=torch.int32)

    check_functions_are_equivalent(fn, device, [x, y])


@pytest.mark.parametrize("repeats", [1, 2, 3, 5])
@pytest.mark.parametrize("dim", [0, 1, -1])
def test_aten_repeat_interleave_basic(device: str, repeats: int, dim: int):
    """Test aten.repeat_interleave with basic parameters"""

    def fn(x):
        return aten.repeat_interleave(x, repeats, dim)

    x = torch.randn(3, 4, device=device)
    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("dtype", [torch.float32, torch.int32, torch.bool])
def test_aten_repeat_interleave_different_dtypes(device: str, dtype: torch.dtype):
    """Test aten.repeat_interleave with different data types"""

    def fn(x):
        return aten.repeat_interleave(x, 2, 0)

    if dtype == torch.bool:
        x = torch.randint(0, 2, (3, 4), dtype=dtype, device=device)
    elif dtype == torch.int32:
        x = torch.randint(0, 10, (3, 4), dtype=dtype, device=device)
    else:
        x = torch.randn(3, 4, dtype=dtype, device=device)

    check_functions_are_equivalent(fn, device, [x])


def test_aten_repeat_interleave_1d(device: str):
    """Test aten.repeat_interleave with 1D tensor"""

    def fn(x):
        return aten.repeat_interleave(x, 3, 0)

    x = torch.randn(5, device=device)
    check_functions_are_equivalent(fn, device, [x])


def test_aten_repeat_interleave_3d(device: str):
    """Test aten.repeat_interleave with 3D tensor"""

    def fn(x):
        return aten.repeat_interleave(x, 2, 1)

    x = torch.randn(2, 3, 4, device=device)
    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("shape", [(1, 5), (5, 1), (1, 1)])
def test_aten_repeat_interleave_edge_cases(device: str, shape: tuple):
    """Test aten.repeat_interleave with edge case shapes"""

    def fn(x):
        return aten.repeat_interleave(x, 2, 0)

    x = torch.randn(*shape, device=device)
    check_functions_are_equivalent(fn, device, [x])


def test_aten_repeat_interleave_large_repeats(device: str):
    """Test aten.repeat_interleave with large repeat count"""

    def fn(x):
        return aten.repeat_interleave(x, 10, 0)

    x = torch.randn(2, 3, device=device)
    check_functions_are_equivalent(fn, device, [x])


def test_aten_split_with_sizes_basic(device: str):
    """Test aten.split_with_sizes with basic splits"""

    def fn(x):
        return aten.split_with_sizes(x, [2, 3, 5], dim=0)

    x = torch.randn(10, 4, device=device)
    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("dim", [0, 1, 2])
def test_aten_split_with_sizes_different_dims(device: str, dim: int):
    """Test aten.split_with_sizes on different dimensions"""

    def fn(x):
        # Adjust split sizes based on dimension
        if dim == 0:
            return aten.split_with_sizes(x, [1, 2, 1], dim=dim)  # sum=4, dim size=4
        elif dim == 1:
            return aten.split_with_sizes(x, [1, 2, 1], dim=dim)  # sum=4, dim size=4
        else:  # dim == 2
            return aten.split_with_sizes(x, [2, 2, 1], dim=dim)  # sum=5, dim size=5

    x = torch.randn(4, 4, 5, device=device)
    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("dim", [-1, -2, -3])
def test_aten_split_with_sizes_negative_dims(device: str, dim: int):
    """Test aten.split_with_sizes with negative dimension indices"""

    def fn(x):
        # Adjust split sizes based on dimension
        if dim == -1:  # last dim, size=5
            return aten.split_with_sizes(x, [2, 2, 1], dim=dim)
        elif dim == -2:  # middle dim, size=4
            return aten.split_with_sizes(x, [1, 2, 1], dim=dim)
        else:  # dim == -3, first dim, size=4
            return aten.split_with_sizes(x, [1, 2, 1], dim=dim)

    x = torch.randn(4, 4, 5, device=device)
    check_functions_are_equivalent(fn, device, [x])


def test_aten_split_with_sizes_uneven(device: str):
    """Test aten.split_with_sizes with uneven splits"""

    def fn(x):
        return aten.split_with_sizes(x, [1, 3, 2, 4], dim=1)

    x = torch.randn(3, 10, device=device)
    check_functions_are_equivalent(fn, device, [x])


def test_aten_split_with_sizes_single_split(device: str):
    """Test aten.split_with_sizes with single split (entire tensor)"""

    def fn(x):
        return aten.split_with_sizes(x, [5], dim=0)

    x = torch.randn(5, 3, device=device)
    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("dtype", [torch.float32, torch.int32, torch.bool])
def test_aten_split_with_sizes_dtypes(device: str, dtype: torch.dtype):
    """Test aten.split_with_sizes with different data types"""

    def fn(x):
        return aten.split_with_sizes(x, [2, 2, 2], dim=0)

    if dtype == torch.bool:
        x = torch.randint(0, 2, (6, 4), dtype=dtype, device=device)
    elif dtype == torch.int32:
        x = torch.randint(0, 10, (6, 4), dtype=dtype, device=device)
    else:
        x = torch.randn(6, 4, dtype=dtype, device=device)

    check_functions_are_equivalent(fn, device, [x])


def test_aten_split_with_sizes_1d(device: str):
    """Test aten.split_with_sizes with 1D tensor"""

    def fn(x):
        return aten.split_with_sizes(x, [3, 3, 4], dim=0)

    x = torch.randn(10, device=device)
    check_functions_are_equivalent(fn, device, [x])


def test_aten_split_with_sizes_3d(device: str):
    """Test aten.split_with_sizes with 3D tensor"""

    def fn(x):
        return aten.split_with_sizes(x, [1, 1, 2], dim=2)

    x = torch.randn(2, 3, 4, device=device)
    check_functions_are_equivalent(fn, device, [x])


def test_aten_split_with_sizes_many_splits(device: str):
    """Test aten.split_with_sizes with many small splits"""

    def fn(x):
        return aten.split_with_sizes(x, [1, 1, 1, 1, 1, 1, 1, 1], dim=0)

    x = torch.randn(8, 3, device=device)
    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("split_sizes", [[2, 0, 3], [0, 5, 0], [0, 0, 5]])
def test_aten_split_with_sizes_zero_size(device: str, split_sizes: list[int]):
    """Test aten.split_with_sizes with zero-sized splits"""

    def fn(x):
        return aten.split_with_sizes(x, split_sizes, dim=0)

    x = torch.randn(5, 3, device=device)
    check_functions_are_equivalent(fn, device, [x])


def test_aten_split_with_sizes_exact_split(device: str):
    """Test aten.split_with_sizes where sizes exactly match dimension"""

    def fn(x):
        return aten.split_with_sizes(x, [2, 2, 2, 2, 2], dim=1)

    x = torch.randn(3, 10, 4, device=device)
    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_aten_mul_scalar(device: str, dtype: torch.dtype):
    """Test aten.mul.Tensor with scalar multiplication"""

    def fn(x):
        return aten.mul.Tensor(x, 25.298221281347036)

    # Create test tensor similar to the user's example
    x = torch.randn(1, 19, 640, dtype=dtype, device=device)

    check_functions_are_equivalent(fn, device, [x])


def test_aten_squeeze_single_dim(device: str):
    """Test aten.squeeze with single dimension"""

    def fn(x):
        return aten.squeeze(x, 1)

    x = torch.randn(3, 1, 5, device=device)
    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("dim", [0, 1, 2, 3])
def test_aten_squeeze_different_dims(device: str, dim: int):
    """Test aten.squeeze on different dimensions"""

    def fn(x):
        return aten.squeeze(x, dim)

    x = torch.randn(1, 3, 1, 5, device=device)
    check_functions_are_equivalent(fn, device, [x])


def test_aten_squeeze_negative_dim(device: str):
    """Test aten.squeeze with negative dimension"""

    def fn(x):
        return aten.squeeze(x, -2)

    x = torch.randn(3, 1, 5, device=device)
    check_functions_are_equivalent(fn, device, [x])


def test_aten_squeeze_multiple_dims(device: str):
    """Test aten.squeeze with multiple dimensions"""

    def fn(x):
        return aten.squeeze(x, [0, 2])

    x = torch.randn(1, 3, 1, 5, device=device)
    check_functions_are_equivalent(fn, device, [x])


def test_aten_squeeze_no_change(device: str):
    """Test aten.squeeze when dimension is not size 1"""

    def fn(x):
        return aten.squeeze(x, 1)

    x = torch.randn(3, 4, 5, device=device)
    check_functions_are_equivalent(fn, device, [x])


def test_softmax_bfloat16_basic(device: str):
    """Test softmax with bfloat16 dtype for basic inputs"""

    def fn(x):
        return aten.softmax(x, dim=-1)

    x = torch.randn(4, 8, dtype=torch.bfloat16, device=device)
    check_functions_are_equivalent(fn, device, [x])


def test_softmax_bfloat16_large_values(device: str):
    """Test softmax with bfloat16 and large input values that might cause overflow"""

    def fn(x):
        return aten.softmax(x, dim=-1)

    # Create tensor with large values that might cause overflow in exp
    x = torch.tensor(
        [[100.0, 200.0, 300.0], [-100.0, -200.0, -300.0], [1000.0, 2000.0, 3000.0]],
        dtype=torch.bfloat16,
        device=device,
    )
    check_functions_are_equivalent(fn, device, [x])


def test_softmax_bfloat16_small_values(device: str):
    """Test softmax with bfloat16 and very small values"""

    def fn(x):
        return aten.softmax(x, dim=-1)

    # Create tensor with very small values
    x = torch.tensor(
        [[1e-7, 2e-7, 3e-7], [1e-8, 2e-8, 3e-8]], dtype=torch.bfloat16, device=device
    )
    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("dim", [0, 1, -1, -2])
def test_softmax_bfloat16_different_dims(device: str, dim: int):
    """Test softmax with bfloat16 on different dimensions"""

    def fn(x):
        return aten.softmax(x, dim=dim)

    x = torch.randn(3, 4, 5, dtype=torch.bfloat16, device=device)
    check_functions_are_equivalent(fn, device, [x])


def test_softmax_bfloat16_uniform_values(device: str):
    """Test softmax with bfloat16 when all values are the same"""

    def fn(x):
        return aten.softmax(x, dim=-1)

    # All values are the same - should result in uniform distribution
    x = torch.ones(3, 6, dtype=torch.bfloat16, device=device) * 5.0
    check_functions_are_equivalent(fn, device, [x])


def test_softmax_bfloat16_mixed_signs(device: str):
    """Test softmax with bfloat16 with mixed positive and negative values"""

    def fn(x):
        return aten.softmax(x, dim=-1)

    x = torch.tensor(
        [
            [-10.0, 0.0, 10.0, -5.0, 5.0],
            [100.0, -100.0, 50.0, -50.0, 0.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
        ],
        dtype=torch.bfloat16,
        device=device,
    )
    check_functions_are_equivalent(fn, device, [x])


def test_softmax_bfloat16_edge_cases(device: str):
    """Test softmax with bfloat16 edge cases including inf and very large differences"""

    def fn(x):
        return aten.softmax(x, dim=-1)

    # Test with values that have large differences
    x = torch.tensor(
        [[-1000.0, 1000.0], [0.0, 1000.0], [-500.0, 500.0]],
        dtype=torch.bfloat16,
        device=device,
    )
    check_functions_are_equivalent(fn, device, [x])


def test_softmax_bfloat16_single_element(device: str):
    """Test softmax with bfloat16 on single element tensors"""

    def fn(x):
        return aten.softmax(x, dim=-1)

    x = torch.tensor([[5.0]], dtype=torch.bfloat16, device=device)
    check_functions_are_equivalent(fn, device, [x])


def test_softmax_bfloat16_4d_tensor(device: str):
    """Test softmax with bfloat16 on 4D tensors (common in neural networks)"""

    def fn(x):
        return aten.softmax(x, dim=-1)

    x = torch.randn(2, 3, 4, 5, dtype=torch.bfloat16, device=device)
    check_functions_are_equivalent(fn, device, [x])


def test_softmax_bfloat16_extreme_range(device: str):
    """Test softmax with bfloat16 with extreme value ranges"""

    def fn(x):
        return aten.softmax(x, dim=-1)

    # Test with maximum range that bfloat16 can represent
    x = torch.tensor(
        [[3.38e38, -3.38e38, 0.0], [1e10, 1e-10, 1.0]],
        dtype=torch.bfloat16,
        device=device,
    )
    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.bfloat16, torch.float16]
)
def test_softmax_different_dtypes(device: str, dtype: torch.dtype):
    """Test softmax with different floating point dtypes"""

    def fn(x):
        return aten.softmax(x, dim=-1)

    x = torch.randn(3, 4, dtype=dtype, device=device) * 10
    check_functions_are_equivalent(fn, device, [x])


def test__softmax_float16_half_to_float(device: str):
    """Test _softmax with float16 and half_to_float=True"""

    def fn(x):
        return aten._softmax(x, dim=-1, half_to_float=True)

    x = torch.randn(4, 6, dtype=torch.float16, device=device)
    check_functions_are_equivalent(fn, device, [x])


def test__softmax_bfloat16_no_half_to_float(device: str):
    """Test _softmax with bfloat16 and half_to_float=False"""

    def fn(x):
        return aten._softmax(x, dim=-1, half_to_float=False)

    x = torch.randn(4, 6, dtype=torch.bfloat16, device=device)
    check_functions_are_equivalent(fn, device, [x])


def test_softmax_with_masked_fill_nan_bug(device: str):
    """Test to reproduce NaN bug with masked_fill followed by softmax"""

    def fn(x, mask):
        # Apply masked_fill with -inf to mask out certain positions
        masked = aten.masked_fill(x, mask, float("-inf"))
        # Apply softmax which should handle -inf correctly
        return aten._softmax(masked, dim=-1, half_to_float=False)

    # Create input tensor with shape similar to the error (1, 4, 19, 19)
    x = torch.randn(1, 4, 19, 19, dtype=torch.bfloat16, device=device)

    # Create a mask that will set some values to -inf
    mask = torch.rand(1, 4, 19, 19, device=device) > 0.7  # ~30% of values masked

    check_functions_are_equivalent(fn, device, [x, mask])


def test_softmax_with_all_masked_nan_bug(device: str):
    """Test softmax when entire row is masked (all -inf)"""

    def fn(x, mask):
        masked = aten.masked_fill(x, mask, float("-inf"))
        return aten._softmax(masked, dim=-1, half_to_float=False)

    x = torch.randn(1, 4, 5, 5, dtype=torch.bfloat16, device=device)

    # Create a mask that masks entire rows
    mask = torch.zeros(1, 4, 5, 5, dtype=torch.bool, device=device)
    mask[0, 0, 0, :] = True  # Mask entire first row of first batch/channel
    mask[0, 1, 2, :] = True  # Mask entire third row of second channel

    check_functions_are_equivalent(fn, device, [x, mask])


def test_softmax_direct_inf_values(device: str):
    """Test softmax with direct -inf values"""

    def fn(x):
        return aten._softmax(x, dim=-1, half_to_float=False)

    # Create tensor with some -inf values directly
    x = torch.randn(1, 2, 4, 4, dtype=torch.bfloat16, device=device)
    x[0, 0, 0, 1] = float("-inf")
    x[0, 0, 0, 3] = float("-inf")
    x[0, 1, 1, :] = float("-inf")  # Entire row is -inf

    check_functions_are_equivalent(fn, device, [x])


def test_softmax_mixed_inf_finite(device: str):
    """Test softmax with mix of -inf and finite values"""

    def fn(x):
        return aten.softmax(x, dim=-1)

    # Create specific test case with -inf and finite values
    x = torch.tensor(
        [
            [1.0, 2.0, float("-inf"), 3.0],
            [float("-inf"), float("-inf"), 1.0, 2.0],
            [float("-inf"), float("-inf"), float("-inf"), float("-inf")],  # All -inf
            [1.0, 1.0, 1.0, 1.0],
        ],
        dtype=torch.bfloat16,
        device=device,
    )
    x = x.unsqueeze(0).unsqueeze(0)  # Make it 4D: (1, 1, 4, 4)

    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("dtype", [torch.float32, torch.int32, torch.bool])
def test_aten_squeeze_different_dtypes(device: str, dtype: torch.dtype):
    """Test aten.squeeze with different data types"""

    def fn(x):
        return aten.squeeze(x, 1)

    if dtype == torch.bool:
        x = torch.randint(0, 2, (3, 1, 5), dtype=dtype, device=device)
    elif dtype == torch.int32:
        x = torch.randint(0, 10, (3, 1, 5), dtype=dtype, device=device)
    else:
        x = torch.randn(3, 1, 5, dtype=dtype, device=device)

    check_functions_are_equivalent(fn, device, [x])


def test_aten_squeeze_all_ones(device: str):
    """Test aten.squeeze with tensor of all size-1 dimensions"""

    def fn(x):
        return aten.squeeze(x, [0, 1, 2, 3])

    x = torch.randn(1, 1, 1, 1, device=device)
    check_functions_are_equivalent(fn, device, [x])


def test_aten_squeeze_2d(device: str):
    """Test aten.squeeze with 2D tensor"""

    def fn(x):
        return aten.squeeze(x, 0)

    x = torch.randn(1, 5, device=device)
    check_functions_are_equivalent(fn, device, [x])


def test_aten_squeeze_5d(device: str):
    """Test aten.squeeze with 5D tensor"""

    def fn(x):
        return aten.squeeze(x, [1, 3])

    x = torch.randn(2, 1, 3, 1, 4, device=device)
    check_functions_are_equivalent(fn, device, [x])


def test_aten_squeeze_empty_dims(device: str):
    """Test aten.squeeze with empty dimensions list"""

    def fn(x):
        return aten.squeeze(x, [])

    x = torch.randn(1, 3, 1, 5, device=device)
    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("shape", [(1,), (1, 1), (1, 1, 1)])
def test_aten_squeeze_edge_cases(device: str, shape: tuple):
    """Test aten.squeeze with edge case shapes"""

    def fn(x):
        return aten.squeeze(x, list(range(len(shape))))

    x = torch.randn(*shape, device=device)
    check_functions_are_equivalent(fn, device, [x])


def test_aten_logical_and_bool_tensors(device: str):
    """Test aten.logical_and with boolean tensors"""

    def fn(x, y):
        return aten.logical_and(x, y)

    x = torch.tensor([True, False, True, False], device=device)
    y = torch.tensor([True, True, False, False], device=device)
    check_functions_are_equivalent(fn, device, [x, y])


def test_aten_logical_and_numeric_tensors(device: str):
    """Test aten.logical_and with numeric tensors (converted to bool)"""

    def fn(x, y):
        return aten.logical_and(x, y)

    x = torch.tensor([1, 0, 2, -1], dtype=torch.float32, device=device)
    y = torch.tensor([3, 0, 0, 4], dtype=torch.float32, device=device)
    check_functions_are_equivalent(fn, device, [x, y])


@pytest.mark.parametrize(
    "dtype", [torch.int32, torch.int64, torch.float32, torch.float64]
)
def test_aten_logical_and_different_dtypes(device: str, dtype: torch.dtype):
    """Test aten.logical_and with different numeric data types"""

    def fn(x, y):
        return aten.logical_and(x, y)

    x = torch.tensor([1, 0, 2, -1], dtype=dtype, device=device)
    y = torch.tensor([3, 0, 0, 4], dtype=dtype, device=device)
    check_functions_are_equivalent(fn, device, [x, y])


def test_aten_logical_and_2d_tensors(device: str):
    """Test aten.logical_and with 2D tensors"""

    def fn(x, y):
        return aten.logical_and(x, y)

    x = torch.randint(0, 2, (3, 4), dtype=torch.bool, device=device)
    y = torch.randint(0, 2, (3, 4), dtype=torch.bool, device=device)
    check_functions_are_equivalent(fn, device, [x, y])


def test_aten_logical_and_broadcasting(device: str):
    """Test aten.logical_and with broadcasting"""

    def fn(x, y):
        return aten.logical_and(x, y)

    x = torch.randint(0, 3, (3, 4), dtype=torch.int32, device=device)
    y = torch.randint(0, 3, (4,), dtype=torch.int32, device=device)
    check_functions_are_equivalent(fn, device, [x, y])


def test_aten_logical_and_mixed_types(device: str):
    """Test aten.logical_and with mixed boolean and numeric types"""

    def fn(x, y):
        return aten.logical_and(x, y)

    x = torch.tensor([True, False, True, False], device=device)
    y = torch.tensor([1, 0, 2, 0], dtype=torch.float32, device=device)
    check_functions_are_equivalent(fn, device, [x, y])


def test_aten_logical_and_zeros_and_ones(device: str):
    """Test aten.logical_and with patterns of zeros and ones"""

    def fn(x, y):
        return aten.logical_and(x, y)

    x = torch.tensor([0, 1, 0, 1], dtype=torch.float32, device=device)
    y = torch.tensor([0, 0, 1, 1], dtype=torch.float32, device=device)
    check_functions_are_equivalent(fn, device, [x, y])


def test_aten_logical_and_all_false(device: str):
    """Test aten.logical_and with all false values"""

    def fn(x, y):
        return aten.logical_and(x, y)

    x = torch.zeros(4, dtype=torch.float32, device=device)
    y = torch.zeros(4, dtype=torch.float32, device=device)
    check_functions_are_equivalent(fn, device, [x, y])


def test_aten_logical_and_all_true(device: str):
    """Test aten.logical_and with all true values"""

    def fn(x, y):
        return aten.logical_and(x, y)

    x = torch.ones(4, dtype=torch.float32, device=device)
    y = torch.ones(4, dtype=torch.float32, device=device)
    check_functions_are_equivalent(fn, device, [x, y])


def test_aten_logical_and_negative_values(device: str):
    """Test aten.logical_and with negative values (should be treated as true)"""

    def fn(x, y):
        return aten.logical_and(x, y)

    x = torch.tensor([-1, -2, 0, 3], dtype=torch.float32, device=device)
    y = torch.tensor([4, 0, -5, 6], dtype=torch.float32, device=device)
    check_functions_are_equivalent(fn, device, [x, y])


def test_aten_logical_and_3d_tensors(device: str):
    """Test aten.logical_and with 3D tensors"""

    def fn(x, y):
        return aten.logical_and(x, y)

    x = torch.randint(0, 2, (2, 3, 4), dtype=torch.bool, device=device)
    y = torch.randint(0, 2, (2, 3, 4), dtype=torch.bool, device=device)
    check_functions_are_equivalent(fn, device, [x, y])


def test_aten_logical_and_scalar_like(device: str):
    """Test aten.logical_and with scalar-like tensors"""

    def fn(x, y):
        return aten.logical_and(x, y)

    x = torch.tensor(True, device=device)
    y = torch.tensor(False, device=device)
    check_functions_are_equivalent(fn, device, [x, y])


@pytest.mark.parametrize(
    "shape_pair", [((3, 1), (3, 4)), ((1, 4), (3, 4)), ((3, 1, 1), (3, 4, 5))]
)
def test_aten_logical_and_broadcasting_shapes(device: str, shape_pair: tuple):
    """Test aten.logical_and with various broadcasting shapes"""

    def fn(x, y):
        return aten.logical_and(x, y)

    x_shape, y_shape = shape_pair
    x = torch.randint(0, 2, x_shape, dtype=torch.int32, device=device)
    y = torch.randint(0, 2, y_shape, dtype=torch.int32, device=device)
    check_functions_are_equivalent(fn, device, [x, y])


@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("keepdim", [True, False])
def test_aten_amax_single_dim(device: str, dim: int, keepdim: bool):
    """Test aten_amax with single dimension"""

    def fn(x):
        return aten.amax(x, dim=[dim], keepdim=keepdim)

    x = torch.randn(3, 4, 5, device=device)
    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("dims", [[0, 1], [1, 2], [0, 2]])
@pytest.mark.parametrize("keepdim", [True, False])
def test_aten_amax_multiple_dims(device: str, dims: list[int], keepdim: bool):
    """Test aten_amax with multiple dimensions"""

    def fn(x):
        return aten.amax(x, dim=dims, keepdim=keepdim)

    x = torch.randn(3, 4, 5, device=device)
    check_functions_are_equivalent(fn, device, [x])


def test_aten_max_no_dim(device: str):
    """Test aten_max without dimension (returns single value)"""

    def fn(x):
        return aten.max(x)

    x = torch.randn(3, 4, 5, device=device)
    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("keepdim", [True, False])
def test_aten_max_with_dim(device: str, dim: int, keepdim: bool):
    """Test aten_max with dimension (returns values and indices tuple)"""

    def fn(x):
        return aten.max(x, dim=dim, keepdim=keepdim)

    x = torch.randn(3, 4, 5, device=device)
    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.float32])
def test_aten_max_different_dtypes(device: str, dtype: torch.dtype):
    """Test aten_max with different data types"""

    def fn(x):
        return aten.max(x, dim=1, keepdim=False)

    if dtype.is_floating_point:
        x = torch.randn(3, 4, dtype=dtype, device=device)
    else:
        x = torch.randint(-10, 10, (3, 4), dtype=dtype, device=device)

    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_aten_amin_all_dims(device: str, dtype: torch.dtype):
    """Test aten_amin with default empty dim list (reduces over all dimensions)"""
    # Skip float16 on CPU as MAX doesn't support f16 on CPU
    if device == "cpu" and dtype == torch.float16:
        pytest.skip("float16 not supported on CPU in MAX")

    def fn(x):
        return aten.amin(x)

    # Test with different shapes
    x = torch.randn(3, 4, 5, dtype=dtype, device=device)
    check_functions_are_equivalent(fn, device, [x])

    # Test with 1D tensor
    x1d = torch.randn(10, dtype=dtype, device=device)
    check_functions_are_equivalent(fn, device, [x1d])


@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("keepdim", [True, False])
def test_aten_amin_single_dim(device: str, dim: int, keepdim: bool):
    """Test aten_amin with single dimension"""

    def fn(x):
        return aten.amin(x, dim=[dim], keepdim=keepdim)

    x = torch.randn(3, 4, 5, device=device)
    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("dims", [[0, 1], [1, 2], [0, 2]])
@pytest.mark.parametrize("keepdim", [True, False])
def test_aten_amin_multiple_dims(device: str, dims: list[int], keepdim: bool):
    """Test aten_amin with multiple dimensions"""

    def fn(x):
        return aten.amin(x, dim=dims, keepdim=keepdim)

    x = torch.randn(3, 4, 5, device=device)
    check_functions_are_equivalent(fn, device, [x])


def test_aten_min_no_dim(device: str):
    """Test aten_min without dimension (returns single value)"""

    def fn(x):
        return aten.min(x)

    x = torch.randn(3, 4, 5, device=device)
    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("keepdim", [True, False])
def test_aten_min_with_dim(device: str, dim: int, keepdim: bool):
    """Test aten_min with dimension (returns values and indices tuple)"""

    def fn(x):
        return aten.min(x, dim=dim, keepdim=keepdim)

    x = torch.randn(3, 4, 5, device=device)
    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.float32])
def test_aten_min_different_dtypes(device: str, dtype: torch.dtype):
    """Test aten_min with different data types"""

    def fn(x):
        return aten.min(x, dim=1, keepdim=False)

    if dtype.is_floating_point:
        x = torch.randn(3, 4, dtype=dtype, device=device)
    else:
        x = torch.randint(-10, 10, (3, 4), dtype=dtype, device=device)

    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(2, 3), (1, 4, 4)])
@pytest.mark.parametrize("value", [-1.5, 42])
def test_fill_scalar_basic(device: str, dtype: torch.dtype, shape: tuple, value: float):
    """Test basic fill.Scalar functionality with different dtypes, shapes, and values"""

    def fn(x):
        return aten.fill.Scalar(x, value)

    # Create input tensor
    x = torch.randn(shape, dtype=dtype, device=device)

    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("shape", [(2, 3), (1, 4, 4)])
@pytest.mark.parametrize("value", [-5, 42])
def test_fill_scalar_integer_dtypes(
    device: str, dtype: torch.dtype, shape: tuple, value: int
):
    """Test fill.Scalar functionality with integer dtypes"""

    def fn(x):
        return aten.fill.Scalar(x, value)

    # Create input tensor with integer values
    x = torch.zeros(shape, dtype=dtype, device=device) + 1

    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("value", [-5, 100])
def test_fill_scalar_integer_values(device: str, value: int):
    """Test fill.Scalar with integer values"""

    def fn(x):
        return aten.fill.Scalar(x, value)

    # Test with float tensor
    x = torch.randn(3, 4, device=device)

    check_functions_are_equivalent(fn, device, [x])


def test_fill_scalar_single_element(device: str):
    """Test fill.Scalar with single element tensor"""

    def fn(x):
        return torch.ops.aten.fill.Scalar(x, 7.5)

    # Single element tensor
    x = torch.tensor([1.0], device=device)

    check_functions_are_equivalent(fn, device, [x])


def test_fill_scalar_zero_dim(device: str):
    """Test fill.Scalar with single element tensor"""

    def fn(x):
        return torch.ops.aten.fill.Scalar(x, 7.5)

    # Single element tensor
    x = torch.tensor(1.0, device=device)

    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.xfail(reason="Fixme, currently off to support eager mode")
def test_max_pool2d_error_message_not_supported_output(device: str):
    def fn(x):
        return aten.max_pool2d_with_indices(x, kernel_size=2, stride=2)

    # Test different sizes
    batch_size, channels = 1, 2
    x = torch.randn(batch_size, channels, 16, 16)
    with pytest.raises(
        BackendCompilerFailed,
        match="The implementation of aten.max_pool2d_with_indices doesn't support returning indices yet.",
    ):
        check_functions_are_equivalent(fn, device, [x])


@pytest.mark.xfail(reason="Fixme, currently off to support eager mode")
def test_max_pool2d_error_message_not_supported_in_graph(device: str):
    def fn(x):
        return aten.max_pool2d_with_indices(x, kernel_size=2, stride=2)[1] * 2

    # Test different sizes
    batch_size, channels = 1, 2
    x = torch.randn(batch_size, channels, 16, 16)
    with pytest.raises(
        BackendCompilerFailed,
        match="The implementation of aten.max_pool2d_with_indices doesn't support returning indices yet.",
    ):
        check_functions_are_equivalent(fn, device, [x])
