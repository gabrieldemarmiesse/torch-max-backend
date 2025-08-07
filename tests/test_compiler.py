import torch
import torch.nn.functional as F
from collections.abc import Callable
from max_torch_backend import MaxCompiler
import pytest
from torch._dynamo import mark_dynamic


def check_functions_are_equivalent(
    fn: Callable,
    device: str | None,
    inputs: list[torch.Tensor],
    fn_compiled: Callable | None = None,
):
    fn_compiled = fn_compiled or torch.compile(backend=MaxCompiler)(fn)
    if device is not None:
        inputs = [input_tensor.to(device) for input_tensor in inputs]

    output_original = fn(*inputs)
    output_compiled = fn_compiled(*inputs)

    assert type(output_original) == type(output_compiled)

    if isinstance(output_original, torch.Tensor):
        output_original = [output_original]
        output_compiled = [output_compiled]

    for original, compiled in zip(output_original, output_compiled):
        assert original.shape == compiled.shape
        assert original.device == compiled.device
        assert original.dtype == compiled.dtype
        assert torch.allclose(original, compiled, rtol=1e-4, atol=1e-5)


def test_basic_addition(device: str):
    def fn(x, y):
        return x + y

    a = torch.randn(3)
    b = torch.randn(3)

    check_functions_are_equivalent(fn, device, [a, b])


def test_operator_add(device: str, tensor_shapes: tuple):
    def fn(x, y):
        return x + y

    a = torch.randn(tensor_shapes)
    b = torch.randn(tensor_shapes)

    check_functions_are_equivalent(fn, device, [a, b])


def test_subtraction(device: str, tensor_shapes: tuple):
    def fn(x, y):
        return x - y

    a = torch.randn(tensor_shapes)
    b = torch.randn(tensor_shapes)

    check_functions_are_equivalent(fn, device, [a, b])


def test_multiplication(device: str, tensor_shapes: tuple):
    def fn(x, y):
        return x * y

    a = torch.randn(tensor_shapes)
    b = torch.randn(tensor_shapes)

    check_functions_are_equivalent(fn, device, [a, b])


def test_division(device: str, tensor_shapes: tuple):
    def fn(x, y):
        return x / y

    a = torch.randn(tensor_shapes)
    b = torch.randn(tensor_shapes) + 1.0  # Avoid division by zero

    check_functions_are_equivalent(fn, device, [a, b])


def test_floor_division(device: str, tensor_shapes: tuple):
    def fn(x, y):
        return x // y

    a = torch.randn(tensor_shapes) * 10
    b = torch.randn(tensor_shapes).abs() + 1.0  # Avoid division by zero

    check_functions_are_equivalent(fn, device, [a, b])


def test_power(device: str, tensor_shapes: tuple):
    def fn(x, y):
        return x**y

    a = torch.randn(tensor_shapes).abs() + 0.1  # Avoid negative base
    b = torch.randn(tensor_shapes) * 2  # Keep exponent reasonable

    check_functions_are_equivalent(fn, device, [a, b])


def test_modulo(device: str, tensor_shapes: tuple):
    def fn(x, y):
        return x % y

    a = torch.randn(tensor_shapes) * 10
    b = torch.randn(tensor_shapes).abs() + 1.0  # Avoid division by zero

    check_functions_are_equivalent(fn, device, [a, b])


def test_abs(device: str, tensor_shapes: tuple):
    def fn(x):
        return torch.abs(x)

    a = torch.randn(tensor_shapes)

    check_functions_are_equivalent(fn, device, [a])


def test_cos(device: str, tensor_shapes: tuple):
    def fn(x):
        return torch.cos(x)

    a = torch.randn(tensor_shapes)

    check_functions_are_equivalent(fn, device, [a])


def test_sin(device: str, tensor_shapes: tuple):
    def fn(x):
        return torch.sin(x)

    a = torch.randn(tensor_shapes)

    check_functions_are_equivalent(fn, device, [a])


def test_cat(device: str, tensor_shapes: tuple):
    def fn(x, y):
        return torch.cat([x, y], dim=0)

    a = torch.randn(tensor_shapes)
    b = torch.randn(tensor_shapes)

    check_functions_are_equivalent(fn, device, [a, b])


def test_combination_add_mul(device: str, tensor_shapes: tuple):
    def fn(x, y, z):
        return (x + y) * z

    a = torch.randn(tensor_shapes)
    b = torch.randn(tensor_shapes)
    c = torch.randn(tensor_shapes)

    check_functions_are_equivalent(fn, device, [a, b, c])


def test_combination_sub_div(device: str, tensor_shapes: tuple):
    def fn(x, y, z):
        return (x - y) / z

    a = torch.randn(tensor_shapes)
    b = torch.randn(tensor_shapes)
    c = torch.randn(tensor_shapes) + 1.0  # Avoid division by zero

    check_functions_are_equivalent(fn, device, [a, b, c])


def test_combination_trig_arithmetic(device: str, tensor_shapes: tuple):
    def fn(x, y):
        return torch.sin(x) + torch.cos(y)

    a = torch.randn(tensor_shapes)
    b = torch.randn(tensor_shapes)

    check_functions_are_equivalent(fn, device, [a, b])


def test_combination_abs_mul_add(device: str, tensor_shapes: tuple):
    def fn(x, y, z):
        return torch.abs(x) * y + z

    a = torch.randn(tensor_shapes)
    b = torch.randn(tensor_shapes)
    c = torch.randn(tensor_shapes)

    check_functions_are_equivalent(fn, device, [a, b, c])


def test_combination_pow_mod(device: str, tensor_shapes: tuple):
    def fn(x, y):
        return (x**2) % y

    a = torch.randn(tensor_shapes).abs() + 0.1
    b = torch.randn(tensor_shapes).abs() + 1.0

    check_functions_are_equivalent(fn, device, [a, b])


def test_complex_combination(device: str, tensor_shapes: tuple):
    def fn(x, y, z):
        return torch.abs(torch.sin(x) * y + torch.cos(z))

    a = torch.randn(tensor_shapes)
    b = torch.randn(tensor_shapes)
    c = torch.randn(tensor_shapes)

    check_functions_are_equivalent(fn, device, [a, b, c])


def test_scalar_shapes(device: str):
    def fn(x, y):
        return x + y * 2

    a = torch.randn(())  # Scalar tensor
    b = torch.randn(())

    check_functions_are_equivalent(fn, device, [a, b])


def test_broadcasting_compatible(device: str):
    def fn(x, y):
        return x + y

    a = torch.randn(5, 1)
    b = torch.randn(1, 5)

    check_functions_are_equivalent(fn, device, [a, b])


def test_conv2d_basic(device: str):
    """Test basic conv2d with default parameters"""

    def fn(x, w):
        return F.conv2d(x, w)

    batch_size, in_channels, height, width = 2, 3, 8, 8
    out_channels, kernel_size = 4, 3

    x = torch.randn(batch_size, in_channels, height, width)
    w = torch.randn(out_channels, in_channels, kernel_size, kernel_size)

    check_functions_are_equivalent(fn, device, [x, w])


def test_conv2d_with_bias(device: str):
    """Test conv2d with bias"""

    def fn(x, w, b):
        return F.conv2d(x, w, b)

    batch_size, in_channels, height, width = 2, 3, 8, 8
    out_channels, kernel_size = 4, 3

    x = torch.randn(batch_size, in_channels, height, width)
    w = torch.randn(out_channels, in_channels, kernel_size, kernel_size)
    b = torch.randn(out_channels)

    check_functions_are_equivalent(fn, device, [x, w, b])


def test_conv2d_stride_int(device: str):
    """Test conv2d with integer stride"""

    def fn(x, w):
        return F.conv2d(x, w, stride=2)

    batch_size, in_channels, height, width = 2, 3, 16, 16
    out_channels, kernel_size = 4, 3

    x = torch.randn(batch_size, in_channels, height, width)
    w = torch.randn(out_channels, in_channels, kernel_size, kernel_size)

    check_functions_are_equivalent(fn, device, [x, w])


def test_conv2d_stride_tuple(device: str):
    """Test conv2d with tuple stride"""

    def fn(x, w):
        return F.conv2d(x, w, stride=(2, 3))

    batch_size, in_channels, height, width = 2, 3, 16, 16
    out_channels, kernel_size = 4, 3

    x = torch.randn(batch_size, in_channels, height, width)
    w = torch.randn(out_channels, in_channels, kernel_size, kernel_size)

    check_functions_are_equivalent(fn, device, [x, w])


def test_conv2d_padding_int(device: str):
    """Test conv2d with integer padding"""

    def fn(x, w):
        return F.conv2d(x, w, padding=1)

    batch_size, in_channels, height, width = 2, 3, 8, 8
    out_channels, kernel_size = 4, 3

    x = torch.randn(batch_size, in_channels, height, width)
    w = torch.randn(out_channels, in_channels, kernel_size, kernel_size)

    check_functions_are_equivalent(fn, device, [x, w])


def test_conv2d_padding_tuple(device: str):
    """Test conv2d with tuple padding"""

    def fn(x, w):
        return F.conv2d(x, w, padding=(1, 2))

    batch_size, in_channels, height, width = 2, 3, 8, 8
    out_channels, kernel_size = 4, 3

    x = torch.randn(batch_size, in_channels, height, width)
    w = torch.randn(out_channels, in_channels, kernel_size, kernel_size)

    check_functions_are_equivalent(fn, device, [x, w])


@pytest.mark.xfail(reason="Dilation not implemented yet on max")
def test_conv2d_dilation_int(device: str):
    """Test conv2d with integer dilation"""

    def fn(x, w):
        return F.conv2d(x, w, dilation=2)

    batch_size, in_channels, height, width = 2, 3, 16, 16
    out_channels, kernel_size = 4, 3

    x = torch.randn(batch_size, in_channels, height, width)
    w = torch.randn(out_channels, in_channels, kernel_size, kernel_size)

    check_functions_are_equivalent(fn, device, [x, w])


@pytest.mark.xfail(reason="Dilation not implemented yet on max")
def test_conv2d_dilation_tuple(device: str):
    """Test conv2d with tuple dilation"""

    def fn(x, w):
        return F.conv2d(x, w, dilation=(2, 3))

    batch_size, in_channels, height, width = 2, 3, 16, 16
    out_channels, kernel_size = 4, 3

    x = torch.randn(batch_size, in_channels, height, width)
    w = torch.randn(out_channels, in_channels, kernel_size, kernel_size)

    check_functions_are_equivalent(fn, device, [x, w])


def test_conv2d_all_params(device: str):
    """Test conv2d with all parameters specified"""

    def fn(x, w, b):
        return F.conv2d(x, w, b, stride=2, padding=1, dilation=1)

    batch_size, in_channels, height, width = 2, 3, 16, 16
    out_channels, kernel_size = 4, 3

    x = torch.randn(batch_size, in_channels, height, width)
    w = torch.randn(out_channels, in_channels, kernel_size, kernel_size)
    b = torch.randn(out_channels)

    check_functions_are_equivalent(fn, device, [x, w, b])


def test_conv2d_1x1_kernel(device: str):
    """Test conv2d with 1x1 kernel (pointwise convolution)"""

    def fn(x, w):
        return F.conv2d(x, w)

    batch_size, in_channels, height, width = 2, 3, 8, 8
    out_channels = 4

    x = torch.randn(batch_size, in_channels, height, width)
    w = torch.randn(out_channels, in_channels, 1, 1)

    check_functions_are_equivalent(fn, device, [x, w])


def test_conv2d_large_kernel(device: str):
    """Test conv2d with larger kernel"""

    def fn(x, w):
        return F.conv2d(x, w, padding=2)

    batch_size, in_channels, height, width = 2, 3, 16, 16
    out_channels, kernel_size = 4, 5

    x = torch.randn(batch_size, in_channels, height, width)
    w = torch.randn(out_channels, in_channels, kernel_size, kernel_size)

    check_functions_are_equivalent(fn, device, [x, w])


def test_conv2d_asymmetric_kernel(device: str):
    """Test conv2d with asymmetric kernel"""

    def fn(x, w):
        return F.conv2d(x, w, padding=(1, 2))

    batch_size, in_channels, height, width = 2, 3, 8, 8
    out_channels = 4

    x = torch.randn(batch_size, in_channels, height, width)
    w = torch.randn(out_channels, in_channels, 3, 5)  # 3x5 kernel

    check_functions_are_equivalent(fn, device, [x, w])


@pytest.mark.xfail(reason="Different input sizes not handled yet")
def test_conv2d_different_input_sizes(device: str):
    """Test conv2d with different input tensor sizes"""

    def fn(x, w):
        return F.conv2d(x, w, padding=1)

    # Test various input sizes
    sizes = [(1, 1, 4, 4), (3, 8, 32, 32), (2, 16, 64, 64)]

    for batch_size, in_channels, height, width in sizes:
        out_channels, kernel_size = 4, 3

        x = torch.randn(batch_size, in_channels, height, width)
        w = torch.randn(out_channels, in_channels, kernel_size, kernel_size)

        check_functions_are_equivalent(fn, device, [x, w])


def test_conv2d_edge_cases(device: str):
    """Test conv2d edge cases"""

    # Single pixel output
    def fn1(x, w):
        return F.conv2d(x, w)

    batch_size, in_channels = 1, 2
    out_channels, kernel_size = 3, 3

    x = torch.randn(batch_size, in_channels, 3, 3)  # Exactly kernel size
    w = torch.randn(out_channels, in_channels, kernel_size, kernel_size)

    check_functions_are_equivalent(fn1, device, [x, w])


def test_conv2d_combined_with_other_ops(device: str):
    """Test conv2d combined with other operations"""

    def fn(x, w, b, y):
        conv_out = F.conv2d(x, w, b, padding=1)
        return conv_out + y

    batch_size, in_channels, height, width = 2, 3, 8, 8
    out_channels, kernel_size = 4, 3

    x = torch.randn(batch_size, in_channels, height, width)
    w = torch.randn(out_channels, in_channels, kernel_size, kernel_size)
    b = torch.randn(out_channels)
    # y should have same shape as conv output: (2, 4, 8, 8)
    y = torch.randn(batch_size, out_channels, height, width)

    check_functions_are_equivalent(fn, device, [x, w, b, y])


def test_embedding_basic(device: str):
    """Test basic embedding lookup"""

    def fn(indices, weight):
        return F.embedding(indices, weight)

    vocab_size, embedding_dim = 10, 5
    seq_length = 4

    # Create indices tensor (LongTensor)
    indices = torch.randint(0, vocab_size, (seq_length,))
    weight = torch.randn(vocab_size, embedding_dim)

    check_functions_are_equivalent(fn, device, [indices, weight])


def test_embedding_2d_indices(device: str):
    """Test embedding with 2D indices (batch processing)"""

    def fn(indices, weight):
        return F.embedding(indices, weight)

    vocab_size, embedding_dim = 20, 8
    batch_size, seq_length = 3, 6

    indices = torch.randint(0, vocab_size, (batch_size, seq_length))
    weight = torch.randn(vocab_size, embedding_dim)

    check_functions_are_equivalent(fn, device, [indices, weight])


def test_embedding_3d_indices(device: str):
    """Test embedding with 3D indices"""

    def fn(indices, weight):
        return F.embedding(indices, weight)

    vocab_size, embedding_dim = 15, 4
    batch_size, seq_length, depth = 2, 3, 4

    indices = torch.randint(0, vocab_size, (batch_size, seq_length, depth))
    weight = torch.randn(vocab_size, embedding_dim)

    check_functions_are_equivalent(fn, device, [indices, weight])


def test_embedding_single_index(device: str):
    """Test embedding with single index (scalar)"""

    def fn(indices, weight):
        return F.embedding(indices, weight)

    vocab_size, embedding_dim = 10, 3

    indices = torch.tensor(5)  # Scalar tensor
    weight = torch.randn(vocab_size, embedding_dim)

    check_functions_are_equivalent(fn, device, [indices, weight])


def test_embedding_combined_with_other_ops(device: str):
    """Test embedding combined with other operations"""

    def fn(indices, weight, bias):
        embedded = F.embedding(indices, weight)
        return embedded + bias

    vocab_size, embedding_dim = 10, 5
    seq_length = 4

    indices = torch.randint(0, vocab_size, (seq_length,))
    weight = torch.randn(vocab_size, embedding_dim)
    bias = torch.randn(embedding_dim)

    check_functions_are_equivalent(fn, device, [indices, weight, bias])


def test_embedding_with_padding_idx(device: str):
    """Test embedding with padding_idx parameter"""

    def fn(indices, weight):
        return F.embedding(indices, weight, padding_idx=0)

    vocab_size, embedding_dim = 8, 4

    # Include padding index (0) in the indices
    indices = torch.tensor([[0, 1, 2, 0, 3], [4, 0, 5, 6, 0]])
    weight = torch.randn(vocab_size, embedding_dim)

    check_functions_are_equivalent(fn, device, [indices, weight])


def test_embedding_padding_idx_different_values(device: str):
    """Test embedding with different padding_idx values"""

    def fn_pad_0(indices, weight):
        return F.embedding(indices, weight, padding_idx=0)

    def fn_pad_2(indices, weight):
        return F.embedding(indices, weight, padding_idx=2)

    vocab_size, embedding_dim = 6, 3

    indices_0 = torch.tensor([0, 1, 3, 0])  # Using 0 as padding
    indices_2 = torch.tensor([1, 2, 4, 2])  # Using 2 as padding
    weight = torch.randn(vocab_size, embedding_dim)

    check_functions_are_equivalent(fn_pad_0, device, [indices_0, weight])
    check_functions_are_equivalent(fn_pad_2, device, [indices_2, weight])


def test_embedding_padding_idx_scalar(device: str):
    """Test embedding with padding_idx on scalar indices"""

    def fn(indices, weight):
        return F.embedding(indices, weight, padding_idx=0)

    vocab_size, embedding_dim = 5, 3

    indices = torch.tensor(0)  # Scalar padding index
    weight = torch.randn(vocab_size, embedding_dim)

    check_functions_are_equivalent(fn, device, [indices, weight])


def test_tensor_slice_basic(device: str):
    def fn(x):
        return x[1:3]  # Basic slice along first dimension

    x = torch.randn(5, 4)

    check_functions_are_equivalent(fn, device, [x])


def test_tensor_slice_2d(device: str):
    def fn(x):
        return x[1:3, 0:2]  # Slice along both dimensions

    x = torch.randn(5, 4)

    check_functions_are_equivalent(fn, device, [x])


def test_tensor_slice_negative_index(device: str):
    def fn(x):
        return x[-2:]  # Negative slice

    x = torch.randn(5, 3)

    check_functions_are_equivalent(fn, device, [x])


def test_to_float(device: str):
    def fn(x):
        return x.float()

    x = torch.randint(0, 10, (5,))

    check_functions_are_equivalent(fn, device, [x])


class MaxCompilerCallCount:
    def __init__(self):
        self.call_count = 0

    def __call__(self, *args, **kwargs):
        self.call_count += 1
        return MaxCompiler(*args, **kwargs)


def test_dynamic_shapes(device: str):
    """Testing the behavior with mark_dynamic()."""

    def fn(x, y):
        return x + y

    counter = MaxCompilerCallCount()
    fn_compiled = torch.compile(backend=counter)(fn)

    a = torch.randn(20, 2).to(device)
    b = torch.randn(2).to(device)

    mark_dynamic(a, 0)

    check_functions_are_equivalent(fn, None, [a, b], fn_compiled)

    for i in range(5, 15):
        a = torch.randn(i, 2).to(device)
        b = torch.randn(2).to(device)
        mark_dynamic(a, 0)

        check_functions_are_equivalent(fn, None, [a, b], fn_compiled)
        # Ensure only one instance of the MaxCompiler is created
    assert counter.call_count == 1


def test_recompilation(device: str):
    """Testing the behavior without mark_dynamic()."""

    def fn(x, y):
        return x + y

    counter = MaxCompilerCallCount()
    fn_compiled = torch.compile(backend=counter)(fn)

    a = torch.randn(20, 2).to(device)
    b = torch.randn(2).to(device)

    check_functions_are_equivalent(fn, None, [a, b], fn_compiled)

    a = torch.randn(10, 2).to(device)
    b = torch.randn(2).to(device)

    check_functions_are_equivalent(fn, None, [a, b], fn_compiled)
    # Ensure a second instance of the MaxCompiler is created
    assert counter.call_count == 2

    # TODO: Make it work if called with more shapes (dynamo doesn't recompile)
