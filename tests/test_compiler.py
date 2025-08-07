import torch
from collections.abc import Callable
from max_torch_backend import MaxCompiler
from torch._dynamo import mark_dynamic


def check_functions_are_equivalent(
    fn: Callable, device: str, inputs: list[torch.Tensor]
):
    fn_compiled = torch.compile(backend=MaxCompiler)(fn)

    inputs_on_device = [input_tensor.to(device) for input_tensor in inputs]

    output_original = fn(*inputs_on_device)
    output_compiled = fn_compiled(*inputs_on_device)

    assert type(output_original) == type(output_compiled)

    if isinstance(output_original, torch.Tensor):
        output_original = [output_original]
        output_compiled = [output_compiled]

    for original, compiled in zip(output_original, output_compiled):
        assert original.shape == compiled.shape
        assert original.device == compiled.device
        assert original.dtype == compiled.dtype
        assert torch.allclose(original, compiled, rtol=1e-4)


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


def test_dynamic_shapes(device: str):
    def fn(x, y):
        return x + y

    a = torch.randn(5, 2)
    b = torch.randn(2)

    mark_dynamic(a, 0)

    check_functions_are_equivalent(fn, device, [a, b])

    a = torch.randn(3, 2)
    b = torch.randn(2)

    check_functions_are_equivalent(fn, device, [a, b])
