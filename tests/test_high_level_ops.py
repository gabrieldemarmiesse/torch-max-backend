import pytest
import torch

from torch_max_backend.testing import (
    Conf,
    check_functions_are_equivalent,
    check_outputs,
)


def test_basic_addition(conf: Conf):
    def fn(x, y):
        return x + y

    a = torch.randn(3)
    b = torch.randn(3)

    check_outputs(fn, conf, [a, b])


def test_iadd(conf: Conf):
    if conf.device.startswith("max") and not conf.compile:
        pytest.xfail("This fails for some reason. Segfault. To investigate later.")

    def fn(x, y):
        x += y
        return x

    a = torch.randn(3)
    b = torch.randn(3)

    check_outputs(fn, conf, [a, b])


def test_t_method(conf: Conf):
    def fn(x):
        return x.t()

    a = torch.randn(3, 4)

    check_outputs(fn, conf, [a])


def test_t_function(device: str):
    def fn(x):
        return torch.t(x)

    a = torch.randn(3, 4)

    check_functions_are_equivalent(fn, device, [a])


def test_new_ones(device: str):
    def fn(x):
        return x.new_ones((3, 3))

    a = torch.randn(3)

    check_functions_are_equivalent(fn, device, [a])


def test_new_ones_device(device: str):
    def fn(x):
        return x.new_ones((3, 3), device=torch.device(device))

    a = torch.randn(3)

    check_functions_are_equivalent(fn, "cpu", [a])


def test_new_ones_dtype(device: str):
    def fn(x):
        return x.new_ones((3, 3), dtype=torch.uint8)

    a = torch.randn(3)

    check_functions_are_equivalent(fn, device, [a])


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


def test_subtraction_different_dtypes(device: str, tensor_shapes: tuple):
    def fn(x, y):
        return x - y

    a = torch.randn(tensor_shapes, dtype=torch.float32)
    b = torch.randint(0, 10, tensor_shapes, dtype=torch.int64)

    check_functions_are_equivalent(fn, device, [a, b])


def test_multiplication(device: str, tensor_shapes: tuple):
    def fn(x, y):
        return x * y

    a = torch.randn(tensor_shapes)
    b = torch.randn(tensor_shapes)

    check_functions_are_equivalent(fn, device, [a, b])


def test_multiplication_int32(device: str, tensor_shapes: tuple):
    def fn(x, y):
        return x * y

    a = torch.randint(0, 10, size=tensor_shapes, dtype=torch.int32)
    b = torch.randint(0, 10, size=tensor_shapes, dtype=torch.int32)

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
