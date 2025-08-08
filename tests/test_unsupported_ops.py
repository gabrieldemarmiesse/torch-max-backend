import pytest
import torch
from torch._dynamo.exc import BackendCompilerFailed
from max_torch_backend import MaxCompiler


def test_unsupported_function_error_message():
    def fn(x):
        return torch.exp(x)

    fn_compiled = torch.compile(backend=MaxCompiler)(fn)

    a = torch.randn(3)

    with pytest.raises(
        BackendCompilerFailed, match="Function .* not supported by the Max backend yet"
    ):
        fn_compiled(a)


def test_unsupported_matmul_error():
    def fn(x, y):
        return torch.matmul(x, y)

    fn_compiled = torch.compile(backend=MaxCompiler)(fn)

    a = torch.randn(3, 4)
    b = torch.randn(4, 5)

    with pytest.raises(
        BackendCompilerFailed, match="Function .* not supported by the Max backend yet"
    ):
        fn_compiled(a, b)


def test_unsupported_log_error():
    def fn(x):
        return torch.log(x)

    fn_compiled = torch.compile(backend=MaxCompiler)(fn)

    a = torch.randn(3).abs()  # Ensure positive values for log

    with pytest.raises(
        BackendCompilerFailed, match="Function .* not supported by the Max backend yet"
    ):
        fn_compiled(a)


def test_unsupported_max_error():
    def fn(x):
        return torch.max(x)

    fn_compiled = torch.compile(backend=MaxCompiler)(fn)

    a = torch.randn(3, 4)

    with pytest.raises(
        BackendCompilerFailed, match="Function .* not supported by the Max backend yet"
    ):
        fn_compiled(a)


def test_error_message_includes_function_name():
    def fn(x):
        return torch.tanh(x)

    fn_compiled = torch.compile(backend=MaxCompiler)(fn)

    a = torch.randn(3)

    with pytest.raises(BackendCompilerFailed) as exc_info:
        fn_compiled(a)

    error_message = str(exc_info.value)
    assert "tanh" in error_message or "torch.tanh" in error_message
    assert "not supported by the Max backend yet" in error_message
