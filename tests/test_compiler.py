import operator

import pytest
import torch

from src.max_torch_backend import my_compiler


class TestCompiler:
    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param(
                "cuda",
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="CUDA not available"
                ),
            ),
        ],
    )
    def test_basic_addition(self, device):
        def fn(x, y):
            return x + y

        fn_compiled = torch.compile(backend=my_compiler)(fn)

        a = torch.randn(3).to(device=device)
        b = torch.randn(3).to(device=device)

        output_original = fn(a, b)
        output_compiled = fn_compiled(a, b)

        assert torch.allclose(output_original, output_compiled[0], rtol=1e-5)
        assert output_original.device == output_compiled[0].device

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param(
                "cuda",
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="CUDA not available"
                ),
            ),
        ],
    )
    def test_multiple_operations(self, device):
        def fn(x, y, z):
            return x + y - z

        fn_compiled = torch.compile(backend=my_compiler)(fn)

        a = torch.randn(3).to(device=device)
        b = torch.randn(3).to(device=device)
        c = torch.randn(3).to(device=device)

        output_original = fn(a, b, c)
        output_compiled = fn_compiled(a, b, c)

        assert torch.allclose(output_original, output_compiled[0], rtol=1e-5)
        assert output_original.device == output_compiled[0].device

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param(
                "cuda",
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="CUDA not available"
                ),
            ),
        ],
    )
    def test_multiple_outputs(self, device):
        def fn(x, y, z):
            return x + y + z, x - y

        fn_compiled = torch.compile(backend=my_compiler)(fn)

        a = torch.randn(3).to(device=device)
        b = torch.randn(3).to(device=device)
        c = torch.randn(3).to(device=device)

        outputs_original = fn(a, b, c)
        outputs_compiled = fn_compiled(a, b, c)

        for out_orig, out_comp in zip(outputs_original, outputs_compiled, strict=False):
            assert torch.allclose(out_orig, out_comp, rtol=1e-5)
            assert out_orig.device == out_comp.device

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param(
                "cuda",
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="CUDA not available"
                ),
            ),
        ],
    )
    @pytest.mark.parametrize(
        "operation,torch_func",
        [
            (operator.add, torch.add),
            (operator.sub, torch.sub),
            (operator.mul, torch.mul),
            (operator.truediv, torch.div),
        ],
    )
    def test_supported_operations(self, device, operation, torch_func):
        def fn_op(x, y):
            return operation(x, y)

        def fn_torch(x, y):
            return torch_func(x, y)

        fn_op_compiled = torch.compile(backend=my_compiler)(fn_op)
        fn_torch_compiled = torch.compile(backend=my_compiler)(fn_torch)

        a = torch.randn(3).to(device=device)
        b = torch.randn(3).to(device=device)

        output_op_original = fn_op(a, b)
        output_op_compiled = fn_op_compiled(a, b)
        output_torch_original = fn_torch(a, b)
        output_torch_compiled = fn_torch_compiled(a, b)

        assert torch.allclose(output_op_original, output_op_compiled[0], rtol=1e-5)
        assert torch.allclose(
            output_torch_original, output_torch_compiled[0], rtol=1e-5
        )

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param(
                "cuda",
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="CUDA not available"
                ),
            ),
        ],
    )
    def test_math_functions(self, device):
        def fn(x, y):
            return torch.abs(x) + torch.cos(y) - torch.sin(x)

        fn_compiled = torch.compile(backend=my_compiler)(fn)

        a = torch.randn(3).to(device=device)
        b = torch.randn(3).to(device=device)

        output_original = fn(a, b)
        output_compiled = fn_compiled(a, b)

        assert torch.allclose(output_original, output_compiled[0], rtol=1e-5)
        assert output_original.device == output_compiled[0].device

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param(
                "cuda",
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="CUDA not available"
                ),
            ),
        ],
    )
    def test_complex_function(self, device):
        def fn(x, y, z):
            return x + y + z, x + torch.abs(z) - torch.cos(y) + 1

        fn_compiled = torch.compile(backend=my_compiler)(fn)

        a = torch.randn(3).to(device=device)
        b = torch.randn(3).to(device=device)
        c = torch.randn(3).to(device=device)

        outputs_original = fn(a, b, c)
        outputs_compiled = fn_compiled(a, b, c)

        for out_orig, out_comp in zip(outputs_original, outputs_compiled, strict=False):
            assert torch.allclose(out_orig, out_comp, rtol=1e-5)
            assert out_orig.device == out_comp.device

    @pytest.mark.parametrize("shape", [(3,), (2, 3), (4, 5, 6)])
    def test_different_shapes(self, shape):
        def fn(x, y):
            return x + y

        fn_compiled = torch.compile(backend=my_compiler)(fn)

        a = torch.randn(shape)
        b = torch.randn(shape)

        output_original = fn(a, b)
        output_compiled = fn_compiled(a, b)

        assert torch.allclose(output_original, output_compiled[0], rtol=1e-5)

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param(
                "cuda",
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="CUDA not available"
                ),
            ),
        ],
    )
    def test_scalar_operations(self, device):
        def fn(x):
            return x + 1, x * 2, x - 3

        fn_compiled = torch.compile(backend=my_compiler)(fn)

        a = torch.randn(3).to(device=device)

        outputs_original = fn(a)
        outputs_compiled = fn_compiled(a)

        for out_orig, out_comp in zip(outputs_original, outputs_compiled, strict=False):
            assert torch.allclose(out_orig, out_comp, rtol=1e-5)
            assert out_orig.device == out_comp.device
