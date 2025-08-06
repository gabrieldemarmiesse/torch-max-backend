import pytest
import torch

from src.max_torch_backend import my_compiler


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
def test_basic_addition(device):
    def fn(x, y):
        return x + y

    fn_compiled = torch.compile(backend=my_compiler)(fn)

    a = torch.randn(3).to(device=device)
    b = torch.randn(3).to(device=device)

    output_original = fn(a, b)
    output_compiled = fn_compiled(a, b)

    assert torch.allclose(output_original[0], output_compiled[0], rtol=1e-5)
    assert output_original[0].device == output_compiled[0].device
