from pathlib import Path

import pytest
import torch
from torch.ops import aten

import torch_max_backend
import torch_max_backend.torch_compile_backend.compiler
from torch_max_backend import (
    MAPPING_TORCH_ATEN_TO_MAX,
    make_torch_op_from_mojo,
    max_backend,
)
from torch_max_backend.testing import check_functions_are_equivalent


def test_graph_break_with_python_loop_over_tensor(device: str):
    """Test graph break caused by Python loops over tensor elements"""

    def fn_with_python_loop(x):
        x = x * x
        # Python iteration over tensor shapes causes graph breaks
        result = x
        for i in range(int(x[0, 0])):  # This will cause graph break
            result = result * (i + 1)
        return result

    x = torch.randint(1, 3, (3, 2)).to(torch.float32)
    explanation = torch._dynamo.explain(fn_with_python_loop)(x)
    assert explanation.graph_break_count == 1
    assert explanation.graph_count == 2
    check_functions_are_equivalent(fn_with_python_loop, device, [x])


def test_error_message_exception_in_op(monkeypatch):
    def not_working_add(x, y):
        raise RuntimeError("Ho no crash!")

    monkeypatch.setitem(MAPPING_TORCH_ATEN_TO_MAX, aten.add, not_working_add)

    def fn(x, y):
        return x + y

    with pytest.raises(RuntimeError) as exc_info:
        torch.compile(backend=max_backend)(fn)(torch.randn(2, 3), torch.randn(2, 3))

    assert "return x + y" in str(exc_info.value)
    assert "Ho no crash!" in str(exc_info.value)
    assert "torch._ops.aten.aten::add" in str(exc_info.value)
    assert "https://github.com/gabrieldemarmiesse/torch-max-backend/issues" in str(
        exc_info.value
    )
    assert "not_working_add" in str(exc_info.value)


def test_decomposition_overload_packet(monkeypatch):
    """We verify that we skip decomposition for ops that are in the decomposition table,
    and that we registered as an OpOverloadPacket (here `aten.transpose`).
    """

    def fn(x):
        x = x * 2
        return torch.transpose(x, 0, 1) * 2

    # grab the input of init_compiler
    input_gm = None
    init_compiler = (
        torch_max_backend.torch_compile_backend.compiler.BaseMaxCompiler.__init__
    )

    def fake_init_compiler(self, gm, *args, **kwargs):
        nonlocal input_gm
        input_gm = gm
        return init_compiler(self, gm, *args, **kwargs)

    monkeypatch.setattr(
        torch_max_backend.torch_compile_backend.compiler.BaseMaxCompiler,
        "__init__",
        fake_init_compiler,
    )

    a = torch.compile(backend=max_backend)(fn)
    a(torch.randn(2, 3))

    # it's normally decomposed. We check that it's not the case since we
    # implemented it ourselves.
    assert aten.transpose.int in [node.target for node in input_gm.graph.nodes]


def allocate_outputs_grayscale(pic: torch.Tensor) -> torch.Tensor:
    return pic.new_empty(pic.shape[:-1], dtype=torch.float32)


my_torch_grayscale = make_torch_op_from_mojo(
    Path(__file__).parent / "dummy_mojo_kernels",
    "grayscale",
    allocate_outputs_grayscale,
)


def grayscale_eager(pic: torch.Tensor):
    pic = pic.to(dtype=torch.float32)
    r = pic[:, :, 0]
    g = pic[:, :, 1]
    b = pic[:, :, 2]
    return torch.clamp((0.21 * r + 0.71 * g + 0.07 * b), max=255)


def test_mojo_custom_op(device: str):
    img = torch.randn(224, 224, 3, device=device).to(dtype=torch.uint8)

    x = my_torch_grayscale(img)
    y = grayscale_eager(img)
    torch.testing.assert_close(x, y)
    check_functions_are_equivalent(
        grayscale_eager, None, [img], fn_compiled=my_torch_grayscale
    )

    def more_complexe_graph(x: torch.Tensor):
        x = x + 8
        y = my_torch_grayscale(x)
        y = y - 16
        return y

    def more_complexe_graph_eager(x: torch.Tensor):
        x = x + 8
        y = grayscale_eager(x)
        y = y - 16
        return y

    x = more_complexe_graph(img)
    y = more_complexe_graph_eager(img)

    complexe_graph_compiled = torch.compile(backend=max_backend, fullgraph=True)(
        more_complexe_graph
    )
    z = complexe_graph_compiled(img)
    torch.testing.assert_close(x, y)
    torch.testing.assert_close(x, z)

    explanation = torch._dynamo.explain(more_complexe_graph)(img)
    assert explanation.graph_break_count == 0
    assert explanation.graph_count == 1

    check_functions_are_equivalent(
        more_complexe_graph_eager, None, [img], fn_compiled=complexe_graph_compiled
    )


def allocate_outputs_grayscale_multi(
    pic: torch.Tensor, noise: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return tuple(pic.new_empty(noise.shape, dtype=torch.float32) for _ in range(2))


my_torch_grayscale_multi = make_torch_op_from_mojo(
    Path(__file__).parent / "dummy_mojo_kernels",
    "grayscale_multi",
    allocate_outputs_grayscale_multi,
)


def grayscale_multi_eager(
    pic: torch.Tensor, noise: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    pic = pic.to(dtype=torch.float32)
    r = pic[:, :, 0] + noise
    g = pic[:, :, 1] + noise
    b = pic[:, :, 2] + noise

    return (torch.clamp((0.21 * r + 0.71 * g + 0.07 * b), max=255), r)
