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
