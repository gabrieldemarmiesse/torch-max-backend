from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch._dynamo import mark_dynamic
from torch.ops import aten

import torch_max_backend
import torch_max_backend.torch_compile_backend.compiler
from torch_max_backend import make_torch_op_from_mojo, max_backend
from torch_max_backend.testing import check_functions_are_equivalent


def test_basic_training(device: str):
    class MyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 2)

        def forward(self, x):
            return self.linear(x)

    model = MyModel().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    def train_step(x, y):
        model.train()
        optimizer.zero_grad()
        output = model(x)
        loss = F.mse_loss(output, y)
        loss.backward()
        optimizer.step()
        return loss

    a = torch.randn(5, 3).to(device)
    b = torch.randn(5, 2).to(device)

    # We need to reset the parameters before each test
    # to check the model weights afterwards
    model.linear.weight.data.fill_(0.01)
    model.linear.bias.data.fill_(0.01)

    loss_not_compiled = train_step(a, b).cpu().detach().numpy()
    weight_not_compiled = model.linear.weight.data.cpu().numpy()
    bias_not_compiled = model.linear.bias.data.cpu().numpy()

    # Now with the default backed
    model.linear.weight.data.fill_(0.01)
    model.linear.bias.data.fill_(0.01)

    loss_compiled_default = torch.compile()(train_step)(a, b).cpu().detach().numpy()
    weight_compiled_default = model.linear.weight.data.cpu().numpy()
    bias_compiled_default = model.linear.bias.data.cpu().numpy()

    np.testing.assert_allclose(
        loss_not_compiled, loss_compiled_default, rtol=5e-2, atol=5e-3
    )
    np.testing.assert_allclose(
        weight_not_compiled, weight_compiled_default, rtol=5e-2, atol=5e-3
    )
    np.testing.assert_allclose(
        bias_not_compiled, bias_compiled_default, rtol=5e-2, atol=5e-3
    )

    model.linear.weight.data.fill_(0.01)
    model.linear.bias.data.fill_(0.01)

    loss_compiled = (
        torch.compile(backend=max_backend)(train_step)(a, b).cpu().detach().numpy()
    )
    weight_compiled = model.linear.weight.data.cpu().numpy()
    bias_compiled = model.linear.bias.data.cpu().numpy()

    np.testing.assert_allclose(loss_not_compiled, loss_compiled, rtol=5e-2, atol=5e-3)
    np.testing.assert_allclose(
        weight_not_compiled, weight_compiled, rtol=5e-2, atol=5e-3
    )
    np.testing.assert_allclose(bias_not_compiled, bias_compiled, rtol=5e-2, atol=5e-3)


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


def test_mojo_custom_op_multi(device: str):
    img = torch.randn(224, 224, 3, device=device).to(dtype=torch.uint8)
    noise = torch.randint(0, 2, (224, 224), device=device).to(dtype=torch.uint8)

    x1, x2 = my_torch_grayscale_multi(img, noise)
    y1, y2 = grayscale_multi_eager(img, noise)
    torch.testing.assert_close(x1, y1)
    torch.testing.assert_close(x2, y2)
    check_functions_are_equivalent(
        grayscale_multi_eager, None, [img, noise], fn_compiled=my_torch_grayscale_multi
    )

    def more_complexe_graph(
        x: torch.Tensor, noise: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + 8
        y1, y2 = my_torch_grayscale_multi(x, noise)
        y1 = y1 - 16
        y2 = y2 + 3
        return y1, y2

    def more_complexe_graph_eager(
        x: torch.Tensor, noise: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + 8
        y1, y2 = grayscale_multi_eager(x, noise)
        y1 = y1 - 16
        y2 = y2 + 3
        return y1, y2

    x1, x2 = more_complexe_graph(img, noise)
    y1, y2 = more_complexe_graph_eager(img, noise)

    complexe_graph_compiled = torch.compile(backend=max_backend, fullgraph=True)(
        more_complexe_graph
    )
    z1, z2 = complexe_graph_compiled(img, noise)
    torch.testing.assert_close(x1, y1)
    torch.testing.assert_close(x1, z1)
    torch.testing.assert_close(x2, y2)
    torch.testing.assert_close(x2, z2)

    explanation = torch._dynamo.explain(more_complexe_graph)(img, noise)
    assert explanation.graph_break_count == 0
    assert explanation.graph_count == 1

    check_functions_are_equivalent(
        more_complexe_graph_eager,
        None,
        [img, noise],
        fn_compiled=complexe_graph_compiled,
    )


def test_mojo_custom_op_multi_dynamic_dims(device: str):
    img = torch.randn(224, 224, 3, device=device).to(dtype=torch.uint8)
    mark_dynamic(img, 0)
    mark_dynamic(img, 1)
    noise = torch.randint(0, 2, (224, 224), device=device).to(dtype=torch.uint8)
    mark_dynamic(noise, 0)
    mark_dynamic(noise, 1)

    x1, x2 = my_torch_grayscale_multi(img, noise)
    y1, y2 = grayscale_multi_eager(img, noise)
    torch.testing.assert_close(x1, y1)
    torch.testing.assert_close(x2, y2)
    check_functions_are_equivalent(
        grayscale_multi_eager, None, [img, noise], fn_compiled=my_torch_grayscale_multi
    )

    def more_complexe_graph(
        x: torch.Tensor, noise: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + 8
        y1, y2 = my_torch_grayscale_multi(x, noise)
        y1 = y1 - 16
        y2 = y2 + 3
        return y1, y2

    def more_complexe_graph_eager(
        x: torch.Tensor, noise: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + 8
        y1, y2 = grayscale_multi_eager(x, noise)
        y1 = y1 - 16
        y2 = y2 + 3
        return y1, y2

    x1, x2 = more_complexe_graph(img, noise)
    y1, y2 = more_complexe_graph_eager(img, noise)

    complexe_graph_compiled = torch.compile(backend=max_backend, fullgraph=True)(
        more_complexe_graph
    )
    z1, z2 = complexe_graph_compiled(img, noise)
    torch.testing.assert_close(x1, y1)
    torch.testing.assert_close(x1, z1)
    torch.testing.assert_close(x2, y2)
    torch.testing.assert_close(x2, z2)

    explanation = torch._dynamo.explain(more_complexe_graph)(img, noise)
    assert explanation.graph_break_count == 0
    assert explanation.graph_count == 1

    check_functions_are_equivalent(
        more_complexe_graph_eager,
        None,
        [img, noise],
        fn_compiled=complexe_graph_compiled,
    )
