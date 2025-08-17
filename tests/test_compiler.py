import torch
import torch.nn.functional as F
from torch_max_backend import max_backend
from torch._dynamo import mark_dynamic
import io
from unittest.mock import patch
import numpy as np
from torch_max_backend.testing import check_functions_are_equivalent
from torch_max_backend import MAPPING_TORCH_ATEN_TO_MAX
from torch.ops import aten
import pytest
from torch._dynamo.exc import BackendCompilerFailed


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


def test_get_attr_parameter(device: str):
    """Test get_attr node with parameter access"""

    class ParameterModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(3, 4))
            self.bias = torch.nn.Parameter(torch.randn(4))

        def forward(self, x):
            # This will create get_attr nodes for self.weight and self.bias
            return x @ self.weight + self.bias

    module = ParameterModule().to(device)

    x = torch.randn(2, 3)

    # Verify get_attr nodes are in the graph
    # Test with tracing to ensure get_attr nodes are created
    traced = torch.fx.symbolic_trace(module)
    get_attr_nodes = [node for node in traced.graph.nodes if node.op == "get_attr"]
    assert len(get_attr_nodes) >= 2, (
        f"Expected at least 2 get_attr nodes, got {len(get_attr_nodes)}"
    )
    # Should have nodes for weight and bias
    targets = [node.target for node in get_attr_nodes]
    assert "weight" in targets
    assert "bias" in targets

    check_functions_are_equivalent(module, device, [x])


def test_get_attr_nested_parameter(device: str):
    """Test get_attr node with nested module parameter access"""

    class NestedModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 4)
            self.scale = torch.nn.Parameter(torch.tensor(2.0))

        def forward(self, x):
            # This will create get_attr nodes for nested parameters
            return self.linear(x) * self.scale

    module = NestedModule().to(device)

    x = torch.randn(2, 3)

    # Verify get_attr nodes are in the graph
    traced = torch.fx.symbolic_trace(module)
    get_attr_nodes = [node for node in traced.graph.nodes if node.op == "get_attr"]
    # Should have at least the scale parameter as get_attr
    # Linear might be optimized into call_module instead
    targets = [node.target for node in get_attr_nodes]
    assert "scale" in targets

    check_functions_are_equivalent(module, device, [x])


def test_get_attr_buffer(device: str):
    """Test get_attr node with buffer access"""

    class ModuleWithBuffer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("running_mean", torch.zeros(4))
            self.weight = torch.nn.Parameter(torch.ones(4))

        def forward(self, x):
            # This will create get_attr nodes for both parameter and buffer
            return (x + self.running_mean) * self.weight

    module = ModuleWithBuffer().to(device)

    x = torch.randn(2, 4)

    # Verify get_attr nodes are in the graph
    traced = torch.fx.symbolic_trace(module)
    get_attr_nodes = [node for node in traced.graph.nodes if node.op == "get_attr"]
    targets = [node.target for node in get_attr_nodes]
    # Should have weight and running_mean
    assert "weight" in targets
    assert "running_mean" in targets

    check_functions_are_equivalent(module, device, [x])


def test_get_attr_multiple_parameters(device: str):
    """Test get_attr nodes with multiple parameters"""

    class MultiParamModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight1 = torch.nn.Parameter(torch.randn(3, 4))
            self.weight2 = torch.nn.Parameter(torch.randn(4, 2))
            self.bias1 = torch.nn.Parameter(torch.randn(4))
            self.bias2 = torch.nn.Parameter(torch.randn(2))

        def forward(self, x):
            # Multiple get_attr nodes will be created
            h = x @ self.weight1 + self.bias1
            return h @ self.weight2 + self.bias2

    module = MultiParamModule().to(device)

    x = torch.randn(2, 3)

    check_functions_are_equivalent(module, device, [x])


def test_get_attr_with_arithmetic(device: str):
    """Test get_attr nodes combined with arithmetic operations"""

    class ArithmeticModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(3.0))
            self.offset = torch.nn.Parameter(torch.tensor(1.5))

        def forward(self, x, y):
            # get_attr nodes will be used for scale and offset
            return (x * self.scale + self.offset) + y

    module = ArithmeticModule().to(device)

    x = torch.randn(2, 3)
    y = torch.randn(2, 3)

    check_functions_are_equivalent(module, device, [x, y])


def test_get_attr_constant_tensor(device: str):
    """Test get_attr node with constant tensor"""

    class ConstantModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Register a constant tensor (not a parameter)
            self.register_buffer(
                "constant", torch.tensor([1.0, 2.0, 3.0]), persistent=False
            )

        def forward(self, x):
            # This will create a get_attr node for the constant
            return x + self.constant

    module = ConstantModule().to(device)

    x = torch.randn(2, 3)

    check_functions_are_equivalent(module, device, [x])


def test_get_attr_deeply_nested(device: str):
    """Test get_attr node with deeply nested module hierarchy"""

    class InnerModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner_weight = torch.nn.Parameter(torch.randn(3, 3))

    class MiddleModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = InnerModule()
            self.middle_bias = torch.nn.Parameter(torch.randn(3))

    class OuterModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.middle = MiddleModule()

        def forward(self, x):
            # This will create get_attr nodes with dotted paths
            return x @ self.middle.inner.inner_weight + self.middle.middle_bias

    module = OuterModule().to(device)

    x = torch.randn(2, 3)

    check_functions_are_equivalent(module, device, [x])


def test_get_attr_mixed_with_functions(device: str):
    """Test get_attr nodes mixed with function calls"""

    class MixedModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(3, 4))

        def forward(self, x):
            # Mix get_attr with function calls
            linear_out = x @ self.weight
            return torch.sin(linear_out) + torch.cos(linear_out)

    module = MixedModule().to(device)

    x = torch.randn(2, 3)

    check_functions_are_equivalent(module, device, [x])


def test_get_attr_simple_constant(device: str):
    """Test get_attr with a simple constant parameter"""

    class SimpleConstantModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Create a simple parameter that will definitely create get_attr
            self.constant = torch.nn.Parameter(torch.tensor([2.0, 3.0, 4.0]))

        def forward(self, x):
            # Simple addition that should create get_attr node
            return x + self.constant

    module = SimpleConstantModule().to(device)

    x = torch.randn(3)

    # Verify get_attr nodes are in the graph
    traced = torch.fx.symbolic_trace(module)
    get_attr_nodes = [node for node in traced.graph.nodes if node.op == "get_attr"]
    assert len(get_attr_nodes) >= 1
    targets = [node.target for node in get_attr_nodes]
    assert "constant" in targets

    check_functions_are_equivalent(module, device, [x])


def test_get_attr_torch_tensor(device: str):
    class SimpleConstantModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.constant = torch.tensor([2.0, 3.0, 4.0]).to(device)

        def forward(self, x):
            # Simple addition that should create get_attr node
            return x + self.constant

    module = SimpleConstantModule().to(device)

    x = torch.randn(3)

    # Verify get_attr nodes are in the graph
    traced = torch.fx.symbolic_trace(module)
    get_attr_nodes = [node for node in traced.graph.nodes if node.op == "get_attr"]
    assert len(get_attr_nodes) >= 1
    targets = [node.target for node in get_attr_nodes]
    assert "constant" in targets

    check_functions_are_equivalent(module, device, [x])


# Graph Break Tests
def test_graph_break_with_print(device: str):
    """Test graph break caused by print statements"""

    def fn_with_print(x):
        a = x + 1
        print(f"Processing tensor with shape: {x.shape}")
        return a * 2

    x = torch.randn(3, 4)
    explanation = torch._dynamo.explain(fn_with_print)(x)
    assert explanation.graph_break_count == 1
    assert explanation.graph_count == 2

    # This should cause a graph break due to print
    with patch("sys.stdout", new_callable=io.StringIO):
        check_functions_are_equivalent(fn_with_print, device, [x])


def test_graph_break_with_item_access(device: str):
    def fn_with_item(x):
        x = x * x
        if x[0, 0] > 0:
            return x * 2
        else:
            return x

    x = torch.randn(2, 3) + 1.0  # Ensure non-zero values
    explanation = torch._dynamo.explain(fn_with_item)(x)
    assert explanation.graph_break_count == 1
    assert explanation.graph_count == 2
    check_functions_are_equivalent(fn_with_item, device, [x])


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


def test_graph_break_with_python_loop_over_tensor_complexe_dtypes(device: str):
    """Test graph break caused by Python loops over tensor elements"""

    def fn_with_python_loop(x):
        x = x * x
        result = x
        for i in range(int(x[0, 0])):  # This will cause graph break
            result = (result * (i + 1)).to(torch.int32)
        return result

    x = torch.randint(1, 3, (3, 2)).to(torch.int32)
    explanation = torch._dynamo.explain(fn_with_python_loop)(x)
    assert explanation.graph_break_count == 1
    assert explanation.graph_count == 2
    check_functions_are_equivalent(fn_with_python_loop, device, [x])


def test_graph_break_with_string_operations(device: str):
    """Test graph break caused by string operations"""

    def fn_with_string_ops(x):
        x = x * 2
        tensor_info = f"Tensor shape: {x}, dtype: {x.dtype}"
        # Just return the tensor since we can't return strings
        return x * (len(tensor_info) % 10)

    x = torch.randn(2, 3)
    explanation = torch._dynamo.explain(fn_with_string_ops)(x)
    assert explanation.graph_break_count == 1
    assert explanation.graph_count == 2
    # This should cause graph breaks due to string operations
    check_functions_are_equivalent(fn_with_string_ops, device, [x])


def test_multiple_graph_breaks_in_sequence(device: str):
    """Test function with multiple operations that cause graph breaks"""

    def fn_with_multiple_breaks(x):
        # First graph break: print
        x = x * x
        print(f"Input shape: {x.shape}")

        x = x + 1

        print(f"Result computed {x.shape}")

        return x * x

    x = torch.randn(2, 3)
    explanation = torch._dynamo.explain(fn_with_multiple_breaks)(x)
    assert explanation.graph_break_count == 2
    assert explanation.graph_count == 3

    with patch("sys.stdout", new_callable=io.StringIO):
        check_functions_are_equivalent(fn_with_multiple_breaks, device, [x])


def test_no_graph_breaks_with_supported_operations(device: str):
    def well_supported_fn(x, y):
        # Only use operations that should be well supported
        z = x + y
        z = torch.sin(z)
        z = torch.cos(z)
        z = z * 2
        z = torch.abs(z)
        return z

    x = torch.randn(3, 4)
    y = torch.randn(3, 4)
    explanation = torch._dynamo.explain(well_supported_fn)(x, y)
    assert explanation.graph_break_count == 0
    assert explanation.graph_count == 1
    check_functions_are_equivalent(well_supported_fn, device, [x, y])


class max_backendCallCount:
    def __init__(self, compiler):
        self.call_count = 0
        self.compiler = compiler

    def __call__(self, *args, **kwargs):
        self.call_count += 1
        return self.compiler(*args, **kwargs)


def test_dynamic_shapes(device: str):
    """Testing the behavior with mark_dynamic()."""

    def fn(x, y):
        return x + y

    counter = max_backendCallCount(max_backend)
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
        # Ensure only one instance of the max_backend is created
    assert counter.call_count == 1


def test_recompilation(device: str):
    """Testing the behavior without mark_dynamic()."""

    def fn(x, y):
        return x + y

    counter = max_backendCallCount(max_backend)
    fn_compiled = torch.compile(backend=counter)(fn)

    a = torch.randn(20, 2).to(device)
    b = torch.randn(2).to(device)

    check_functions_are_equivalent(fn, None, [a, b], fn_compiled)

    a = torch.randn(10, 2).to(device)
    b = torch.randn(2).to(device)

    check_functions_are_equivalent(fn, None, [a, b], fn_compiled)
    # Ensure a second instance of the max_backend is created
    assert counter.call_count == 2

    # TODO: Make it work if called with more shapes (dynamo doesn't recompile)


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


@pytest.mark.xfail("FIXME if you can")
def test_error_message_exception_in_op_decomposed(monkeypatch):
    def not_working_add(x, y):
        raise RuntimeError("Ho no crash!")

    monkeypatch.setitem(MAPPING_TORCH_ATEN_TO_MAX, aten.add, not_working_add)

    def fn(x):
        return torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")

    with pytest.raises(RuntimeError) as exc_info:
        torch.compile(backend=max_backend)(fn)(torch.randn(2, 3, 4, 4))

    assert "return torch.nn.functional.interpolate(x, scale_f" in str(exc_info.value)
    assert "Ho no crash!" in str(exc_info.value)
    assert "torch._ops.aten.aten::add" in str(exc_info.value)
    assert "https://github.com/gabrieldemarmiesse/torch-max-backend/issues" in str(
        exc_info.value
    )
    assert "not_working_add" in str(exc_info.value)


def test_error_message_op_not_supported(monkeypatch):
    monkeypatch.delitem(MAPPING_TORCH_ATEN_TO_MAX, aten.add)

    def fn(x, y):
        return x + y

    with pytest.raises(BackendCompilerFailed) as exc_info:
        torch.compile(backend=max_backend)(fn)(torch.randn(2, 3), torch.randn(2, 3))

    assert "return x + y" in str(exc_info.value)
    assert "torch._ops.aten.aten::add" in str(exc_info.value)
    assert "https://github.com/gabrieldemarmiesse/torch-max-backend/issues" in str(
        exc_info.value
    )
    assert "is not supported" in str(exc_info.value)


@pytest.mark.xfail("FIXME if you can")
def test_error_message_op_not_supported_decomposed(monkeypatch):
    monkeypatch.delitem(MAPPING_TORCH_ATEN_TO_MAX, aten.add)

    def fn(x):
        return torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")

    with pytest.raises(BackendCompilerFailed) as exc_info:
        torch.compile(backend=max_backend)(fn)(torch.randn(2, 3, 4, 4))

    assert "return torch.nn.functional.interpolate(x, scale_f" in str(exc_info.value)
    assert "torch._ops.aten.aten::add" in str(exc_info.value)
    assert "https://github.com/gabrieldemarmiesse/torch-max-backend/issues" in str(
        exc_info.value
    )
    assert "is not supported" in str(exc_info.value)
