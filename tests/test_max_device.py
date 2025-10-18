"""Unit tests for basic max_device functionality"""

import pytest
import torch

from torch_max_backend import TorchMaxTensor, max_backend, register_max_devices
from torch_max_backend.max_device.torch_max_tensor import (
    find_equivalent_max_device,
    get_ordered_accelerators,
)

pytestmark = pytest.mark.xdist_group(name="group1")


@pytest.fixture(autouse=True)
def setup_max_device():
    """Setup max_device for all tests"""
    register_max_devices()


def test_tensor_to_max_device(max_device):
    """Test converting regular tensor to max_device"""
    # Create CPU tensor
    cpu_tensor = torch.tensor([1.0, 2.0, 3.0])

    # Convert to max_device
    max_tensor = cpu_tensor.to(max_device)

    # Check type and properties
    assert isinstance(max_tensor, TorchMaxTensor)
    assert max_tensor.shape == (3,)
    assert max_tensor.dtype == torch.float32


def test_max_tensor_to_cpu(max_device):
    """Test converting MaxTensor back to CPU"""
    # Create tensor on max_device
    cpu_tensor = torch.tensor([1.0, 2.0, 3.0])
    max_tensor = cpu_tensor.to(max_device)

    # Convert back to CPU
    result = max_tensor.to("cpu")

    # Check result
    assert isinstance(result, torch.Tensor)
    torch.testing.assert_close(result, cpu_tensor)


def test_factory_arange(max_device):
    """Test torch.arange with max_device"""
    tensor = torch.arange(5, device=max_device)

    assert isinstance(tensor, TorchMaxTensor)
    assert tensor.shape == (5,)

    # Convert to CPU to check values
    cpu_result = tensor.to("cpu")
    expected = torch.arange(5)
    torch.testing.assert_close(cpu_result, expected)


@pytest.mark.xfail(reason="Fixme")
def test_factory_rand(max_device):
    """Test torch.rand with max_device"""
    tensor = torch.rand(3, 4, device=max_device)

    assert isinstance(tensor, TorchMaxTensor)
    assert tensor.shape == (3, 4)

    # Check that values are in [0, 1] range when converted to CPU
    cpu_result = tensor.to("cpu")
    assert torch.all(cpu_result >= 0)
    assert torch.all(cpu_result <= 1)


@pytest.mark.xfail(reason="Fixme")
def test_factory_empty(max_device):
    """Test torch.empty with max_device"""
    tensor = torch.empty(2, 3, device=max_device)

    assert isinstance(tensor, TorchMaxTensor)
    assert tensor.shape == (2, 3)


def test_device_string_variations():
    """Test different max_device string formats"""
    # Basic max_device
    t1 = torch.tensor([1.0]).to("max_device")
    assert isinstance(t1, TorchMaxTensor)

    # With index (should also work)
    t2 = torch.tensor([1.0]).to("max_device:0")
    assert isinstance(t2, TorchMaxTensor)


@pytest.mark.xfail(reason="TODO: add pretty repr and str")
def test_tensor_properties(max_device):
    """Test that MaxTensor preserves tensor properties"""
    original = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    max_tensor = original.to(max_device)

    assert max_tensor.shape == (2, 2)
    assert max_tensor.dtype == torch.float64
    assert max_tensor.device == torch.device(max_device)

    # Test repr
    repr_str = repr(max_tensor)
    assert max_device in repr_str
    assert "size=(2, 2)" in repr_str


def test_round_trip_conversion(max_device):
    """Test CPU -> max_device -> CPU round trip"""
    original = torch.tensor([1.0, 2.0, 3.0, 4.0])

    # Round trip
    max_tensor = original.to(max_device)
    result = max_tensor.to("cpu")

    # Should be equal
    torch.testing.assert_close(result, original)


def test_dtype_preservation(max_device):
    """Test that dtypes are preserved during conversion"""
    for dtype in [torch.float32, torch.float64, torch.int32, torch.int64]:
        original = torch.tensor([1, 2, 3], dtype=dtype)
        max_tensor = original.to(max_device)
        result = max_tensor.to("cpu")

        assert result.dtype == dtype
        torch.testing.assert_close(result, original)


def test_multiple_conversions():
    """Test multiple to() calls don't cause issues"""
    tensor = torch.tensor([1.0, 2.0])

    # Multiple conversions should work
    max1 = tensor.to("max_device")
    max2 = max1.to("max_device")  # Should return same tensor
    cpu1 = max2.to("cpu")
    cpu2 = cpu1.to("cpu")  # Should work normally

    # Test operations step by step for clearer errors
    diff = max1 - max2
    squared = diff**2
    summed = torch.sum(squared)
    cpu_result = summed.to("cpu")
    result_value = cpu_result.item()
    assert result_value == 0

    torch.testing.assert_close(cpu2, tensor)


def test_device_ordering():
    """Test that device ordering follows GPU first, CPU last convention"""
    ordered_accelerators = get_ordered_accelerators()

    # Check that we have both GPU and CPU
    gpu_devices = [acc for acc in ordered_accelerators if acc.label == "gpu"]
    cpu_devices = [acc for acc in ordered_accelerators if acc.label == "cpu"]

    # Should have at least one device
    assert len(ordered_accelerators) > 0

    # If we have both GPU and CPU, GPU should come first
    if gpu_devices and cpu_devices:
        # First device should be GPU
        assert ordered_accelerators[0].label == "gpu"
        # Last device should be CPU
        assert ordered_accelerators[-1].label == "cpu"


def test_device_mapping_consistency():
    """Test that CPU maps to highest index and GPU to lower indices"""

    ordered_accelerators = get_ordered_accelerators()

    if len(ordered_accelerators) > 1:
        # Test CPU device mapping
        cpu_device = torch.device("cpu")
        max_cpu = find_equivalent_max_device(cpu_device)

        # CPU should map to a CPU accelerator
        assert max_cpu.label == "cpu"

        # Find CPU in ordered list - should be last if we have multiple devices
        cpu_indices = [
            i for i, acc in enumerate(ordered_accelerators) if acc.label == "cpu"
        ]
        if cpu_indices:
            # If CPU exists, it should be at the highest index
            assert cpu_indices[-1] == len(ordered_accelerators) - 1


def test_gpu_first_cpu_last_convention():
    """Test the specific convention: device 0 = first GPU, highest index = CPU"""

    ordered_accelerators = get_ordered_accelerators()

    # If we have both GPU and CPU
    gpu_count = sum(1 for acc in ordered_accelerators if acc.label == "gpu")
    cpu_count = sum(1 for acc in ordered_accelerators if acc.label == "cpu")

    if gpu_count > 0 and cpu_count > 0:
        # First device should be GPU
        assert ordered_accelerators[0].label == "gpu"

        # Last device should be CPU
        assert ordered_accelerators[-1].label == "cpu"

        # Test that max_device (index 0) goes to GPU
        t_gpu = torch.tensor([1.0]).to("max_device")
        assert isinstance(t_gpu, TorchMaxTensor)

        # Test that highest index goes to CPU
        cpu_index = len(ordered_accelerators) - 1
        t_cpu = torch.tensor([1.0]).to(f"max_device:{cpu_index}")
        assert isinstance(t_cpu, TorchMaxTensor)


# Original tests from the existing file
def function_equivalent_on_both_devices(func, device, *args, **kwargs):
    out1 = func(*args, device=device, **kwargs)
    out2 = func(*args, device="cpu", **kwargs)
    if isinstance(out1, list | tuple):
        assert type(out1) == type(out2)
    else:
        assert isinstance(out1, torch.Tensor)
        assert isinstance(out2, torch.Tensor)
        out1 = [out1]
        out2 = [out2]

    # We transfer on device 1
    out1 = [o.to("cpu") for o in out1]

    for i, (o1, o2) in enumerate(zip(out1, out2)):
        assert o1.device == o2.device, f"Issue with output {i}"
        assert o1.shape == o2.shape, f"Issue with output {i}"
        assert o1.dtype == o2.dtype, f"Issue with output {i}"
        assert torch.allclose(o1, o2, rtol=1e-4, atol=1e-4), f"Issue with output {i}"


def test_max_device_basic(max_device):
    def do_sqrt(device):
        a = torch.arange(4, device=device, dtype=torch.float32)
        return torch.sqrt(a)

    function_equivalent_on_both_devices(do_sqrt, max_device)


def test_max_device_basic_arange_sqrt(max_device):
    a = torch.arange(4, device=max_device, dtype=torch.float32)

    sqrt_result = torch.sqrt(a)

    result_cpu = sqrt_result.to("cpu")
    assert torch.allclose(
        result_cpu, torch.tensor([0.0, 1.0, 1.4142, 1.7320]), atol=1e-4
    )

    b = torch.arange(4, device=max_device, dtype=torch.float32)
    chained = sqrt_result + b
    chained_cpu = chained.to("cpu")
    assert torch.allclose(
        chained_cpu, torch.tensor([0.0, 2.0, 3.4142, 4.7320]), atol=1e-4
    )


def test_device_creation(max_device):
    torch_device = torch.device(max_device)
    arr = torch.arange(4, device=torch_device, dtype=torch.float32)
    arr_cpu = arr.to("cpu")

    assert torch.allclose(arr_cpu, torch.tensor([0.0, 1.0, 2.0, 3.0]), atol=1e-4)


def test_device_basic_full(max_device):
    def do_full(device):
        a = torch.full((2, 3), 7.0, device=device, dtype=torch.float32)
        return a

    function_equivalent_on_both_devices(do_full, max_device)


def test_convolution_2d(max_device):
    input_tensor_cpu = torch.randn(1, 3, 32, 32, device="cpu")
    weight_cpu = torch.randn(6, 3, 5, 5, device="cpu")
    bias_cpu = torch.randn(6, device="cpu")

    def do_convolution(device):
        input_tensor = input_tensor_cpu.to(device)
        weight = weight_cpu.to(device)
        bias = bias_cpu.to(device)
        return torch.nn.functional.conv2d(
            input_tensor, weight, bias=bias, stride=1, padding=2
        )

    function_equivalent_on_both_devices(do_convolution, max_device)


def test_simple_module(max_device):
    linear = torch.nn.Linear(4, 8)

    def run_module(device):
        my_linear = linear.to(device)
        return my_linear.weight

    function_equivalent_on_both_devices(run_module, max_device)


@pytest.mark.xfail(reason="Fixme")
def test_compile_with_max_device(max_device):
    @torch.compile(backend=max_backend)
    def do_sqrt(device):
        a = torch.arange(4, device=device, dtype=torch.float32)
        return torch.sqrt(a)

    function_equivalent_on_both_devices(do_sqrt, max_device)
