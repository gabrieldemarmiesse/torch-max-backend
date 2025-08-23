from torch_max_backend.max_device import MaxDeviceBackend
import torch


def test_max_device_basic_arange_sqrt():
    backend = MaxDeviceBackend()
    backend.register()

    a = torch.arange(4, device="max_gpu", dtype=torch.float32)

    sqrt_result = torch.sqrt(a)

    result_cpu = sqrt_result.to("cpu")
    assert torch.allclose(
        result_cpu, torch.tensor([0.0, 1.0, 1.4142, 1.7320]), atol=1e-4
    )

    b = torch.arange(4, device="max_gpu", dtype=torch.float32)
    chained = sqrt_result + b
    chained_cpu = chained.to("cpu")
    assert torch.allclose(
        chained_cpu, torch.tensor([0.0, 2.0, 3.4142, 4.7320]), atol=1e-4
    )
