import torch


def function_equivalent_on_both_devices(func, devices, *args, **kwargs):
    out1 = func(*args, device=devices[0], **kwargs)
    out2 = func(*args, device=devices[1], **kwargs)
    if isinstance(out1, list | tuple):
        assert type(out1) == type(out2)
    else:
        assert isinstance(out1, torch.Tensor)
        assert isinstance(out2, torch.Tensor)
        out1 = [out1]
        out2 = [out2]

    # We transfer on device 1
    out2 = [o.to(devices[0]) for o in out2]

    for i, (o1, o2) in enumerate(zip(out1, out2)):
        assert o1.device == o2.device, f"Issue with output {i}"
        assert o1.shape == o2.shape, f"Issue with output {i}"
        assert o1.dtype == o2.dtype, f"Issue with output {i}"
        assert torch.allclose(o1, o2, rtol=1e-4, atol=1e-4), f"Issue with output {i}"


def test_max_device_basic(equivalent_devices):
    def do_sqrt(device):
        a = torch.arange(4, device=device, dtype=torch.float32)
        return torch.sqrt(a)

    function_equivalent_on_both_devices(do_sqrt, equivalent_devices)


def test_max_device_basic_arange_sqrt(equivalent_devices):
    for device in equivalent_devices:
        a = torch.arange(4, device=device, dtype=torch.float32)

        sqrt_result = torch.sqrt(a)

        result_cpu = sqrt_result.to("cpu")
        assert torch.allclose(
            result_cpu, torch.tensor([0.0, 1.0, 1.4142, 1.7320]), atol=1e-4
        )

        b = torch.arange(4, device=device, dtype=torch.float32)
        chained = sqrt_result + b
        chained_cpu = chained.to("cpu")
        assert torch.allclose(
            chained_cpu, torch.tensor([0.0, 2.0, 3.4142, 4.7320]), atol=1e-4
        )
