"""PrivateUse1 device-module, transfer, and RNG runtime contracts."""

import gc
import weakref
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from torch_mojo_backend import register_mojo_devices
from torch_mojo_backend.mojo_device import torch_mojo_tensor as mojo_tensor_module
from torch_mojo_backend.mojo_device.torch_mojo_device_module import (
    _reserve_philox_state,
)
from torch_mojo_backend.mojo_device.torch_mojo_tensor import (
    _PENDING_H2D,
    find_equivalent_max_device,
    get_ordered_accelerators,
)

pytestmark = pytest.mark.xdist_group(name="group1")


@pytest.fixture(autouse=True)
def setup_mojo_device():
    register_mojo_devices()


def test_mojo_is_the_default_torch_accelerator():
    assert torch.accelerator.current_accelerator(check_available=True) == torch.device(
        "mojo"
    )


def test_torch_accelerator_synchronize_uses_mojo_device_module(monkeypatch):
    calls = []
    original_synchronize = torch.mojo.synchronize
    original_device = torch.mojo.current_device()

    def recording_synchronize(device=None):
        calls.append(device)
        return original_synchronize(device)

    monkeypatch.setattr(torch.mojo, "synchronize", recording_synchronize)
    try:
        torch.accelerator.synchronize()
        torch.accelerator.synchronize("mojo")
        torch.accelerator.synchronize(0)
    finally:
        torch.mojo.set_device(original_device)

    assert calls == [original_device, original_device, 0]
    with pytest.raises(ValueError, match="doesn't match the current accelerator mojo"):
        torch.accelerator.synchronize("cpu")


def test_indexless_mojo_device_uses_current_device():
    accelerators = get_ordered_accelerators()
    if len(accelerators) < 2:
        pytest.skip("requires two Mojo devices, including the MAX CPU device")
    original_index = torch.mojo.current_device()
    alternate_index = (original_index + 1) % len(accelerators)
    try:
        torch.mojo.set_device(alternate_index)
        assert (
            find_equivalent_max_device(torch.device("mojo"))
            == accelerators[alternate_index]
        )
        assert torch.empty(1, device="mojo")._device == accelerators[alternate_index]
    finally:
        torch.mojo.set_device(original_index)


def test_non_blocking_cpu_to_mojo_transfers(mojo_device):
    source = torch.arange(4096, dtype=torch.float32)
    via_to = source.to(mojo_device, non_blocking=True)
    via_copy = torch.empty_like(via_to)
    via_copy.copy_(source, non_blocking=True)

    torch.accelerator.synchronize(mojo_device)
    torch.testing.assert_close(via_to.cpu(), source)
    torch.testing.assert_close(via_copy.cpu(), source)


def test_non_blocking_cpu_source_lifetime(mojo_device):
    source = torch.arange(1 << 20, dtype=torch.int32)
    expected = source.clone()
    uploaded = source.to(mojo_device, non_blocking=True)
    del source
    for _ in range(8):
        torch.empty_like(expected).fill_(-1)

    torch.accelerator.synchronize(mojo_device)
    assert uploaded._device not in _PENDING_H2D
    torch.testing.assert_close(uploaded.cpu(), expected)


def test_non_blocking_mojo_to_cpu_transfer(mojo_device):
    expected = torch.arange(1 << 16, dtype=torch.float32)
    source = expected.to(mojo_device)
    downloaded = source.to("cpu", non_blocking=True)

    torch.accelerator.synchronize(mojo_device)
    torch.testing.assert_close(downloaded, expected)
    assert source._device not in mojo_tensor_module._PENDING_D2H


@pytest.mark.parametrize("direction", ["h2d", "d2h"])
def test_transfer_query_failure_retains_new_dma_owner(direction):
    class QueryErrorEvent:
        def is_ready(self):
            raise RuntimeError("injected event query failure")

    class CurrentEvent:
        def is_ready(self):
            return False

    class FakeStream:
        def __init__(self):
            self.current = CurrentEvent()

        def record_event(self):
            return self.current

    class FakeDevice:
        def __init__(self):
            self.default_stream = FakeStream()

    device = FakeDevice()
    old_owner = object()
    new_owner = object()
    if direction == "h2d":
        pending = mojo_tensor_module._PENDING_H2D
        lock = mojo_tensor_module._PENDING_H2D_LOCK

        def record():
            mojo_tensor_module._record_h2d_source(device, new_owner, True)

    else:
        pending = mojo_tensor_module._PENDING_D2H
        lock = mojo_tensor_module._PENDING_D2H_LOCK

        def record():
            mojo_tensor_module._record_d2h_owner(device, new_owner)

    with lock:
        pending[device] = mojo_tensor_module.deque([(QueryErrorEvent(), old_owner)])
    try:
        with pytest.raises(RuntimeError, match="injected event query failure"):
            record()
        assert [entry[1] for entry in pending[device]] == [old_owner, new_owner]
    finally:
        with lock:
            pending.pop(device, None)


@pytest.mark.parametrize("direction", ["h2d", "d2h"])
def test_transfer_record_and_sync_failure_retains_owner(direction):
    class Owner:
        pass

    class FailingStream:
        def record_event(self):
            raise RuntimeError("injected event record failure")

        def synchronize(self):
            raise RuntimeError("injected recovery sync failure")

    class FakeDevice:
        def __init__(self):
            self.default_stream = FailingStream()

    device = FakeDevice()
    owner = Owner()
    owner_ref = weakref.ref(owner)
    try:
        with pytest.raises(RuntimeError, match="injected recovery sync failure"):
            if direction == "h2d":
                mojo_tensor_module._record_h2d_source(device, owner, True)
            else:
                mojo_tensor_module._record_d2h_owner(device, owner)
        del owner
        gc.collect()
        assert owner_ref() is not None
    finally:
        with mojo_tensor_module._FAILED_TRANSFER_OWNERS_LOCK:
            mojo_tensor_module._FAILED_TRANSFER_OWNERS.pop(device, None)


def test_mojo_rng_state_exact_replay_and_high_bit_seed(mojo_device):
    device = torch.device(mojo_device)
    seed = (1 << 63) + 0x12345
    torch.mojo.manual_seed_all(seed)
    initial = torch.mojo.get_rng_state(device)
    assert initial.dtype == torch.uint8
    assert initial.shape == (16,)

    assert _reserve_philox_state(device, 17) == (seed, 0)
    advanced = torch.mojo.get_rng_state(device)
    torch.mojo.set_rng_state(initial, device)
    assert _reserve_philox_state(device, 17) == (seed, 0)
    torch.testing.assert_close(torch.mojo.get_rng_state(device), advanced)


def test_mojo_rng_state_is_per_device():
    if torch.mojo.device_count() < 2:
        pytest.skip("requires two Mojo devices")
    first = torch.device("mojo:0")
    second = torch.device("mojo:1")
    torch.mojo.manual_seed_all(20260718)
    second_before = torch.mojo.get_rng_state(second)
    assert _reserve_philox_state(first, 257) == (20260718, 0)
    torch.testing.assert_close(torch.mojo.get_rng_state(second), second_before)


def test_mojo_rng_state_rejects_malformed_state(mojo_device):
    device = torch.device(mojo_device)
    with pytest.raises(ValueError, match="16-element uint8"):
        torch.mojo.set_rng_state(torch.zeros(16, dtype=torch.int64), device)
    with pytest.raises(ValueError, match="16-element uint8"):
        torch.mojo.set_rng_state(torch.zeros(15, dtype=torch.uint8), device)


def test_mojo_rng_reservations_are_atomic_and_reject_wrap(mojo_device):
    device = torch.device(mojo_device)
    torch.mojo.manual_seed_all(20260718)
    reservations = 256
    with ThreadPoolExecutor(max_workers=16) as pool:
        bases = list(
            pool.map(lambda _: _reserve_philox_state(device, 1)[1], range(reservations))
        )
    assert sorted(bases) == list(range(reservations))

    seed = (1 << 63) + 7
    counter = (1 << 64) - 1
    encoded = seed.to_bytes(8, "little") + counter.to_bytes(8, "little")
    torch.mojo.set_rng_state(torch.tensor(list(encoded), dtype=torch.uint8), device)
    before = torch.mojo.get_rng_state(device)
    with pytest.raises(OverflowError, match="would wrap"):
        _reserve_philox_state(device, 1)
    torch.testing.assert_close(torch.mojo.get_rng_state(device), before)


def test_torch_fork_rng_restores_mojo_counter(mojo_device):
    device = torch.device(mojo_device)
    index = torch.mojo.current_device() if device.index is None else device.index
    torch.mojo.manual_seed_all(91)
    before = torch.mojo.get_rng_state(device)
    with torch.random.fork_rng(devices=[index], device_type="mojo"):
        assert _reserve_philox_state(device, 33) == (91, 0)
    torch.testing.assert_close(torch.mojo.get_rng_state(device), before)
