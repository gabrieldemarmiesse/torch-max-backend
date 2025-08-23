from torch_max_backend import get_accelerators
import torch
import numpy as np


accelerators = list(get_accelerators())
_current_device = 0


def _is_in_bad_fork():
    return False


def manual_seed_all(seed):
    np.random.seed(seed)


def device_count():
    return len(accelerators)


def get_rng_state(device=None):
    return torch.tensor(np.random.get_state()[1])


def set_rng_state(new_state, device=None):
    if isinstance(new_state, torch.Tensor):
        new_state = new_state.cpu().numpy()
    np_state = ("MT19937", new_state, 624, 0, 0.0)
    np.random.set_state(np_state)


def is_available():
    # Always true as there is at least the CPU
    return True


def current_device():
    return 0  # TODO change


def set_device(device_idx: int):
    global _current_device
    if device_idx < 0 or device_idx >= device_count():
        raise ValueError(f"Invalid device index {device_idx}")
    _current_device = device_idx


def get_amp_supported_dtype():
    return [torch.float16, torch.bfloat16]  # TODO change


# TODO: necessary?
def max_gpu(self):
    print("hello")
