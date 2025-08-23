from max.driver import Accelerator, accelerator_count, CPU
import warnings
from max.driver import Device


def get_accelerators() -> list[Device]:
    result = []
    if accelerator_count() > 0:
        for i in range(accelerator_count()):
            try:
                result.append(Accelerator(i))
            except ValueError as e:
                warnings.warn(f"Failed to create accelerator {i}. {e}")
    # This way, people can do torch.device("max_device:0") even if there is
    # no accelerator and get gpu or cpu automatically.
    result.append(CPU())
    return result
