from torch.utils._python_dispatch import TorchDispatchMode


class LoggingMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        print(f"Aten function called: {func}")
        return func(*args, **kwargs or {})


def log_aten_calls():
    LoggingMode().__enter__()
