import inspect
from collections.abc import Callable
from dataclasses import dataclass

import torch

from torch_max_backend import max_backend


class CallChecker:
    """Asserts that at least one of the registered implementations ran.

    Ops covered by the max_device fast eager path have two implementations:
    the graph one in `aten_functions` (used by the torch.compile backend)
    and the Mojo-kernel one in `aten_fast` (used by max_device eager mode).
    A test registers the `aten_functions` twin; `register` automatically
    also accepts the matching `aten_fast.fast_<name>` twin, so the same
    test passes whether the op routed to the graph path (compile) or the
    fast path (eager) — no per-test bookkeeping needed.
    """

    def __init__(self):
        self._functions_to_check = None
        self._counts_before_starting_to_check = None

    @staticmethod
    def _fast_twin(func: Callable) -> Callable | None:
        """The aten_fast.fast_<name> counterpart of an aten_functions twin,
        if one exists and is instrumented with a call counter."""
        name = getattr(func, "__name__", "")
        if not name.startswith("aten"):
            return None
        try:
            from torch_max_backend.eager_kernels import aten_fast
        except Exception:
            return None
        twin = getattr(aten_fast, f"fast_{name}", None)
        if twin is not None and hasattr(twin, "call_count"):
            return twin
        return None

    def register(self, *funcs: Callable):
        expanded: list[Callable] = []
        for func in funcs:
            expanded.append(func)
            twin = self._fast_twin(func)
            if twin is not None and twin not in expanded:
                expanded.append(twin)
        self._functions_to_check = tuple(expanded)
        self._counts_before_starting_to_check = [
            f.call_count for f in self._functions_to_check
        ]

    def check_was_called(self):
        if self._functions_to_check is None:
            raise ValueError(
                "No function to check was set, call call_checker.register first"
            )
        if not any(
            func.call_count > count_before
            for func, count_before in zip(
                self._functions_to_check, self._counts_before_starting_to_check
            )
        ):
            names = ", ".join(f.__name__ for f in self._functions_to_check)
            raise AssertionError(
                f"Expected one of [{names}] to be called at least once in the test, but none was"
            )


def check_functions_are_equivalent(
    fn: Callable,
    device: str | None,
    inputs: list[torch.Tensor],
    fn_compiled: Callable | None = None,
    rtol=None,
    atol=None,
):
    fn_compiled = fn_compiled or torch.compile(backend=max_backend)(fn)
    if device is not None:
        inputs = [input_tensor.to(device) for input_tensor in inputs]

    # We use the compiled first because compiled never changes
    # the input tensors, while the original function might.
    output_compiled = fn_compiled(*inputs)
    output_original = fn(*inputs)

    assert type(output_original) == type(output_compiled)

    if isinstance(output_original, torch.Tensor):
        output_original = [output_original]
        output_compiled = [output_compiled]

    for i, (original, compiled) in enumerate(zip(output_original, output_compiled)):
        assert original.shape == compiled.shape, f"Issue with output {i}"
        assert original.device == compiled.device, f"Issue with output {i}"
        assert original.dtype == compiled.dtype, f"Issue with output {i}"
        torch.testing.assert_close(original, compiled, rtol=rtol, atol=atol)


@dataclass
class Conf:
    device: str
    compile: bool

    def __str__(self) -> str:
        if self.compile:
            word = "compiled"
        else:
            word = "eager"
        return f"{self.device}, {word}"


def to_device(tensors: list[torch.Tensor], device: str) -> list[torch.Tensor]:
    return [torch.clone(tensor).to(device) for tensor in tensors]


def check_outputs(
    fn: Callable, conf: Conf, inputs: list[torch.Tensor], *, rtol=None, atol=None
):
    # We compare to eager cpu execution
    # We first check if the function has a device argument
    has_device_arg = "device" in inspect.signature(fn).parameters
    inputs_cpu = to_device(inputs, "cpu")
    if has_device_arg:
        outputs_eager_cpu = fn(*inputs_cpu, device="cpu")
    else:
        outputs_eager_cpu = fn(*inputs_cpu)

    if conf.compile:
        fn_to_run = torch.compile(fn, backend=max_backend)
    else:
        fn_to_run = fn

    inputs_on_device = to_device(inputs, conf.device)
    if has_device_arg:
        outputs_conf = fn_to_run(*inputs_on_device, device=conf.device)
    else:
        outputs_conf = fn_to_run(*inputs_on_device)

    # Now we compare outputs
    if isinstance(outputs_eager_cpu, torch.Tensor):
        outputs_eager_cpu = [outputs_eager_cpu]
        outputs_conf = [outputs_conf]

    for i, (output_eager_cpu, output_conf) in enumerate(
        zip(outputs_eager_cpu, outputs_conf)
    ):
        expected_device = torch.device(conf.device)
        if not (output_conf.device == expected_device):
            raise AssertionError(
                f"Issue with output {i}, expected device {repr(expected_device)} but got {repr(output_conf.device)}"
            )
        assert output_eager_cpu.shape == output_conf.shape, f"Issue with output {i}"
        assert output_eager_cpu.dtype == output_conf.dtype, f"Issue with output {i}"
        # move to cpu for comparison
        output_conf_cpu = output_conf.to("cpu")
        torch.testing.assert_close(
            output_eager_cpu, output_conf_cpu, rtol=rtol, atol=atol
        )
