"""Host-only regressions for eager autograd temporary lifetimes."""

import weakref

from torch_mojo_backend.eager_kernels import aten_fast


class _TrackedTensor:
    def __init__(self, name: str, dtype: object, device: object, events: list[str]):
        self.name = name
        self._dtype = dtype
        self._shape = (2, 3)
        self._device = device
        self._numel = 6
        self._events = events

    def __del__(self):
        self._events.append(f"released:{self.name}")


def test_log_softmax_backward_uses_addcmul_and_releases_inputs(monkeypatch):
    """Avoid a correction buffer and release fused inputs without a sync."""
    dtype = object()
    device = object()
    events: list[str] = []
    references: dict[str, weakref.ReferenceType[_TrackedTensor]] = {}
    output = _TrackedTensor("output", dtype, device, events)
    grad_output = _TrackedTensor("grad_output", dtype, device, events)
    grad_input = _TrackedTensor("grad_input", dtype, device, events)

    def make_tracked(name: str) -> _TrackedTensor:
        tensor = _TrackedTensor(name, dtype, device, events)
        references[name] = weakref.ref(tensor)
        return tensor

    def addcmul(actual_grad, probabilities, summed, *, value):
        assert actual_grad is grad_output
        assert probabilities is references["probabilities"]()
        assert summed is references["summed"]()
        assert value == -1.0
        events.append("addcmul")
        return grad_input

    def unexpected_separate_op(*_args, **_kwargs):
        raise AssertionError("log-softmax backward must not materialize correction")

    monkeypatch.setattr(
        aten_fast,
        "_t",
        lambda tensor: tensor if isinstance(tensor, _TrackedTensor) else None,
    )
    monkeypatch.setattr(aten_fast, "_FLOAT_DTYPES", (dtype,))
    monkeypatch.setattr(aten_fast, "_torch_dtype_to_max", lambda actual: actual)
    monkeypatch.setattr(
        aten_fast, "fast_aten_sum", lambda *_args, **_kwargs: make_tracked("summed")
    )
    monkeypatch.setattr(
        aten_fast,
        "fast_aten_exp",
        lambda *_args, **_kwargs: make_tracked("probabilities"),
    )
    monkeypatch.setattr(aten_fast, "fast_aten_addcmul", addcmul)
    monkeypatch.setattr(aten_fast, "fast_aten_mul", unexpected_separate_op)
    monkeypatch.setattr(aten_fast, "fast_aten_sub", unexpected_separate_op)
    monkeypatch.setattr(aten_fast, "_cast_tensor", unexpected_separate_op)

    result = aten_fast.fast_aten__log_softmax_backward_data(
        grad_output, output, 1, dtype
    )

    assert result is grad_input
    assert events.index("addcmul") < events.index("released:probabilities")
    assert events.index("addcmul") < events.index("released:summed")
    assert references["probabilities"]() is None
    assert references["summed"]() is None
