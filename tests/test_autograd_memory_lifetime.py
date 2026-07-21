"""Host-only regressions for eager autograd temporary lifetimes."""

from types import SimpleNamespace
import weakref

from torch_mojo_backend.mojo_device import mojo_device_autograd as autograd


class _TrackedTensor:
    def __init__(self, name: str, dtype: object, events: list[str]):
        self.name = name
        self._dtype = dtype
        self._shape = (2, 3)
        self._device = object()
        self._events = events

    def __del__(self):
        self._events.append(f"released:{self.name}")


def test_log_softmax_backward_uses_addcmul_and_releases_inputs(monkeypatch):
    """Avoid a correction buffer and release fused inputs without a sync."""
    dtype = object()
    events: list[str] = []
    references: dict[str, weakref.ReferenceType[_TrackedTensor]] = {}
    output = _TrackedTensor("output", dtype, events)
    grad_output = _TrackedTensor("grad_output", dtype, events)
    grad_input = _TrackedTensor("grad_input", dtype, events)

    def make_tracked(name: str) -> _TrackedTensor:
        tensor = _TrackedTensor(name, dtype, events)
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

    fake_fast = SimpleNamespace(
        NOT_HANDLED=object(),
        fast_aten_sum=lambda *_args, **_kwargs: make_tracked("summed"),
        fast_aten_exp=lambda *_args, **_kwargs: make_tracked("probabilities"),
        fast_aten_addcmul=addcmul,
        fast_aten_mul=unexpected_separate_op,
        fast_aten_sub=unexpected_separate_op,
        _cast_tensor=lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(autograd, "_fast", lambda: fake_fast)
    monkeypatch.setattr(autograd, "_restore_saved_mojo_tensors", lambda _ctx: (output,))

    ctx = SimpleNamespace(dim=1, input_dtype=dtype)
    result = autograd._LogSoftmaxAutograd.backward(ctx, grad_output)

    assert result == (grad_input, None, None)
    assert events.index("addcmul") < events.index("released:probabilities")
    assert events.index("addcmul") < events.index("released:summed")
    assert references["probabilities"]() is None
    assert references["summed"]() is None
