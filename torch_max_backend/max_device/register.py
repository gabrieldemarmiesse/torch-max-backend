import torch

from .max_device_aten_ops import _aten_ops_registry

_registered = False


def _declare_mojo_tensor_as_plain_tensor():
    """Add TorchMojoTensor to torch's HANDLED_TYPES allowlists.

    torch.compile treats TorchMojoTensor as a plain tensor everywhere
    (it has no __torch_dispatch__/__torch_function__ behavior; all backend
    logic lives behind the PrivateUse1 dispatch key), but a couple of
    exact-type allowlists don't know that:

    - aot_autograd's first-invocation `_AnalyzeCustomOpInputOutputMode`
      returns NotImplemented for unknown tensor types, which makes every
      dispatched op fail on mojo tensors under eager-executing compile
      backends (e.g. "aot_eager"). runtime_wrappers imported the tuple by
      value, so patch both bindings.
    - FakeTensorMode returns NotImplemented for ops whose args include an
      unrecognized tensor subclass, to give that subclass's
      __torch_dispatch__ a chance to run. TorchMojoTensor has none, so
      nothing handles the op and tracing fails with "Multiple dispatch
      failed" — hit when dynamo lifts a mojo tensor constant created
      mid-trace (e.g. `torch.tensor([], device="mojo")` in HF generate).
    """
    import torch._functorch._aot_autograd.runtime_wrappers as runtime_wrappers
    import torch._subclasses.fake_tensor as fake_tensor_module
    import torch.fx.experimental.proxy_tensor as proxy_tensor

    from .torch_max_tensor import TorchMojoTensor

    if TorchMojoTensor not in proxy_tensor.HANDLED_TYPES:
        proxy_tensor.HANDLED_TYPES = (*proxy_tensor.HANDLED_TYPES, TorchMojoTensor)
    runtime_wrappers.HANDLED_TYPES = proxy_tensor.HANDLED_TYPES

    original_check = fake_tensor_module._check_for_subclass_arg

    def check_for_subclass_arg_except_mojo(x):
        return original_check(x) and not isinstance(x, TorchMojoTensor)

    fake_tensor_module._check_for_subclass_arg = check_for_subclass_arg_except_mojo

    # Once past the subclass check, lifting a real mojo tensor constant
    # still fails: the const-propagation path is gated on `type(out) is
    # Tensor` (its no_dispatch clone is unsafe for subclasses), and the
    # fallback validation rejects non-fake inputs. Fakeify lifted mojo
    # constants directly instead; at runtime AOTAutograd passes the real
    # constant as a graph input like any other mojo tensor.
    original_dispatch_impl = fake_tensor_module.FakeTensorMode._dispatch_impl

    def dispatch_impl_lifting_mojo_constants(self, func, types, args, kwargs):
        if func in self.lift_fns and args and isinstance(args[0], TorchMojoTensor):
            return self.fake_tensor_converter.from_real_tensor(self, args[0])
        return original_dispatch_impl(self, func, types, args, kwargs)

    fake_tensor_module.FakeTensorMode._dispatch_impl = (
        dispatch_impl_lifting_mojo_constants
    )


def _keep_mojo_kernels_out_of_fake_tensor_construction():
    """Make FakeTensor construction skip the PrivateUse1 Python kernels.

    `FakeTensor.__new__` calls `Tensor._make_subclass(cls, elem, ...,
    device_for_backend_keys=mojo)`, which internally dispatches
    `aten::detach` on `elem`. While torch.compile traces mojo graphs, `elem`
    is regularly a meta tensor whose dispatch keys carry PrivateUse1 (view
    and matmul meta outputs inherit the fake input's keys), so that detach
    lands in our Python kernel — and `_make_subclass` requires a result
    with no Python object associated, which a Python kernel can never
    produce ("already associated to a python object" RuntimeError).

    Excluding PrivateUse1 for the duration of `FakeTensor.__new__` routes
    that internal detach to the stock C++ meta kernel. Real mojo tensors
    are never legitimately consumed during FakeTensor *construction*, so
    nothing of ours belongs there. (PyTorch's own python-backend reference
    test has this same rough edge — see test_privateuseone_python_backend
    "prevent compile-time FakeTensor crashes".)
    """
    from torch._subclasses.fake_tensor import FakeTensor

    exclude_privateuse1 = torch._C.DispatchKeySet(torch._C.DispatchKey.PrivateUse1)
    original_new = FakeTensor.__new__

    def fake_new_without_mojo_kernels(cls, *args, **kwargs):
        with torch._C._ExcludeDispatchKeyGuard(exclude_privateuse1):
            return original_new(cls, *args, **kwargs)

    FakeTensor.__new__ = staticmethod(fake_new_without_mojo_kernels)


def register_max_devices():
    """Enable the mojo device globally and register all aten ops"""
    from torch.utils.backend_registration import _setup_privateuseone_for_python_backend

    # since it's so recent we import it here.
    global _registered
    if _registered:
        # Already registered
        return

    _setup_privateuseone_for_python_backend("mojo")

    # Register all collected aten operations
    for op_name, func in _aten_ops_registry:
        torch.library.impl(op_name, "privateuseone")(func)

    _declare_mojo_tensor_as_plain_tensor()
    _keep_mojo_kernels_out_of_fake_tensor_construction()

    _registered = True
