"""Shared dispatch helpers for eager cross entropy.

``aten::cross_entropy_loss`` is CompositeImplicitAutograd in PyTorch.  Once a
backend installs concrete PrivateUse1 and AutogradPrivateUse1 kernels, calling
the operator again for an unsupported regime would recurse into those kernels.
``OpOverload.decompose`` invokes the pinned composite body directly while its
constituent ATen operations dispatch normally to the Mojo backend.
"""

from typing import no_type_check

import torch


@no_type_check
def decompose_cross_entropy_loss(
    input, target, weight=None, reduction=1, ignore_index=-100, label_smoothing=0.0
):
    result = torch.ops.aten.cross_entropy_loss.default.decompose(
        input, target, weight, reduction, ignore_index, label_smoothing
    )
    if result is NotImplemented:
        raise RuntimeError(
            "aten::cross_entropy_loss lost its CompositeImplicitAutograd "
            "implementation; the Mojo fallback cannot preserve PyTorch semantics"
        )
    return result


__all__ = ["decompose_cross_entropy_loss"]
