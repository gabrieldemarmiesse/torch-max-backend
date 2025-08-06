import inspect

import torch
from max.torch.torch import MaxOp


class MyMaxOp(MaxOp):
    def __init__(self, *args, num_inputs: int, **kwargs):
        self.num_inputs = num_inputs
        super().__init__(*args, **kwargs)

    @property
    def torch_signature(self) -> inspect.Signature:
        dps_args = [
            inspect.Parameter(
                f"__out{i}",
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=torch.Tensor,
            )
            for i in range(self.num_outputs)
        ]
        args = [
            inspect.Parameter(
                f"__input{i}",
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=torch.Tensor,
            )
            for i in range(self.num_inputs)
        ]
        return inspect.Signature((*dps_args, *args), return_annotation=None)
