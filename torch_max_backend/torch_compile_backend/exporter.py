"""We use torch.export to generate a MAX graph"""

import max.engine
import torch
import torch.export
from torch.export.graph_signature import InputKind

from torch_max_backend.aten_functions import DECOMPOSITION_TABLE
from torch_max_backend.torch_compile_backend.compiler import (
    _GraphFactory,
    global_max_objects,
)


def export_to_max_graph(
    model: torch.nn.Module, example_inputs: tuple[torch.Tensor, ...], force_device=None
) -> max.engine.Model:
    exported_program = torch.export.export(model, example_inputs, strict=True)
    with torch.no_grad():
        exported_program = exported_program.run_decompositions(
            decomp_table=DECOMPOSITION_TABLE
        )
    state_dict = model.state_dict()
    # embed weights into the graph
    replace_inputs = {}
    for input_spec in exported_program.graph_signature.input_specs:
        if input_spec.kind != InputKind.PARAMETER:
            continue
        replace_inputs[input_spec.arg.name] = state_dict[input_spec.target]

    factory = _GraphFactory(replace_inputs, force_device=force_device)
    graph, _ = factory.create_graph(exported_program.graph)

    model = global_max_objects().session.load(graph)
    return model
