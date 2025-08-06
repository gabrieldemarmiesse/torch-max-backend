import torch


def get_the_number_of_outputs(gm: torch.fx.GraphModule) -> int:
    last_node = next(iter(reversed(gm.graph.nodes)))
    assert last_node.op == "output"
    return len(last_node.args[0])
