import torch
from max.dtype import DType

from max.graph import Graph
from max.torch.torch import max_device_ref
import max.graph.value
from max import engine
from max.driver import Accelerator, accelerator_count, CPU
from .mappings import MAPPING_TORCH_TO_MOJO_FUNCTIONS
import uuid
import warnings

from max.graph import DeviceRef


def get_fully_qualified_name(func):
    result = ""
    if hasattr(func, "__module__"):
        result += func.__module__ + "."

    if hasattr(func, "__qualname__"):
        result += func.__qualname__

    result += " of type " + str(type(func)) + " "
    return result


def keep_only_tensors(inputs: list, detach: bool = False) -> list[torch.Tensor]:
    result = []
    for x in inputs:
        if isinstance(x, torch.Tensor):
            if detach:
                x = x.detach()
            result.append(x)
    return result


class TensorsBook:
    def __init__(self):
        self.tensors = {}

    def __setitem__(self, name: str, tensor):
        self.tensors[name] = tensor

    def convert_to_max(self, something):
        if isinstance(something, torch.fx.Node):
            return self.tensors[something.name]
        elif isinstance(something, str):
            return something
        elif isinstance(something, int):
            return something
        elif isinstance(something, float):
            return something
        elif isinstance(something, slice):
            return something
        elif isinstance(something, torch.fx.immutable_collections.immutable_list):
            return [self.convert_to_max(x) for x in something]
        elif isinstance(something, tuple):
            return tuple(self.convert_to_max(x) for x in something)
        elif isinstance(something, torch.device):
            return something
        elif isinstance(something, torch.dtype):
            return something
        elif something is None:
            return None
        elif something == ...:
            return ...
        elif isinstance(something, torch.nn.Module):
            return something
        raise ValueError(f"Unsupported type when reading the graph: {type(something)}")


class GraphFunction:
    def __init__(self, gm: torch.fx.GraphModule):
        self.gm = gm

    def fetch_attr(self, target: str):
        """Fetch an attribute from the Module hierarchy of self.gm.
        Args:
            target (str): The fully-qualified name of the attribute to fetch
        """
        target_atoms = target.split(".")
        attr_itr = self.gm
        for i, atom in enumerate(target_atoms):
            if not hasattr(attr_itr, atom):
                raise RuntimeError(
                    f"Node referenced nonexistent target {'.'.join(target_atoms[: i + 1])}"
                )
            attr_itr = getattr(attr_itr, atom)
        return attr_itr

    def __call__(
        self, *args: max.graph.value.TensorValue
    ) -> tuple[max.graph.value.TensorValue, ...]:
        tensor_book = TensorsBook()
        args_index = 0
        for node_idx, node in enumerate(self.gm.graph.nodes):
            if node.op == "placeholder":
                if node.name.startswith("s"):
                    # shape input
                    continue
                tensor_book[node.name] = args[args_index]
                args_index += 1
            elif node.op in ("call_function", "call_method"):
                func_args = [tensor_book.convert_to_max(x) for x in node.args]
                func_kwags = {
                    k: tensor_book.convert_to_max(v) for k, v in node.kwargs.items()
                }
                if node.target not in MAPPING_TORCH_TO_MOJO_FUNCTIONS:
                    if isinstance(node.target, str):
                        raise ValueError(
                            f"Failing at node {node_idx}. Method torch.Tensor.{node.target} not supported by the Max backend yet."
                        )
                    else:
                        raise ValueError(
                            f"Failing at node {node_idx}. Function {get_fully_qualified_name(node.target)} not supported by the Max backend yet."
                        )
                tensor = MAPPING_TORCH_TO_MOJO_FUNCTIONS[node.target](
                    *func_args, **func_kwags
                )
                tensor_book[node.name] = tensor
            elif node.op == "get_attr":
                attr_value = self.fetch_attr(node.target)
                tensor_book[node.name] = attr_value
            elif node.op == "output":
                return tuple(tensor_book.convert_to_max(x) for x in node.args[0])
            else:
                raise ValueError(f"Unsupported node type: {node.op}")


def generate_input_types(
    example_inputs: list[torch.Tensor],
) -> list[max.graph.value.TensorType]:
    result = []
    for inp in example_inputs:
        if not isinstance(inp, torch.Tensor):
            continue
        shape = []
        for dim_idx, dim in enumerate(inp.shape):
            if dim_idx in getattr(inp, "_dynamo_dynamic_indices", {}):
                shape.append("a" + str(uuid.uuid4()).replace("-", "_"))
            else:
                shape.append(int(dim))
        result.append(
            max.graph.value.TensorType(
                dtype=DType.from_torch(inp.dtype),
                shape=shape,
                device=max_device_ref(inp.device),
            )
        )
    return result


def get_accelerators():
    yield CPU()
    if accelerator_count() > 0:
        for i in range(accelerator_count()):
            try:
                yield Accelerator(i)
            except ValueError as e:
                warnings.warn(f"Failed to create accelerator {i}. {e}")


def deviceref_to_torch(device_ref: DeviceRef) -> torch.device:
    """Returns the equivalent PyTorch device for a MAX graph device."""
    if device_ref.api == "cpu":
        return torch.device(f"cpu:{device_ref.id}")
    elif device_ref.api == "cuda":
        return torch.device(f"cuda:{device_ref.id}")
    else:
        raise TypeError(f"Unable to convert {device_ref} to a PyTorch device.")


class MaxCompiler:
    def __init__(
        self, gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor], mode=None
    ):
        self.gm = gm
        self.example_inputs = example_inputs
        gm.graph.print_tabular()
        print(f"number of nodes: {len(gm.graph.nodes)}")

        max_input_specs = generate_input_types(keep_only_tensors(example_inputs))
        print(f"max_input_specs: {max_input_specs}")
        with Graph("some_graph", input_types=max_input_specs) as graph:
            outputs = GraphFunction(self.gm)(*graph.inputs)
            graph.output(*outputs)

        session = engine.InferenceSession(devices=list(get_accelerators()))
        self.model = session.load(graph)

    def __call__(self, *args) -> list[torch.Tensor]:
        # Detach tensors to avoid gradient tracking issues with DLpack
        outputs = self.model.execute(*keep_only_tensors(args, detach=True))
        return [torch.from_dlpack(x) for x in outputs]
