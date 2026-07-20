"""Runtime metadata contract for FP32 foreach clipping kernels.

Tensor addresses, lengths, descriptor counts, and chunk counts are runtime
values. The descriptor cap only bounds one launch argument; callers split
longer lists into batches without changing the compiled kernel.
"""

from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder


comptime FOREACH_DESC_CAP = 64
comptime FOREACH_CHUNK_ELEMENTS = 65_536
comptime FOREACH_THREADS = 256


struct ForeachDesc(
    DevicePassable,
    ImplicitlyCopyable,
    TrivialRegisterPassable,
):
    comptime device_type: AnyType = Self

    var tensor_addr: Int
    var output_addr: Int
    var numel: Int
    var chunk_end: Int

    def __init__(
        out self,
        tensor_addr: Int,
        output_addr: Int,
        numel: Int,
        chunk_end: Int,
    ):
        self.tensor_addr = tensor_addr
        self.output_addr = output_addr
        self.numel = numel
        self.chunk_end = chunk_end

    def _to_device_type(
        self,
        mut encoder: Some[DeviceTypeEncoder],
        target: MutOpaquePointer[_],
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "ForeachDesc"


@always_inline
def empty_foreach_desc() -> ForeachDesc:
    return ForeachDesc(0, 0, 0, 0)
