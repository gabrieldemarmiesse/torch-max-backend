"""Dynamically routed H100 BF16 GEMM kernels.

The accepted-v2 implementation remains the fallback for every regime not
handled by the optimized NN, NT, and TN routes in this module.
"""

from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    barrier,
    block_idx,
    thread_idx,
)
from std.gpu.host import DeviceAttribute, DeviceBuffer, DeviceContext
from std.gpu.host.nvidia.tma import (
    TensorMapSwizzle,
    create_tma_descriptor,
)
from std.gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from std.gpu.memory import AddressSpace, fence_async_view_proxy
from std.memory import stack_allocation
from std.sys.info import _has_sm_9x, _is_sm_9x
from std.utils.index import Index, IndexList
from std.utils.static_tuple import StaticTuple

from layout import Layout, LayoutTensor
from layout.tensor_core_async import (
    TensorCoreAsync,
    tile_layout_k_major,
    tile_layout_mn_major,
    warpgroup_fence,
)
from layout.tma_async import SharedMemBarrier, TMATensorTile

from bf16_gemm_kernels import (
    enqueue_bf16_bmm as _enqueue_accepted_bf16_bmm,
    enqueue_bf16_gemm as _enqueue_accepted_bf16_gemm,
)


comptime _V3_BF16 = DType.bfloat16
comptime _V3_F32 = DType.float32
comptime _V3_PTR = UnsafePointer[Scalar[_V3_BF16], MutAnyOrigin]
comptime _V3_BM = 64
comptime _V3_BN = 128
comptime _V3_BK = 64
comptime _V3_SWIZZLE = TensorMapSwizzle.SWIZZLE_128B
comptime _V3_NO_SWIZZLE = TensorMapSwizzle.SWIZZLE_NONE
comptime _V3_A_LAYOUT = tile_layout_k_major[
    _V3_BF16, _V3_BM, _V3_BK, _V3_SWIZZLE
]()
comptime _V3_TN_A_RAW_LAYOUT = Layout.row_major(_V3_BK, _V3_BM)
comptime _V3_TN_A_LAYOUT = tile_layout_k_major[
    _V3_BF16, _V3_BM, _V3_BK, _V3_NO_SWIZZLE
]()
comptime _V3_B_MN_LAYOUT = tile_layout_mn_major[
    _V3_BF16, _V3_BN, _V3_BK, _V3_SWIZZLE
]()
comptime _V3_WGMMA_SHAPE = Index(64, 128, 16)
comptime _V3_A_TMA = TMATensorTile[
    _V3_BF16, 2, Index(_V3_BM, _V3_BK), Index(_V3_BM, _V3_BK)
]
comptime _V3_TN_A_TMA = TMATensorTile[
    _V3_BF16,
    2,
    Index(_V3_BK, _V3_BM),
    Index(_V3_BK, _V3_BM),
    False,
]
comptime _V3_B_MN_TMA = TMATensorTile[
    _V3_BF16,
    2,
    Index(_V3_BK, _V3_BN),
    Index(_V3_BK, 64),
]
comptime _V3_NT_BM = 128
comptime _V3_NT_BN = 256
comptime _V3_NT_BK = 64
comptime _V3_NT_STAGES = 3
comptime _V3_NT_THREADS = 384
comptime _V3_NT_CONSUMERS = 2
comptime _V3_NT_WGMMA_SHAPE = Index(64, 256, 16)
comptime _V3_NT_A_LAYOUT = tile_layout_k_major[
    _V3_BF16, _V3_NT_BM, _V3_NT_BK, _V3_SWIZZLE
]()
comptime _V3_B_K_LAYOUT = tile_layout_k_major[
    _V3_BF16, _V3_NT_BN, _V3_NT_BK, _V3_SWIZZLE
]()
comptime _V3_NT_A_TMA = TMATensorTile[
    _V3_BF16,
    2,
    Index(_V3_NT_BM, _V3_NT_BK),
    Index(_V3_NT_BM, _V3_NT_BK),
]
comptime _V3_B_K_TMA = TMATensorTile[
    _V3_BF16,
    2,
    Index(_V3_NT_BN, _V3_NT_BK),
    Index(_V3_NT_BN, _V3_NT_BK),
]
comptime _V3_NT_A_PIPE_LAYOUT = Layout.row_major(
    _V3_NT_STAGES, _V3_NT_BM * _V3_NT_BK
)
comptime _V3_NT_B_PIPE_LAYOUT = Layout.row_major(
    _V3_NT_STAGES, _V3_NT_BN * _V3_NT_BK
)


@__llvm_arg_metadata(a_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma, `nvvm.grid_constant`)
@__name("nanogpt_bf16_gemm_v3_nn_wgmma_tma_s2")
def _v3_nn_wgmma_tma_s2(
    a_tma: _V3_A_TMA,
    b_tma: _V3_B_MN_TMA,
    output: _V3_PTR,
    m: Int,
    n: Int,
    k: Int,
):
    comptime if _is_sm_9x():
        var a_smem0 = LayoutTensor[
            _V3_BF16,
            _V3_A_LAYOUT,
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
            alignment=128,
        ].stack_allocation()
        var a_smem1 = LayoutTensor[
            _V3_BF16,
            _V3_A_LAYOUT,
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
            alignment=128,
        ].stack_allocation()
        var b_smem0 = LayoutTensor[
            _V3_BF16,
            _V3_B_MN_LAYOUT,
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
            alignment=128,
        ].stack_allocation()
        var b_smem1 = LayoutTensor[
            _V3_BF16,
            _V3_B_MN_LAYOUT,
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
            alignment=128,
        ].stack_allocation()
        var mbar = stack_allocation[
            2,
            SharedMemBarrier,
            address_space=AddressSpace.SHARED,
            alignment=8,
        ]()
        if thread_idx.x == 0:
            mbar[0].init()
            mbar[1].init()

        comptime CFRAG = _V3_BM * _V3_BN // 128
        var accum = LayoutTensor[
            _V3_F32,
            Layout.row_major(1, CFRAG),
            MutAnyOrigin,
            address_space=AddressSpace.LOCAL,
        ].stack_allocation()
        _ = accum.fill(0.0)

        comptime wgmma = TensorCoreAsync[
            _V3_F32,
            _V3_BF16,
            _V3_BF16,
            _V3_WGMMA_SHAPE,
            a_swizzle=_V3_SWIZZLE,
            b_swizzle=_V3_SWIZZLE,
            transpose_b=False,
        ]()
        var phase0: UInt32 = 0
        var phase1: UInt32 = 0
        var blocks_m = m // _V3_BM
        var blocks_n = n // _V3_BN
        var lin = Int(block_idx.x)
        var group_span = 8 * blocks_n
        var group = lin // group_span
        var rem = lin % group_span
        var rows_in_group = min(8, blocks_m - group * 8)
        var m0 = (group * 8 + rem % rows_in_group) * _V3_BM
        var n0 = (rem // rows_in_group) * _V3_BN
        var num_tiles = k // _V3_BK
        comptime TMA_BYTES = (_V3_BM + _V3_BN) * _V3_BK * 2

        barrier()
        if thread_idx.x == 0:
            mbar[0].expect_bytes(Int32(TMA_BYTES))
            a_tma.async_copy(a_smem0, mbar[0], (0, m0))
            b_tma.async_copy(b_smem0, mbar[0], (n0, 0))
            if num_tiles > 1:
                mbar[1].expect_bytes(Int32(TMA_BYTES))
                a_tma.async_copy(a_smem1, mbar[1], (_V3_BK, m0))
                b_tma.async_copy(b_smem1, mbar[1], (n0, _V3_BK))
        barrier()

        var tile = 0
        while tile < num_tiles:
            var stage = tile % 2
            if stage == 0:
                mbar[0].wait(phase0)
                phase0 ^= 1
                warpgroup_fence(accum)
                wgmma.arrive()
                wgmma.wgmma(a_smem0, b_smem0, accum)
                wgmma.commit_group()
                warpgroup_fence(accum)
                wgmma.wait_group()
            else:
                mbar[1].wait(phase1)
                phase1 ^= 1
                warpgroup_fence(accum)
                wgmma.arrive()
                wgmma.wgmma(a_smem1, b_smem1, accum)
                wgmma.commit_group()
                warpgroup_fence(accum)
                wgmma.wait_group()

            barrier()
            var future = tile + 2
            if future < num_tiles and thread_idx.x == 0:
                var future_k = future * _V3_BK
                if stage == 0:
                    mbar[0].expect_bytes(Int32(TMA_BYTES))
                    a_tma.async_copy(a_smem0, mbar[0], (future_k, m0))
                    b_tma.async_copy(b_smem0, mbar[0], (n0, future_k))
                else:
                    mbar[1].expect_bytes(Int32(TMA_BYTES))
                    a_tma.async_copy(a_smem1, mbar[1], (future_k, m0))
                    b_tma.async_copy(b_smem1, mbar[1], (n0, future_k))
            tile += 1

        var tid = Int(thread_idx.x)
        var warp = tid // 32
        var lane = tid % 32
        var base_row = warp * 16 + lane // 4
        var base_col = (lane % 4) * 2

        @parameter
        for q in range(CFRAG // 2):
            var e = q * 2
            var row = base_row + (q % 2) * 8
            var col = base_col + (q // 2) * 8
            var pair = SIMD[_V3_BF16, 2](
                accum.ptr[e].cast[_V3_BF16](),
                accum.ptr[e + 1].cast[_V3_BF16](),
            )
            output.store[alignment=4]((m0 + row) * n + n0 + col, pair)


def _v3_enqueue_nn_wgmma_tma_s2(
    output: _V3_PTR,
    a: _V3_PTR,
    b: _V3_PTR,
    m: Int,
    n: Int,
    k: Int,
    grid_x: Int,
    ctx: DeviceContext,
) raises:
    var a_desc = create_tma_descriptor[_V3_BF16, 2, _V3_SWIZZLE](
        DeviceBuffer(
            ctx,
            a.address_space_cast[AddressSpace.GENERIC](),
            1,
            owning=False,
        ),
        IndexList[2](m, k),
        IndexList[2](k, 1),
        IndexList[2](_V3_BM, _V3_BK),
    )
    var b_desc = create_tma_descriptor[_V3_BF16, 2, _V3_SWIZZLE](
        DeviceBuffer(
            ctx,
            b.address_space_cast[AddressSpace.GENERIC](),
            1,
            owning=False,
        ),
        IndexList[2](k, n),
        IndexList[2](n, 1),
        IndexList[2](_V3_BK, 64),
    )
    var a_tma = _V3_A_TMA(a_desc)
    var b_tma = _V3_B_MN_TMA(b_desc)
    ctx.enqueue_function[_v3_nn_wgmma_tma_s2](
        a_tma,
        b_tma,
        output,
        m,
        n,
        k,
        grid_dim=(grid_x,),
        block_dim=(128,),
    )


@__llvm_arg_metadata(a_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma, `nvvm.grid_constant`)
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(_V3_NT_THREADS))
)
@__name("nanogpt_bf16_gemm_v3_nt_ws_m128n256_tma_s3")
def _v3_nt_ws_m128n256_tma_s3(
    a_tma: _V3_NT_A_TMA,
    b_tma: _V3_B_K_TMA,
    output: _V3_PTR,
    m: Int,
    n: Int,
    k: Int,
):
    comptime if _is_sm_9x():
        var a_pipeline = LayoutTensor[
            _V3_BF16,
            _V3_NT_A_PIPE_LAYOUT,
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
            alignment=128,
        ].stack_allocation()
        var b_pipeline = LayoutTensor[
            _V3_BF16,
            _V3_NT_B_PIPE_LAYOUT,
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
            alignment=128,
        ].stack_allocation()
        var full_barriers = stack_allocation[
            _V3_NT_STAGES,
            SharedMemBarrier,
            address_space=AddressSpace.SHARED,
            alignment=8,
        ]()
        var empty_barriers = stack_allocation[
            _V3_NT_STAGES,
            SharedMemBarrier,
            address_space=AddressSpace.SHARED,
            alignment=8,
        ]()
        if thread_idx.x == 0:
            comptime for stage in range(_V3_NT_STAGES):
                full_barriers[stage].init()
                empty_barriers[stage].init(Int32(_V3_NT_CONSUMERS))
            a_tma.prefetch_descriptor()
            b_tma.prefetch_descriptor()

        comptime CFRAG = 64 * _V3_NT_BN // 128
        var warp_group_idx = Int(thread_idx.x) // 128
        var warp_group_thread_idx = Int(thread_idx.x) % 128
        var blocks_m = (m + _V3_NT_BM - 1) // _V3_NT_BM
        var blocks_n = (n + _V3_NT_BN - 1) // _V3_NT_BN
        var lin = Int(block_idx.x)
        var group_span = 8 * blocks_n
        var group = lin // group_span
        var rem = lin % group_span
        var rows_in_group = min(8, blocks_m - group * 8)
        var m0 = (group * 8 + rem % rows_in_group) * _V3_NT_BM
        var n0 = (rem // rows_in_group) * _V3_NT_BN
        var num_tiles = k // _V3_NT_BK
        comptime TMA_BYTES = (_V3_NT_BM + _V3_NT_BN) * _V3_NT_BK * 2

        if warp_group_idx > 0 and warp_group_thread_idx == 0:
            comptime for stage in range(_V3_NT_STAGES):
                _ = empty_barriers[stage].arrive()
        barrier()

        if warp_group_idx == 0:
            warpgroup_reg_dealloc[24]()
            if thread_idx.x == 0:
                var tile = 0
                while tile < num_tiles:
                    var stage = tile % _V3_NT_STAGES
                    var phase = UInt32((tile // _V3_NT_STAGES) % 2)
                    empty_barriers[stage].wait(phase)
                    full_barriers[stage].expect_bytes(Int32(TMA_BYTES))

                    var a_tile = LayoutTensor[
                        _V3_BF16,
                        _V3_NT_A_LAYOUT,
                        MutAnyOrigin,
                        address_space=AddressSpace.SHARED,
                        alignment=128,
                    ](a_pipeline.ptr + stage * _V3_NT_BM * _V3_NT_BK)
                    var b_tile = LayoutTensor[
                        _V3_BF16,
                        _V3_B_K_LAYOUT,
                        MutAnyOrigin,
                        address_space=AddressSpace.SHARED,
                        alignment=128,
                    ](b_pipeline.ptr + stage * _V3_NT_BN * _V3_NT_BK)
                    var k0 = tile * _V3_NT_BK
                    a_tma.async_copy(a_tile, full_barriers[stage], (k0, m0))
                    b_tma.async_copy(b_tile, full_barriers[stage], (k0, n0))
                    tile += 1
        else:
            warpgroup_reg_alloc[232]()
            var accum = LayoutTensor[
                _V3_F32,
                Layout.row_major(1, CFRAG),
                MutAnyOrigin,
                address_space=AddressSpace.LOCAL,
            ].stack_allocation()
            _ = accum.fill(0.0)
            comptime wgmma = TensorCoreAsync[
                _V3_F32,
                _V3_BF16,
                _V3_BF16,
                _V3_NT_WGMMA_SHAPE,
                a_swizzle=_V3_SWIZZLE,
                b_swizzle=_V3_SWIZZLE,
                transpose_b=True,
            ]()

            var tile = 0
            while tile < num_tiles:
                var stage = tile % _V3_NT_STAGES
                var phase = UInt32((tile // _V3_NT_STAGES) % 2)
                full_barriers[stage].wait(phase)
                var a_tile = LayoutTensor[
                    _V3_BF16,
                    _V3_NT_A_LAYOUT,
                    MutAnyOrigin,
                    address_space=AddressSpace.SHARED,
                    alignment=128,
                ](a_pipeline.ptr + stage * _V3_NT_BM * _V3_NT_BK)
                var b_tile = LayoutTensor[
                    _V3_BF16,
                    _V3_B_K_LAYOUT,
                    MutAnyOrigin,
                    address_space=AddressSpace.SHARED,
                    alignment=128,
                ](b_pipeline.ptr + stage * _V3_NT_BN * _V3_NT_BK)
                warpgroup_fence(accum)
                wgmma.arrive()
                wgmma.wgmma[_V3_NT_CONSUMERS](
                    a_tile, b_tile, accum, warp_group_idx - 1
                )
                wgmma.commit_group()
                warpgroup_fence(accum)
                wgmma.wait_group()
                if warp_group_thread_idx == 0:
                    _ = empty_barriers[stage].arrive()
                tile += 1

            var tid = warp_group_thread_idx
            var warp = tid // 32
            var lane = tid % 32
            var base_row = warp * 16 + lane // 4
            var base_col = (lane % 4) * 2
            comptime for q in range(CFRAG // 2):
                var e = q * 2
                var row = (warp_group_idx - 1) * 64 + base_row + (q % 2) * 8
                var col = base_col + (q // 2) * 8
                var pair = SIMD[_V3_BF16, 2](
                    accum.ptr[e].cast[_V3_BF16](),
                    accum.ptr[e + 1].cast[_V3_BF16](),
                )
                if m0 + row < m and n0 + col + 1 < n:
                    output.store[alignment=4]((m0 + row) * n + n0 + col, pair)


def _v3_enqueue_nt_ws_m128n256_tma_s3(
    output: _V3_PTR,
    a: _V3_PTR,
    b: _V3_PTR,
    m: Int,
    n: Int,
    k: Int,
    grid_x: Int,
    ctx: DeviceContext,
) raises:
    var a_desc = create_tma_descriptor[_V3_BF16, 2, _V3_SWIZZLE](
        DeviceBuffer(
            ctx,
            a.address_space_cast[AddressSpace.GENERIC](),
            1,
            owning=False,
        ),
        IndexList[2](m, k),
        IndexList[2](k, 1),
        IndexList[2](_V3_NT_BM, _V3_NT_BK),
    )
    var b_desc = create_tma_descriptor[_V3_BF16, 2, _V3_SWIZZLE](
        DeviceBuffer(
            ctx,
            b.address_space_cast[AddressSpace.GENERIC](),
            1,
            owning=False,
        ),
        IndexList[2](n, k),
        IndexList[2](k, 1),
        IndexList[2](_V3_NT_BN, _V3_NT_BK),
    )
    var a_tma = _V3_NT_A_TMA(a_desc)
    var b_tma = _V3_B_K_TMA(b_desc)
    ctx.enqueue_function[_v3_nt_ws_m128n256_tma_s3](
        a_tma,
        b_tma,
        output,
        m,
        n,
        k,
        grid_dim=(grid_x,),
        block_dim=(_V3_NT_THREADS,),
    )


@__llvm_arg_metadata(a_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma, `nvvm.grid_constant`)
@__name("nanogpt_bf16_gemm_v3_tn_wgmma_tma_transpose_s2")
def _v3_tn_wgmma_tma_transpose_s2(
    a_tma: _V3_TN_A_TMA,
    b_tma: _V3_B_MN_TMA,
    output: _V3_PTR,
    m: Int,
    n: Int,
    k: Int,
):
    comptime if _is_sm_9x():
        var a_raw0 = LayoutTensor[
            _V3_BF16,
            _V3_TN_A_RAW_LAYOUT,
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
            alignment=128,
        ].stack_allocation()
        var a_raw1 = LayoutTensor[
            _V3_BF16,
            _V3_TN_A_RAW_LAYOUT,
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
            alignment=128,
        ].stack_allocation()
        var a_transposed = LayoutTensor[
            _V3_BF16,
            _V3_TN_A_LAYOUT,
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
            alignment=128,
        ].stack_allocation()
        var b_smem0 = LayoutTensor[
            _V3_BF16,
            _V3_B_MN_LAYOUT,
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
            alignment=128,
        ].stack_allocation()
        var b_smem1 = LayoutTensor[
            _V3_BF16,
            _V3_B_MN_LAYOUT,
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
            alignment=128,
        ].stack_allocation()
        var mbar = stack_allocation[
            2,
            SharedMemBarrier,
            address_space=AddressSpace.SHARED,
            alignment=8,
        ]()
        if thread_idx.x == 0:
            mbar[0].init()
            mbar[1].init()

        comptime CFRAG = _V3_BM * _V3_BN // 128
        var accum = LayoutTensor[
            _V3_F32,
            Layout.row_major(1, CFRAG),
            MutAnyOrigin,
            address_space=AddressSpace.LOCAL,
        ].stack_allocation()
        _ = accum.fill(0.0)

        comptime wgmma = TensorCoreAsync[
            _V3_F32,
            _V3_BF16,
            _V3_BF16,
            _V3_WGMMA_SHAPE,
            a_swizzle=_V3_NO_SWIZZLE,
            b_swizzle=_V3_SWIZZLE,
            transpose_b=False,
        ]()
        var phase0: UInt32 = 0
        var phase1: UInt32 = 0
        var blocks_m = m // _V3_BM
        var blocks_n = n // _V3_BN
        var lin = Int(block_idx.x)
        var group_span = 8 * blocks_n
        var group = lin // group_span
        var rem = lin % group_span
        var rows_in_group = min(8, blocks_m - group * 8)
        var m0 = (group * 8 + rem % rows_in_group) * _V3_BM
        var n0 = (rem // rows_in_group) * _V3_BN
        var num_tiles = k // _V3_BK
        comptime TMA_BYTES = (_V3_BM + _V3_BN) * _V3_BK * 2

        barrier()
        if thread_idx.x == 0:
            mbar[0].expect_bytes(Int32(TMA_BYTES))
            a_tma.async_copy(a_raw0, mbar[0], (m0, 0))
            b_tma.async_copy(b_smem0, mbar[0], (n0, 0))
            if num_tiles > 1:
                mbar[1].expect_bytes(Int32(TMA_BYTES))
                a_tma.async_copy(a_raw1, mbar[1], (m0, _V3_BK))
                b_tma.async_copy(b_smem1, mbar[1], (n0, _V3_BK))
        barrier()

        var tile = 0
        var tid = Int(thread_idx.x)
        while tile < num_tiles:
            var stage = tile % 2
            if stage == 0:
                mbar[0].wait(phase0)
                phase0 ^= 1

                @parameter
                for item in range(_V3_BM * _V3_BK // 128):
                    var raw_idx = tid + item * 128
                    var raw_k = raw_idx // _V3_BM
                    var raw_m = raw_idx % _V3_BM
                    var a_offset = (
                        (raw_m % 8) * 8
                        + (raw_m // 8) * 64
                        + raw_k % 8
                        + (raw_k // 8) * 512
                    )
                    a_transposed.ptr.store[alignment=2](
                        a_offset,
                        a_raw0.ptr.load[alignment=2](raw_idx),
                    )
            else:
                mbar[1].wait(phase1)
                phase1 ^= 1

                @parameter
                for item in range(_V3_BM * _V3_BK // 128):
                    var raw_idx = tid + item * 128
                    var raw_k = raw_idx // _V3_BM
                    var raw_m = raw_idx % _V3_BM
                    var a_offset = (
                        (raw_m % 8) * 8
                        + (raw_m // 8) * 64
                        + raw_k % 8
                        + (raw_k // 8) * 512
                    )
                    a_transposed.ptr.store[alignment=2](
                        a_offset,
                        a_raw1.ptr.load[alignment=2](raw_idx),
                    )

            # Scalar stores use the generic proxy; WGMMA reads the async view.
            fence_async_view_proxy()
            barrier()
            warpgroup_fence(accum)
            wgmma.arrive()
            if stage == 0:
                wgmma.wgmma(a_transposed, b_smem0, accum)
            else:
                wgmma.wgmma(a_transposed, b_smem1, accum)
            wgmma.commit_group()
            warpgroup_fence(accum)
            wgmma.wait_group[0]()

            # All WGMMA readers must finish before the single A tile is reused.
            barrier()
            var future = tile + 2
            if future < num_tiles and thread_idx.x == 0:
                var future_k = future * _V3_BK
                if stage == 0:
                    mbar[0].expect_bytes(Int32(TMA_BYTES))
                    a_tma.async_copy(a_raw0, mbar[0], (m0, future_k))
                    b_tma.async_copy(b_smem0, mbar[0], (n0, future_k))
                else:
                    mbar[1].expect_bytes(Int32(TMA_BYTES))
                    a_tma.async_copy(a_raw1, mbar[1], (m0, future_k))
                    b_tma.async_copy(b_smem1, mbar[1], (n0, future_k))
            tile += 1

        var warp = tid // 32
        var lane = tid % 32
        var base_row = warp * 16 + lane // 4
        var base_col = (lane % 4) * 2

        @parameter
        for q in range(CFRAG // 2):
            var e = q * 2
            var row = base_row + (q % 2) * 8
            var col = base_col + (q // 2) * 8
            var pair = SIMD[_V3_BF16, 2](
                accum.ptr[e].cast[_V3_BF16](),
                accum.ptr[e + 1].cast[_V3_BF16](),
            )
            output.store[alignment=4]((m0 + row) * n + n0 + col, pair)


def _v3_enqueue_tn_wgmma_tma_transpose_s2(
    output: _V3_PTR,
    a: _V3_PTR,
    b: _V3_PTR,
    m: Int,
    n: Int,
    k: Int,
    grid_x: Int,
    ctx: DeviceContext,
) raises:
    var a_desc = create_tma_descriptor[_V3_BF16, 2, _V3_NO_SWIZZLE](
        DeviceBuffer(
            ctx,
            a.address_space_cast[AddressSpace.GENERIC](),
            1,
            owning=False,
        ),
        IndexList[2](k, m),
        IndexList[2](m, 1),
        IndexList[2](_V3_BK, _V3_BM),
    )
    var b_desc = create_tma_descriptor[_V3_BF16, 2, _V3_SWIZZLE](
        DeviceBuffer(
            ctx,
            b.address_space_cast[AddressSpace.GENERIC](),
            1,
            owning=False,
        ),
        IndexList[2](k, n),
        IndexList[2](n, 1),
        IndexList[2](_V3_BK, 64),
    )
    var a_tma = _V3_TN_A_TMA(a_desc)
    var b_tma = _V3_B_MN_TMA(b_desc)
    ctx.enqueue_function[_v3_tn_wgmma_tma_transpose_s2](
        a_tma,
        b_tma,
        output,
        m,
        n,
        k,
        grid_dim=(grid_x,),
        block_dim=(128,),
    )


def enqueue_bf16_gemm(
    output: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    a: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    b: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    bias: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    m: Int,
    n: Int,
    k: Int,
    transpose_a: Bool,
    transpose_b: Bool,
    has_bias: Bool,
    ctx: DeviceContext,
) raises:
    comptime if _has_sm_9x():
        if ctx.api() == "cuda":
            var cc_major = ctx.get_attribute(
                DeviceAttribute.COMPUTE_CAPABILITY_MAJOR
            )
            var cc_minor = ctx.get_attribute(
                DeviceAttribute.COMPUTE_CAPABILITY_MINOR
            )
            if cc_major == 9 and cc_minor == 0:
                # Full-tile NN regime only. The ordered bounds make all
                # descriptor and address products machine-width safe; TMA
                # receives 16B-aligned bases and 128B-compatible rows.
                if (
                    not transpose_a
                    and not transpose_b
                    and not has_bias
                    and m >= _V3_BM
                    and n >= _V3_BN
                    and k >= _V3_BK
                    and m // n >= 8
                    and m % _V3_BM == 0
                    and n % _V3_BN == 0
                    and k % _V3_BK == 0
                    and Int(output) % 16 == 0
                    and Int(a) % 16 == 0
                    and Int(b) % 16 == 0
                    and m <= 2_147_483_647
                    and n <= 2_147_483_647
                    and k <= 2_147_483_647
                    and k <= 9_223_372_036_854_775_807 // m
                    and k <= 9_223_372_036_854_775_807 // n
                    and n <= 9_223_372_036_854_775_807 // m
                ):
                    var blocks_m = m // _V3_BM
                    var blocks_n = n // _V3_BN
                    var max_grid_x = ctx.get_attribute(
                        DeviceAttribute.MAX_GRID_DIM_X
                    )
                    if (
                        blocks_m > 0
                        and blocks_n > 0
                        and max_grid_x > 0
                        and blocks_m <= max_grid_x // blocks_n
                    ):
                        var grid_x = blocks_m * blocks_n
                        if grid_x > 0:
                            _v3_enqueue_nn_wgmma_tma_s2(
                                output, a, b, m, n, k, grid_x, ctx
                            )
                            return
                # Full-tile NT regime. B is physical row-major (n, k); both
                # operands use 128B-swizzled k-major shared layouts.
                if (
                    not transpose_a
                    and transpose_b
                    and not has_bias
                    and m >= _V3_BM
                    and n >= _V3_BN
                    and k >= _V3_BK
                    and m % _V3_BM == 0
                    and n % _V3_BN == 0
                    and k % _V3_BK == 0
                    and Int(output) % 16 == 0
                    and Int(a) % 16 == 0
                    and Int(b) % 16 == 0
                    and m <= 2_147_483_647
                    and n <= 2_147_483_647
                    and k <= 2_147_483_647
                    and k <= 9_223_372_036_854_775_807 // m
                    and k <= 9_223_372_036_854_775_807 // n
                    and n <= 9_223_372_036_854_775_807 // m
                ):
                    var blocks_m = (m + _V3_NT_BM - 1) // _V3_NT_BM
                    var blocks_n = (n + _V3_NT_BN - 1) // _V3_NT_BN
                    var max_grid_x = ctx.get_attribute(
                        DeviceAttribute.MAX_GRID_DIM_X
                    )
                    if (
                        blocks_m > 0
                        and blocks_n > 0
                        and max_grid_x > 0
                        and blocks_m <= max_grid_x // blocks_n
                    ):
                        var grid_x = blocks_m * blocks_n
                        if grid_x > 0:
                            _v3_enqueue_nt_ws_m128n256_tma_s3(
                                output, a, b, m, n, k, grid_x, ctx
                            )
                            return
                # Full-tile TN regime. A is physical row-major (k, m), then
                # cooperatively transposed into an unswizzled k-major WGMMA
                # tile. B retains the proven swizzled MN-major path.
                if (
                    transpose_a
                    and not transpose_b
                    and not has_bias
                    and m >= _V3_BM
                    and n >= _V3_BN
                    and k >= _V3_BK
                    and m % _V3_BM == 0
                    and n % _V3_BN == 0
                    and k % _V3_BK == 0
                    and Int(output) % 16 == 0
                    and Int(a) % 16 == 0
                    and Int(b) % 16 == 0
                    and m <= 2_147_483_647
                    and n <= 2_147_483_647
                    and k <= 2_147_483_647
                    and k <= 9_223_372_036_854_775_807 // m
                    and k <= 9_223_372_036_854_775_807 // n
                    and n <= 9_223_372_036_854_775_807 // m
                ):
                    var blocks_m = m // _V3_BM
                    var blocks_n = n // _V3_BN
                    var max_grid_x = ctx.get_attribute(
                        DeviceAttribute.MAX_GRID_DIM_X
                    )
                    if (
                        blocks_m > 0
                        and blocks_n > 0
                        and max_grid_x > 0
                        and blocks_m <= max_grid_x // blocks_n
                    ):
                        var grid_x = blocks_m * blocks_n
                        if grid_x > 0:
                            _v3_enqueue_tn_wgmma_tma_transpose_s2(
                                output, a, b, m, n, k, grid_x, ctx
                            )
                            return
    _enqueue_accepted_bf16_gemm(
        output,
        a,
        b,
        bias,
        m,
        n,
        k,
        transpose_a,
        transpose_b,
        has_bias,
        ctx,
    )


def enqueue_bf16_bmm(
    output: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    a: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    b: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    batch_count: Int,
    m: Int,
    n: Int,
    k: Int,
    output_batch_stride: Int,
    a_batch_stride: Int,
    b_batch_stride: Int,
    transpose_a: Bool,
    transpose_b: Bool,
    ctx: DeviceContext,
) raises:
    _enqueue_accepted_bf16_bmm(
        output,
        a,
        b,
        batch_count,
        m,
        n,
        k,
        output_batch_stride,
        a_batch_stride,
        b_batch_stride,
        transpose_a,
        transpose_b,
        ctx,
    )
