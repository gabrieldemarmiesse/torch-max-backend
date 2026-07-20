"""fp16 RS (register-A) wgmma emitter — vendored stdlib gap-filler.

The pinned stdlib's register-A ``wgmma_async`` overload
(``std/gpu/compute/mma.mojo``) is comptime-asserted bf16-only with a
hardcoded ``.bf16.bf16`` instruction suffix, although sm_90a wgmma
supports f16 identically (``wgmma.mma_async...f32.f16.f16``). This is
the same stdlib-over-restriction class as the SM100-gated
``cp_async_bulk_reduce`` (see CLAUDE.md). The descriptor-descriptor
(SS) overloads go through the NVVM intrinsic with generic dtype
mapping and need no help.

This module vendors exactly the shapes the kernels need —
m64n128k16 (head_dim=128) and m64n64k16 (head_dim=64), f32 accum —
with the instruction suffix switched to f16. The asm bodies,
register packing, and constraint construction mirror the stdlib's
n == 128 / n == 64 arms 1:1 so the f16 codegen matches the proven
bf16 path. Used ONLY under ``comptime dtype == DType.float16``
forks at the three RS call sites (fwd PV, bwd dV, bwd dK); bf16
keeps the stdlib path byte-identical.
"""

from std.sys import _RegisterPackType
from std.sys._assembly import inlined_assembly
from std.memory import bitcast


def _iota_ties[count: Int]() -> String:
    """"0,1,2,...,count-1" — the input-to-output tie list."""
    var s = String()
    for i in range(count):
        s += String(i)
        if i < count - 1:
            s += ","
    return s


def _reg_spec[count: Int]() -> String:
    """"$0, $1, ..., $count-1" — the output register list."""
    var s = String()
    for i in range(count):
        s += "$" + String(i)
        if i < count - 1:
            s += ", "
    return s


@always_inline
def wgmma_rs_f16_m64n128[
    *, scale_d: Int = 1, trans_b: Int = 1
](
    a_frag: SIMD[DType.float16, 8],
    desc_b: Int64,
    c: SIMD[DType.float32, 64],
) -> SIMD[DType.float32, 64]:
    """One m64n128k16 f32.f16.f16 wgmma with A in registers.

    a_frag: the warp's 8 f16 A elements for this k-step (k-major
    fragment order, identical to the bf16 path). desc_b: the B smem
    descriptor VALUE (``(_wgmma_descriptor[...](ptr) + off).desc``).
    trans_b follows the stdlib mapping (mn-major B / transpose_b ==
    False -> "row" -> 1).
    """
    var a0 = bitcast[DType.uint32, 1](
        SIMD[DType.float16, 2](a_frag[0], a_frag[1])
    )
    var a1 = bitcast[DType.uint32, 1](
        SIMD[DType.float16, 2](a_frag[2], a_frag[3])
    )
    var a2 = bitcast[DType.uint32, 1](
        SIMD[DType.float16, 2](a_frag[4], a_frag[5])
    )
    var a3 = bitcast[DType.uint32, 1](
        SIMD[DType.float16, 2](a_frag[6], a_frag[7])
    )

    comptime n: Int = 128
    comptime input_reg_spec = _reg_spec[n // 2]()
    comptime constraints = (
        "=f," * (n // 2) + "r,r,r,r,l,n,n,n,n," + _iota_ties[n // 2]()
    )

    var r = inlined_assembly[
        """{
            .reg .pred p;
            setp.ne.b32 p, $69, 0;
            wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16
            {""" + input_reg_spec + """},
             {$64, $65, $66, $67},
             $68, p, $70, $71, $72;
        }""",
        _RegisterPackType[
            Float32, Float32, Float32, Float32,
            Float32, Float32, Float32, Float32,
            Float32, Float32, Float32, Float32,
            Float32, Float32, Float32, Float32,
            Float32, Float32, Float32, Float32,
            Float32, Float32, Float32, Float32,
            Float32, Float32, Float32, Float32,
            Float32, Float32, Float32, Float32,
            Float32, Float32, Float32, Float32,
            Float32, Float32, Float32, Float32,
            Float32, Float32, Float32, Float32,
            Float32, Float32, Float32, Float32,
            Float32, Float32, Float32, Float32,
            Float32, Float32, Float32, Float32,
            Float32, Float32, Float32, Float32,
            Float32, Float32, Float32, Float32,
        ],
        constraints=constraints,
    ](
        a0, a1, a2, a3,
        desc_b,
        Int32(scale_d), Int32(1), Int32(1), Int32(trans_b),
        c[0],  c[1],  c[2],  c[3],  c[4],  c[5],  c[6],  c[7],
        c[8],  c[9],  c[10], c[11], c[12], c[13], c[14], c[15],
        c[16], c[17], c[18], c[19], c[20], c[21], c[22], c[23],
        c[24], c[25], c[26], c[27], c[28], c[29], c[30], c[31],
        c[32], c[33], c[34], c[35], c[36], c[37], c[38], c[39],
        c[40], c[41], c[42], c[43], c[44], c[45], c[46], c[47],
        c[48], c[49], c[50], c[51], c[52], c[53], c[54], c[55],
        c[56], c[57], c[58], c[59], c[60], c[61], c[62], c[63],
    )
    var out = SIMD[DType.float32, 64]()
    comptime for i in range(64):
        out[i] = r[i]
    return out


@always_inline
def wgmma_rs_f16_m64n64[
    *, scale_d: Int = 1, trans_b: Int = 1
](
    a_frag: SIMD[DType.float16, 8],
    desc_b: Int64,
    c: SIMD[DType.float32, 32],
) -> SIMD[DType.float32, 32]:
    """One m64n64k16 f32.f16.f16 wgmma with A in registers
    (head_dim=64's PV / dV / dK shape). Mirrors the n == 128 arm
    above with the c-frag halved."""
    var a0 = bitcast[DType.uint32, 1](
        SIMD[DType.float16, 2](a_frag[0], a_frag[1])
    )
    var a1 = bitcast[DType.uint32, 1](
        SIMD[DType.float16, 2](a_frag[2], a_frag[3])
    )
    var a2 = bitcast[DType.uint32, 1](
        SIMD[DType.float16, 2](a_frag[4], a_frag[5])
    )
    var a3 = bitcast[DType.uint32, 1](
        SIMD[DType.float16, 2](a_frag[6], a_frag[7])
    )

    comptime n: Int = 64
    comptime input_reg_spec = _reg_spec[n // 2]()
    comptime constraints = (
        "=f," * (n // 2) + "r,r,r,r,l,n,n,n,n," + _iota_ties[n // 2]()
    )

    var r = inlined_assembly[
        """{
            .reg .pred p;
            setp.ne.b32 p, $37, 0;
            wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16
            {""" + input_reg_spec + """},
             {$32, $33, $34, $35},
             $36, p, $38, $39, $40;
        }""",
        _RegisterPackType[
            Float32, Float32, Float32, Float32,
            Float32, Float32, Float32, Float32,
            Float32, Float32, Float32, Float32,
            Float32, Float32, Float32, Float32,
            Float32, Float32, Float32, Float32,
            Float32, Float32, Float32, Float32,
            Float32, Float32, Float32, Float32,
            Float32, Float32, Float32, Float32,
        ],
        constraints=constraints,
    ](
        a0, a1, a2, a3,
        desc_b,
        Int32(scale_d), Int32(1), Int32(1), Int32(trans_b),
        c[0],  c[1],  c[2],  c[3],  c[4],  c[5],  c[6],  c[7],
        c[8],  c[9],  c[10], c[11], c[12], c[13], c[14], c[15],
        c[16], c[17], c[18], c[19], c[20], c[21], c[22], c[23],
        c[24], c[25], c[26], c[27], c[28], c[29], c[30], c[31],
    )
    var out = SIMD[DType.float32, 32]()
    comptime for i in range(32):
        out[i] = r[i]
    return out
