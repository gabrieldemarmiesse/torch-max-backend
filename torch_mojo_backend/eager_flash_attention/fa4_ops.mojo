"""CPU-only-Torch-compatible bridge to the vendored dense FA4 kernels.

All launches use the backend-owned MAX DeviceContext default stream. They are
asynchronous; synchronization belongs at explicit consumer/benchmark
boundaries, never between forward or backward component kernels.
"""

from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder

from fa4_fwd_launch import launch_fwd_fa4
from fa4_bwd_launch import (
    launch_bwd_preprocess,
    launch_bwd_main,
    launch_bwd_convert,
)


def flash_attention_fwd_bf16_d64_causal(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    var q_addr = Int(py=args[0])
    var k_addr = Int(py=args[1])
    var v_addr = Int(py=args[2])
    var out_addr = Int(py=args[3])
    var lse_addr = Int(py=args[4])
    var batch = Int(py=args[5])
    var seqlen = Int(py=args[6])
    var nheads = Int(py=args[7])
    var softmax_scale = Float32(py=args[8])
    var ctx_addr = Int(py=args[9])

    if batch <= 0 or seqlen <= 0 or nheads <= 0:
        return PythonObject(None)

    launch_fwd_fa4[
        DType.bfloat16,
        64,
        False,
        True,
        1,
        False,
        False,
        False,
        0,
    ](
        batch,
        seqlen,
        nheads,
        softmax_scale,
        q_addr,
        k_addr,
        v_addr,
        out_addr,
        lse_addr,
        0,
        ctx_addr,
    )
    return PythonObject(None)


def flash_attention_bwd_bf16_d64_causal(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    var q_addr = Int(py=args[0])
    var k_addr = Int(py=args[1])
    var v_addr = Int(py=args[2])
    var out_addr = Int(py=args[3])
    var dout_addr = Int(py=args[4])
    var lse_addr = Int(py=args[5])
    var dq_addr = Int(py=args[6])
    var dk_addr = Int(py=args[7])
    var dv_addr = Int(py=args[8])
    var dpsum_addr = Int(py=args[9])
    var lse_log2_addr = Int(py=args[10])
    var dq_accum_addr = Int(py=args[11])
    var batch = Int(py=args[12])
    var seqlen = Int(py=args[13])
    var nheads = Int(py=args[14])
    var softmax_scale = Float32(py=args[15])
    var ctx_addr = Int(py=args[16])

    if batch <= 0 or seqlen <= 0 or nheads <= 0:
        return PythonObject(None)

    launch_bwd_preprocess[
        DType.bfloat16, 64, False, True, 1, False
    ](
        batch,
        seqlen,
        nheads,
        out_addr,
        dout_addr,
        lse_addr,
        dpsum_addr,
        lse_log2_addr,
        dq_accum_addr,
        0,
        0,
        0,
        ctx_addr,
    )
    launch_bwd_main[
        DType.bfloat16, 64, False, True, 1, False, False, 0
    ](
        batch,
        seqlen,
        nheads,
        softmax_scale,
        q_addr,
        k_addr,
        v_addr,
        dout_addr,
        dk_addr,
        dv_addr,
        lse_log2_addr,
        dpsum_addr,
        dq_accum_addr,
        0,
        ctx_addr,
    )
    launch_bwd_convert[
        DType.bfloat16, 64, False, True, 1, False
    ](
        batch,
        seqlen,
        nheads,
        softmax_scale,
        dq_accum_addr,
        dq_addr,
        0,
        ctx_addr,
    )
    return PythonObject(None)


def _check_strided_qkv_layout(
    name: StaticString,
    addr: Int,
    b_stride: Int,
    s_stride: Int,
    h_stride: Int,
    d_stride: Int,
    seqlen: Int,
    nheads: Int,
) raises:
    """Reject any Q/K/V layout outside the strict zero-copy regime.

    Runs BEFORE any descriptor is created or kernel enqueued so a
    violation never partially launches. Strides are in bf16 ELEMENTS.
    """
    if b_stride <= 0 or s_stride <= 0 or h_stride <= 0 or d_stride <= 0:
        raise Error(
            "fa4 strided qkv: ",
            name,
            " strides must all be positive, got (",
            b_stride,
            ", ",
            s_stride,
            ", ",
            h_stride,
            ", ",
            d_stride,
            ")",
        )
    if d_stride != 1:
        raise Error(
            "fa4 strided qkv: ", name, " d_stride must be 1, got ", d_stride
        )
    if h_stride != 64:
        raise Error(
            "fa4 strided qkv: ", name, " h_stride must be 64, got ", h_stride
        )
    if s_stride < nheads * 64:
        raise Error(
            "fa4 strided qkv: ",
            name,
            " s_stride ",
            s_stride,
            " must be >= nheads * 64 = ",
            nheads * 64,
        )
    if b_stride != seqlen * s_stride:
        raise Error(
            "fa4 strided qkv: ",
            name,
            " b_stride ",
            b_stride,
            " must equal seqlen * s_stride = ",
            seqlen * s_stride,
        )
    if addr % 16 != 0:
        raise Error(
            "fa4 strided qkv: ", name, " base address must be 16-byte aligned"
        )
    # TMA global strides are byte strides and every non-innermost one
    # must be a 16-byte multiple (bf16: 2 bytes per element).
    if (
        (b_stride * 2) % 16 != 0
        or (s_stride * 2) % 16 != 0
        or (h_stride * 2) % 16 != 0
    ):
        raise Error(
            "fa4 strided qkv: ",
            name,
            " non-innermost strides must be multiples of 16 bytes, got (",
            b_stride,
            ", ",
            s_stride,
            ", ",
            h_stride,
            ") elements",
        )


def _check_strided_qkv_args(
    batch: Int,
    seqlen: Int,
    nheads: Int,
) raises:
    if batch <= 0 or seqlen <= 0 or nheads <= 0:
        raise Error(
            "fa4 strided qkv: batch, seqlen and nheads must be positive,",
            " got (",
            batch,
            ", ",
            seqlen,
            ", ",
            nheads,
            ")",
        )
    if seqlen % 128 != 0:
        raise Error(
            "fa4 strided qkv: seqlen must be a multiple of 128, got ", seqlen
        )


def flash_attention_fwd_bf16_d64_causal_strided_qkv(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    """Zero-copy fwd: Q/K/V are strided (B, S, H, 64) views described
    by per-tensor runtime element strides (b, s, h, d); out/lse keep
    the contiguous layouts of the dense entry point."""
    var q_addr = Int(py=args[0])
    var q_b_stride = Int(py=args[1])
    var q_s_stride = Int(py=args[2])
    var q_h_stride = Int(py=args[3])
    var q_d_stride = Int(py=args[4])
    var k_addr = Int(py=args[5])
    var k_b_stride = Int(py=args[6])
    var k_s_stride = Int(py=args[7])
    var k_h_stride = Int(py=args[8])
    var k_d_stride = Int(py=args[9])
    var v_addr = Int(py=args[10])
    var v_b_stride = Int(py=args[11])
    var v_s_stride = Int(py=args[12])
    var v_h_stride = Int(py=args[13])
    var v_d_stride = Int(py=args[14])
    var out_addr = Int(py=args[15])
    var lse_addr = Int(py=args[16])
    var batch = Int(py=args[17])
    var seqlen = Int(py=args[18])
    var nheads = Int(py=args[19])
    var softmax_scale = Float32(py=args[20])
    var ctx_addr = Int(py=args[21])

    _check_strided_qkv_args(batch, seqlen, nheads)
    _check_strided_qkv_layout(
        "q",
        q_addr,
        q_b_stride,
        q_s_stride,
        q_h_stride,
        q_d_stride,
        seqlen,
        nheads,
    )
    _check_strided_qkv_layout(
        "k",
        k_addr,
        k_b_stride,
        k_s_stride,
        k_h_stride,
        k_d_stride,
        seqlen,
        nheads,
    )
    _check_strided_qkv_layout(
        "v",
        v_addr,
        v_b_stride,
        v_s_stride,
        v_h_stride,
        v_d_stride,
        seqlen,
        nheads,
    )

    launch_fwd_fa4[
        DType.bfloat16,
        64,
        False,
        True,
        1,
        False,
        False,
        False,
        0,
        strided_qkv=True,
    ](
        batch,
        seqlen,
        nheads,
        softmax_scale,
        q_addr,
        k_addr,
        v_addr,
        out_addr,
        lse_addr,
        0,
        ctx_addr,
        q_b_stride=q_b_stride,
        q_s_stride=q_s_stride,
        q_h_stride=q_h_stride,
        q_d_stride=q_d_stride,
        k_s_stride=k_s_stride,
        k_h_stride=k_h_stride,
        k_d_stride=k_d_stride,
        v_s_stride=v_s_stride,
        v_h_stride=v_h_stride,
        v_d_stride=v_d_stride,
    )
    return PythonObject(None)


def flash_attention_bwd_bf16_d64_causal_strided_qkv(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    """Zero-copy bwd: Q/K/V are strided (B, S, H, 64) views described
    by per-tensor runtime element strides (b, s, h, d); out/dout/lse,
    the dq/dk/dv outputs and all scratch keep the contiguous layouts
    of the dense entry point."""
    var q_addr = Int(py=args[0])
    var q_b_stride = Int(py=args[1])
    var q_s_stride = Int(py=args[2])
    var q_h_stride = Int(py=args[3])
    var q_d_stride = Int(py=args[4])
    var k_addr = Int(py=args[5])
    var k_b_stride = Int(py=args[6])
    var k_s_stride = Int(py=args[7])
    var k_h_stride = Int(py=args[8])
    var k_d_stride = Int(py=args[9])
    var v_addr = Int(py=args[10])
    var v_b_stride = Int(py=args[11])
    var v_s_stride = Int(py=args[12])
    var v_h_stride = Int(py=args[13])
    var v_d_stride = Int(py=args[14])
    var out_addr = Int(py=args[15])
    var dout_addr = Int(py=args[16])
    var lse_addr = Int(py=args[17])
    var dq_addr = Int(py=args[18])
    var dk_addr = Int(py=args[19])
    var dv_addr = Int(py=args[20])
    var dpsum_addr = Int(py=args[21])
    var lse_log2_addr = Int(py=args[22])
    var dq_accum_addr = Int(py=args[23])
    var batch = Int(py=args[24])
    var seqlen = Int(py=args[25])
    var nheads = Int(py=args[26])
    var softmax_scale = Float32(py=args[27])
    var ctx_addr = Int(py=args[28])

    # The whole layout contract is validated up front so preprocess
    # never launches for an unsupported layout.
    _check_strided_qkv_args(batch, seqlen, nheads)
    _check_strided_qkv_layout(
        "q",
        q_addr,
        q_b_stride,
        q_s_stride,
        q_h_stride,
        q_d_stride,
        seqlen,
        nheads,
    )
    _check_strided_qkv_layout(
        "k",
        k_addr,
        k_b_stride,
        k_s_stride,
        k_h_stride,
        k_d_stride,
        seqlen,
        nheads,
    )
    _check_strided_qkv_layout(
        "v",
        v_addr,
        v_b_stride,
        v_s_stride,
        v_h_stride,
        v_d_stride,
        seqlen,
        nheads,
    )

    launch_bwd_preprocess[
        DType.bfloat16, 64, False, True, 1, False
    ](
        batch,
        seqlen,
        nheads,
        out_addr,
        dout_addr,
        lse_addr,
        dpsum_addr,
        lse_log2_addr,
        dq_accum_addr,
        0,
        0,
        0,
        ctx_addr,
    )
    launch_bwd_main[
        DType.bfloat16,
        64,
        False,
        True,
        1,
        False,
        False,
        0,
        strided_qkv=True,
    ](
        batch,
        seqlen,
        nheads,
        softmax_scale,
        q_addr,
        k_addr,
        v_addr,
        dout_addr,
        dk_addr,
        dv_addr,
        lse_log2_addr,
        dpsum_addr,
        dq_accum_addr,
        0,
        ctx_addr,
        q_s_stride=q_s_stride,
        q_h_stride=q_h_stride,
        q_d_stride=q_d_stride,
        k_s_stride=k_s_stride,
        k_h_stride=k_h_stride,
        k_d_stride=k_d_stride,
        v_s_stride=v_s_stride,
        v_h_stride=v_h_stride,
        v_d_stride=v_d_stride,
    )
    launch_bwd_convert[
        DType.bfloat16, 64, False, True, 1, False
    ](
        batch,
        seqlen,
        nheads,
        softmax_scale,
        dq_accum_addr,
        dq_addr,
        0,
        ctx_addr,
    )
    return PythonObject(None)


@export
def PyInit_fa4_ops() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("fa4_ops")
        module.def_py_function[flash_attention_fwd_bf16_d64_causal](
            "flash_attention_fwd_bf16_d64_causal"
        )
        module.def_py_function[flash_attention_bwd_bf16_d64_causal](
            "flash_attention_bwd_bf16_d64_causal"
        )
        module.def_py_function[flash_attention_fwd_bf16_d64_causal_strided_qkv](
            "flash_attention_fwd_bf16_d64_causal_strided_qkv"
        )
        module.def_py_function[flash_attention_bwd_bf16_d64_causal_strided_qkv](
            "flash_attention_bwd_bf16_d64_causal_strided_qkv"
        )
        return module.finalize()
    except error:
        abort(String("failed to create FA4 Python module: ", error))
