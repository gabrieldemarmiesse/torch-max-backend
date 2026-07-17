# Phase 1 GEMM diagnosis

Environment: AMD Instinct MI300X VF (`gfx942`), PyTorch 2.11.0+rocm7.2,
MAX 26.5.0.dev2026061806, Mojo 1.0.0b3.dev2026061806. The measured workload
is the frozen GPT-2 batch-512, prompt-8, BF16 eager generation workload.

## Result

The primary diagnosis is **no-MFMA BF16 kernel caused by a routing/coverage
gap**. The eager backend's dynamic AMD MFMA path exists but is restricted to
FP32. BF16 therefore runs the portable scalar/VALU tiled GEMM. A secondary
dispatch-hygiene issue is that BF16 `addmm` launches a separate bias-add
kernel. Grid starvation affects the worst decode projection, but it is not
the root cause: the same kernel remains 15.47x slower than ROCm on 8192^3 and
23.21x slower on the well-filled M=4096 worst projection.

The cheapest fix is to route BF16 through the existing runtime-dimension AMD
MFMA wrapper, make its input/output type generic, and use its existing bias
epilogue. No PyTorch-facing M/N/K value needs to become compile-time.

## D1. Kernel inventory and operand layout

For the first occurrence of each decode GEMM in `mojo_profile/trace_decode.json`:

| Logical GEMM (M,N,K) | Mojo kernels | Profiled GPU duration | Grid |
|---|---|---:|---:|
| 512,2304,768 | `pure_gemm_tiled_bfloat16_64x64x16_tbFalse` + bias elementwise | 233.330 + 8.057 us | 36x8 + 4608 |
| 512,768,768 | `pure_gemm_tiled_bfloat16_64x64x16_tbFalse` + bias elementwise | 232.769 + 4.890 us | 12x8 + 1536 |
| 512,3072,768 | `pure_gemm_tiled_bfloat16_64x64x16_tbFalse` + bias elementwise | 236.858 + 9.541 us | 48x8 + 6144 |
| 512,768,3072 | `pure_gemm_tiled_bfloat16_64x64x16_tbFalse` + bias elementwise | 917.728 + 5.331 us | 12x8 + 1536 |
| LM head, 512,50257,768 | `pure_gemm_tiled_bfloat16_128x128x16_tbTrue` | 1137.788 us | 393x4 |

The worst addmm has profiler external ID 1142. It launches exactly two
kernels: the GEMM and row-broadcast bias. There is no transpose
materialization, BF16/FP32 cast kernel, or output-copy kernel. Its profiler
input shapes/strides are `[512,3072]` stride `[3072,1]` and `[3072,768]`
stride `[768,1]`. This confirms Hugging Face `Conv1D` is consumed directly as
`x @ W` with W in `[in_features,out_features]` layout.

The matching native ROCm event (external ID 2314) launches one fused
hipBLASLt MFMA kernel. Its trace duration is 26.258 us. Thus bias fusion is a
valid secondary fix, while transpose/cast/copy removal is not applicable.

## D2. Ceiling check

`bench_gemm.py --include-ceiling` used 25 warmups and 100 timed iterations:

| Shape | Mojo median | ROCm median | Mojo/ROCm | Mojo TFLOPS | ROCm TFLOPS |
|---|---:|---:|---:|---:|---:|
| 8192x8192x8192 BF16 | 26,608.203 us | 1,719.762 us | 15.472x | 41.322 | 639.339 |

Mojo is 6.46% of ROCm throughput, far below the 50% decision threshold, on a
shape with abundant parallelism. This proves the selected base BF16 kernel is
deficient; the problem is not limited to M=512 selection.

## D3. Grid-fill arithmetic

The selected decode BF16 tile is 64x64. Workgroup counts are:

| Shape (M,N,K) | Workgroups | Relative to 304 CUs |
|---|---:|---:|
| 512,768,768 | 8x12 = 96 | 0.316 WG/CU |
| 512,768,3072 | 8x12 = 96 | 0.316 WG/CU |
| 512,2304,768 | 8x36 = 288 | 0.947 WG/CU |
| 512,3072,768 | 8x48 = 384 | 1.263 WG/CU |

The worst absolute-gap shape is grid-starved, and BF16 split-K is disabled in
`_gemm_enqueue`. However, the M=4096 version has 64x12 = 768 workgroups and
is still 23.207x slower than ROCm. The LM-head grid has 1,572 workgroups and
is still 8.535x slower. Split-K/tile selection is therefore a follow-on
tuning tool, not the first fix.

After the MFMA routing fix, the Change 9 ROCm trace gives more precise
schedule evidence. hipBLASLt uses macro tiles `64x64x128`, `64x32x128`, and
`128x64x128` for the three K=768 projection families. For the remaining
K-dominant `(512,768,3072)` shape it selects `64x48x128`, a 256-thread
workgroup, and launches 264 workgroups. That output has only
`ceil(512/64)*ceil(768/48) = 128` logical tiles. The roughly 2.06x launch
expansion, together with the kernel's `GSUAMBSK` name, is direct evidence of
a global split/stream-K decomposition. Its traced kernel duration is
26.219 us; there is no separate transpose, cast, bias, or copy kernel.
Therefore split-K is now the evidence-backed next schedule to test for this
one runtime shape regime, rather than another unspecific tile sweep.

## D4. MFMA verification

The routed kernel is `_gemm_tiled_kernel`, whose K loop casts BF16 operands
to FP32 and calls scalar/vector `.fma`; it contains no matrix primitive. The
dispatcher makes this explicit:

- `_amd_dynamic_mfma_dispatch` is called only under
  `dtype == DType.float32`.
- BF16 falls through to `_gemm_tiled_kernel[DType.bfloat16,...]`.
- BF16 split-K is also gated off; only FP32 may set `ksplits > 1`.

The observed 3.055 TFLOPS on the worst exact-shape microbenchmark is 0.235%
of the stated 1.3 PFLOPS BF16 MFMA peak, consistent with VALU execution.

Both `rocprofv3` and legacy `rocprof` counter collection were attempted with
`SQ_INSTS_MFMA`, `SQ_INSTS_VALU`, `SQ_INSTS_VALU_MFMA_BF16`, and
`SQ_INSTS_VALU_FMA_F32`. Launch-under-profiler cannot initialize MAX's HSA
device (`HIP architecture query failed: Failed to initialize HSA runtime`),
and attach mode injects successfully but emits no counter records. Therefore
the counter result is unavailable because the profiler and this MAX runtime
cannot coexist in this environment. The trace identity, dispatcher gates,
kernel source, and ceiling result independently establish the no-MFMA route.

## D5. Routing check

The original routing defect is confirmed: the eager path selected a scalar
BF16 kernel even though the lower-level runtime-dimension MFMA building block
was available. Changes 1 and 2 fixed that routing and fused the bias. The
remaining gap is not another hidden route to a fast stock MI300X kernel.

The exact pinned Modular revision's direct AMD benchmark was run on gfx942 for
`M=512, N=768, K=3072`, BF16, cache-busting enabled. Its stock structured
`AMDMatmul` measured **172.554 us** (100 iterations, 14.001 TFLOP/s), versus
**68.064 us** for the accepted backend kernel and **42.131 us** for ROCm in
the canonical same-session harness. `AMDMatmul` pins a 256x256 output tile for
this BF16 invocation, producing only `ceil(512/256)*ceil(768/256) = 6`
workgroups on 304 CUs. It is therefore grid-starved and is not a route worth
selecting for transformer decode.

The newer structured ping-pong and four-wave implementations in this revision
are explicitly MI355X/CDNA4 kernels. Compiling the ping-pong BF16 path for
gfx942 fails its MMA constraint with `MMA shape requires CDNA4 or newer`; the
four-wave source likewise declares MI355X/CDNA4 and uses the same target. They
cannot be routed on MI300X. The benchmark's vendor branch also incorrectly
tries `cublasCreate_v2` on this ROCm host, so the ROCm comparison remains the
same-session `bench_gemm.py` measurement rather than that broken branch.

A direct stock MAX graph `ops.matmul` was also compiled for the exact
`[512,3072] x [3072,768]` BF16 shape. It aborted before execution with:

```
LLVM ERROR: Cannot select: intrinsic %llvm.amdgcn.fdot2.f32.bf16
```

Thus D5 is closed: after the accepted BF16 MFMA routing fix, the cheapest
remaining work is in the dynamic kernel's MI300X schedule/LDS layout. There is
no faster compatible stock MAX route to call, and N/K remain runtime values.

## Per-shape classification

| Shape (M,N,K) | Classification |
|---|---|
| 512,768,768 | no-MFMA routing; grid starvation; separate bias kernel |
| 512,768,3072 | no-MFMA routing; grid starvation; separate bias kernel |
| 512,2304,768 | no-MFMA routing; separate bias kernel |
| 512,3072,768 | no-MFMA routing; separate bias kernel |
| 4096 projection/MLP shapes | no-MFMA routing; grid is already filled |
| 512,50257,768 LM head | no-MFMA routing; grid is already filled; no extra kernel |
