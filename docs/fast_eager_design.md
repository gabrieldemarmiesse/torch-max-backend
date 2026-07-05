# Fast eager mode via Mojo Python extensions (proof of concept)

## Why

The max_device eager mode routes every ATen op through
`max.experimental.tensor`: each call builds a fresh MLIR graph, runs
MLIR cleanup passes, then interprets or compiles it. Measured on an
RTX 2000 Ada (WSL2), that costs **~2,200 µs per op call** regardless of
tensor size — a (64, 64) add costs the same as a (1024, 1024) one, all
fixed Python/MLIR overhead. Profiling one `add`: ~1 ms graph building,
~0.9 ms `finalize_graph` (MLIR passes), ~0.6 ms dtype promotion; the
actual kernel is noise. For reference, torch's own CUDA eager launch
overhead on the same machine is ~21 µs.

## What this PoC does

It takes the architecture of
[causal-conv1d-mojo](https://github.com/gabrieldemarmiesse/causal-conv1d-mojo)
— Mojo kernels compiled to CPython extension `.so` files, JIT-on-first-use
with a content-addressed disk cache — in the productized form that MAX
itself already uses internally for its eager interpreter
(`max/_interpreter_ops/*.mojo` + `mojo.importer`):

- `torch_max_backend/eager_kernels/elementwise_ops.mojo` implements
  Add/Sub/Mul/Div/Max/Min/Relu/Exp as Mojo kernels over **contiguous
  buffers with fully dynamic shapes**. Dtype dispatch happens at
  **runtime inside Mojo** (every dtype specialization is compiled into
  the one extension), so one `.so` serves every shape and dtype with
  zero recompilation.
- The module is imported through `mojo.importer` (official Mojo import
  hook): first import runs `mojo build --emit shared-lib`, caches under
  `__mojocache__/elementwise_ops.hash-<h>.so`, and every later import
  (any process) just dlopens it. The import itself is **deferred to the
  first fast-path op call** (`_eager_impl` in `max_device_aten_ops.py`),
  so `import torch_max_backend` and torch.compile-only workloads never
  pay the compile; a process that never touches max_device eager mode
  compiles nothing.
- Python-visible functions receive the `max.driver.Buffer` objects
  directly plus `device._device_context_ptr()`. The kernel is enqueued
  on **MAX's own DeviceContext** (same device queue the MAX driver
  uses), so ordering with copies/other MAX work needs no extra
  synchronization.
- `eager_kernels/aten_fast.py` wraps these kernels with
  ATen-compatible signatures. When inputs qualify (realized,
  contiguous, same shape/dtype/device, alpha == 1, ...) the op is one
  extension call; otherwise it **falls back to the existing
  `aten_functions` implementation**, so behavior is unchanged for
  everything the fast path doesn't cover.
- `max_device_aten_ops.py` registers the fast versions for
  add/sub/mul/div/maximum/minimum/relu/exp, gated by
  `TORCH_MAX_BACKEND_FAST_EAGER` (default on).
  **The torch.compile backend is untouched.**

## Measured results (RTX 2000 Ada, WSL2)

Per-op call, (64, 64) float32 on the GPU max_device:

| path | µs/call |
|---|---|
| current graph-based eager | ~2,200 |
| fast path, bare extension call (launch only) | 6.5 |
| fast path, incl. out-alloc + wrap (MaxEagerTensor level) | ~20 |
| fast path, end-to-end `x + y` at the torch level | ~44 |
| torch native CUDA eager (reference) | ~21 |

End-to-end speedup today: **~50×**; the remaining ~24 µs over the bare
call is the PyTorch dispatcher + `TorchMaxTensor` wrapping + beartype,
which can be shaved independently.

## Scaling analysis ("how many extensions is too many?")

Measured on this machine:

- **Loading**: first extension load pays ~55 ms (shared Mojo runtime
  libraries); each additional extension costs **~0.07 ms and ~0.1 MB
  RSS** (1.5 MB `.so`, lazily mapped). 300 extensions would load in
  well under 100 ms total — loading is a non-issue.
- **Size**: ~0.15 MB of `.so` per op (all dtypes, CPU+GPU kernels).
  Full ATen coverage (~200 ops) ≈ 30–60 MB. For calibration, MAX's own
  interpreter op set ships 58 MB of prebuilt extensions in the wheel.
- **Compile time** (the real cost): ~10 s fixed per module + ~2.3 s
  per op (op = ~10 kernel instantiations: 5 dtypes × CPU+GPU). An
  8-op module cold-compiles in ~30 s. Editing a module recompiles the
  whole module (~30–40 s). `mojo build` keeps its own internal cache,
  so rebuilding previously-seen content is several times faster.

Consequences for the design:

- **Compile on first use, but at module granularity, not per-variant.**
  causal-conv1d-mojo compiles one `.so` per comptime config on first
  use because its variant space is combinatorial (dtype × width × 8
  bools) and sparsely used. Our variant space is the opposite — small
  (op × dtype) and dense — and the fixed cost dominates: a single
  (op, dtype, GPU-only) variant compiles in **8.5 s** on this machine,
  while the 8-op × 9-dtype × CPU+GPU module compiles ~150 kernel
  instantiations in 31 s. Per-variant lazy compilation would multiply
  total compile time ~40× (full coverage: hours instead of ~10 min)
  and re-pay ~8.5 s every time a new (op, dtype) pair first appears.
  First-use compilation at module granularity (what `_eager_impl`'s
  lazy import does) keeps the lazy behavior with none of that cost.
- **Don't do one extension per op** (200 modules × 10 s fixed ≈ +30 min
  of avoidable fixed compile cost, and 200 files to manage). Group ops
  by category into modules of ~10–20 ops (elementwise_binary,
  elementwise_unary, reductions, matmul, data_movement, ...) exactly
  like `max/_interpreter_ops` does (~25 modules). Full cold build of
  ~20 modules ≈ 10 min sequential, parallelizable across modules. With
  multiple category modules, import each lazily on the first call of an
  op in that category, so a given workload only compiles the categories
  it actually uses.
- **Don't do one extension per dtype** — dtype dispatch at runtime in
  Mojo costs one branch chain per call (~ns) and collapses the
  extension count by ~10×.
- Editing one op costs one module rebuild (~30–60 s), which argues for
  keeping modules from growing past ~20 ops.
- Ship prebuilt `__mojocache__/` in release wheels (as MAX does) so
  users never compile; CI caches it keyed on the `.mojo` content hash
  + toolchain version (`mojo.importer` handles invalidation).

## Follow-up work (not in this PoC)

- Scalar (tensor ⊕ python number) variants — one extra kernel per op
  family, removes the most common fallback.
- Broadcasting: either fall back (today), or add a stride-aware kernel
  (see `ManagedTensorSlice`/`elementwise` with `IndexList` shapes, or
  expanded strides with 0-stride broadcast dims, MAX_RANK-bounded like
  MAX's interpreter uses `MAX_RANK = 5`).
- Matmul & friends: `linalg`/`nn` Mojo packages (the same kernel
  library the MAX graph compiler uses, including cuBLAS bindings) are
  importable from these modules — coverage does not require rewriting
  kernels.
- In-place variants (`add_`, `relu_`) are trivial: write to the input
  buffer.
- First-import compile UX for the test suite: `mojo.importer` has no
  cross-process lock; warm the cache once (e.g. in
  `scripts/populate_cache_for_tests.py`) before `pytest -n`.
- Shave the remaining per-call overhead (TorchMaxTensor creation goes
  through a meta tensor + `__init__` per op; beartype on hot wrappers).
