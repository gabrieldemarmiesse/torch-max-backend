# AMD GEVM launches four times too many blocks and writes out of bounds

## Environment

- AMD MI300X (`gfx942`), ROCm 7.2
- Mojo `1.0.0b3.dev2026061806`
- MAX `26.5.0.dev2026061806`
- Reproducer is pure Mojo; PyTorch is not involved

## Reproducer

Run the adjacent 32-line reproducer:

```bash
uv run mojo run --target-accelerator gfx942 \
  reproducers/amd_gevm_grid_oob.mojo
```

The output allocation has an `N`-element logical view followed by a `3*N`
guard. The input allocation is also padded, making the invalid accesses
deterministic rather than dependent on allocator placement.

## Observed on MI300X

```text
C[0] = 64.0 ; guard C[N] = 64.0
AssertionError: GEVM wrote beyond its 1xN output
```

All 192 guard elements are overwritten. The same program passes on H100,
where `guard C[N]` remains `-7.0`.

## Root cause

`gemv_gpu_dispatch` defines `WARPS_PER_BLOCK = 1024 / WARP_SIZE` and launches
GEVM with `grid_dim=ceildiv(n, WARPS_PER_BLOCK)`. However, `gevm_kernel`
produces `WARP_SIZE` output columns per block.

- NVIDIA: `WARP_SIZE=32`, `WARPS_PER_BLOCK=32`; the formula works by accident.
- AMD: `WARP_SIZE=64`, `WARPS_PER_BLOCK=16`; it launches 4x too many blocks.

The GEVM grid should use `ceildiv(n, WARP_SIZE)`. A regression test should use
a padded output and verify that every guard element remains unchanged.
