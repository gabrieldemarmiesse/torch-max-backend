"""Shared comptime constants for the FA4-target bwd kernels.

FA4's sm90 bwd config for head_dim 128 non-causal: tile_m=80,
tile_n=128 with SdP_swapAB + dQ_swapAB (see reference_ptx/README.md).
Matched exactly: S^T/dP^T/dQ^T are m64n80k16, dV/dK are m64n128k16
with 5 k-steps, Q/dO double-buffered (the 80-row slots blow the
232 KiB smem cap at 3 stages — and 2 is what FA4 ships).
Seqlen need not divide 80: side buffers are padded to
Spad = ceil(S/80)*80 with lse=+inf / dpsum=0 pad rows, which force
the tail m-tile's P and dS to exact zeros.
"""

comptime kBwdBlockM: Int = 80  # Q rows per inner tile (FA4 tile_m)
comptime kBwdBlockN: Int = 128  # KV rows per block


def kBwdTileM(head_dim: Int, causal: Bool) -> Int:
    """FA4's bwd tile_m: hdim64 = 128 (both causal and non-causal,
    BwdConfig in interface.py:177-184); hdim128 = 64 causal / 80
    non-causal."""
    if head_dim == 64:
        return 128
    return 64 if causal else kBwdBlockM
comptime kBwdNMmaWarpgroups: Int = 2
comptime kBwdNThreads: Int = (kBwdNMmaWarpgroups + 1) * 128

# Q/dO shared-memory ring: Q(m) in slot (2m) % kBwdQdOStages, dO(m)
# in (2m+1) % kBwdQdOStages (i.e. 2 stages each, FA4's config).
comptime kBwdQdOStages: Int = 4

# Preprocess / convert kernels: one Q-row per thread, 128 threads.
comptime kBwdPreBlockM: Int = 128
comptime kBwdPreThreads: Int = 128
# Convert kernel runs 256 threads (one 64-elem dq half-row each).
comptime kBwdCvtThreads: Int = 256
