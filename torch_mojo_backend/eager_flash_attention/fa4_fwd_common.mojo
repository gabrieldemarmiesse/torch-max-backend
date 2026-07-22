"""Shared comptime constants for the FA4-target fwd kernel.

Mirrors Tri Dao FA4's sm90 configs (see reference_ptx/README.md and
reference_ptx/hdim64_port_spec.md):
- head_dim 128: tile 128x128, 2 MMA warpgroups + 1 producer, 384
  threads, setmaxnreg 24/240.
- head_dim 64: tile 192x128, 3 MMA warpgroups + 1 producer, 512
  threads, setmaxnreg 32/160 (the pool 3*128*160 + 128*32 = 65536 is
  exactly the register file).
All are comptime functions of head_dim; the hdim128 evaluations must
keep the original kernels' PTX byte-identical.
"""


def kFa4BlockM(head_dim: Int) -> Int:
    return 192 if head_dim == 64 else 128


comptime kFa4BlockN: Int = 128


def kFa4NMmaWarpgroups(head_dim: Int) -> Int:
    return 3 if head_dim == 64 else 2


def kFa4NThreads(head_dim: Int) -> Int:
    return (kFa4NMmaWarpgroups(head_dim) + 1) * 128


def kFa4ProducerRegs(head_dim: Int) -> Int:
    return 32 if head_dim == 64 else 24


def kFa4ConsumerRegs(head_dim: Int) -> Int:
    return 160 if head_dim == 64 else 240


# K/V shared-memory ring: K(n) lives in slot (2n) % kFa4KVStages,
# V(n) in slot (2n+1) % kFa4KVStages.
comptime kFa4KVStages: Int = 6
