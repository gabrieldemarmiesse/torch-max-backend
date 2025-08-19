#!/usr/bin/env python3

import torch
from torch_max_backend import max_backend
from torch._dynamo.utils import counters
import logging

# Enable logging for cache debugging - focus on cache keys
logging.basicConfig(level=logging.INFO)
autograd_cache_logger = logging.getLogger(
    "torch._functorch._aot_autograd.autograd_cache"
)
autograd_cache_logger.setLevel(logging.INFO)


def simple_model(x):
    return torch.matmul(x, x.transpose(-2, -1))


def test_caching():
    print("Testing AOTAutograd caching with Max backend...")

    # Clear counters
    counters.clear()

    # Create test input
    x = torch.randn(4, 8, 16, device="cuda")

    # Compile the model
    compiled_model = torch.compile(simple_model, backend=max_backend)

    print("\n=== First run (should be cache miss) ===")
    result1 = compiled_model(x)
    print(f"Cache hits: {counters['aot_autograd']['autograd_cache_hit']}")
    print(f"Cache misses: {counters['aot_autograd']['autograd_cache_miss']}")
    print(f"Cache bypasses: {counters['aot_autograd']['autograd_cache_bypass']}")
    print(f"Cache saves: {counters['aot_autograd']['autograd_cache_saved']}")

    print("\n=== Second run (should be cache hit) ===")
    # Clear dynamo to force recompilation but allow cache lookup
    torch._dynamo.reset()
    compiled_model2 = torch.compile(simple_model, backend=max_backend)
    result2 = compiled_model2(x)
    print(f"Cache hits: {counters['aot_autograd']['autograd_cache_hit']}")
    print(f"Cache misses: {counters['aot_autograd']['autograd_cache_miss']}")
    print(f"Cache bypasses: {counters['aot_autograd']['autograd_cache_bypass']}")
    print(f"Cache saves: {counters['aot_autograd']['autograd_cache_saved']}")

    print(f"\nResults match: {torch.allclose(result1, result2)}")


if __name__ == "__main__":
    test_caching()
