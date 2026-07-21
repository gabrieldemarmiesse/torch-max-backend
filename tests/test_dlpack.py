"""Focused tests for the pure-Python DLPack ownership bridge."""

import subprocess
import sys
import textwrap


def test_consumed_capsule_survives_dlpack_module_reload():
    """A live consumer, not a replaceable module registry, owns its export."""
    script = textwrap.dedent(
        """
        import gc
        import importlib
        import weakref

        import torch
        from max.driver import CPU
        from max.dtype import DType
        from torch_mojo_backend.mojo_device import dlpack


        class Holder:
            pass


        source = torch.arange(16, dtype=torch.float32)
        expected = source.clone()
        holder = Holder()
        holder.source = source
        holder_ref = weakref.ref(holder)
        capsule = dlpack.make_capsule(
            holder, source.data_ptr(), source.shape, DType.float32, CPU()
        )
        result = torch.from_dlpack(capsule)
        del capsule, holder, source
        gc.collect()

        unconsumed_source = torch.arange(4, dtype=torch.int32)
        unconsumed_holder = Holder()
        unconsumed_holder.source = unconsumed_source
        unconsumed_holder_ref = weakref.ref(unconsumed_holder)
        unconsumed_capsule = dlpack.make_capsule(
            unconsumed_holder,
            unconsumed_source.data_ptr(),
            unconsumed_source.shape,
            DType.int32,
            CPU(),
        )
        del unconsumed_holder, unconsumed_source

        importlib.reload(dlpack)
        gc.collect()
        assert holder_ref() is not None
        assert unconsumed_holder_ref() is not None
        torch.testing.assert_close(result, expected)

        del result
        del unconsumed_capsule
        gc.collect()
        assert holder_ref() is None
        assert unconsumed_holder_ref() is None
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, (
        f"subprocess exited with {completed.returncode}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
