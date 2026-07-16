import os

POSITIVE_VALUES = ("1", "true", "yes")


def profiling_enabled():
    """
    Check if profiling is enabled by looking for the environment variable.
    """
    x = os.environ.get("TORCH_MAX_BACKEND_PROFILE", "0").lower()
    py_x = os.environ.get("PYTORCH_MAX_BACKEND_PROFILE", "0").lower()

    return x in POSITIVE_VALUES or py_x in POSITIVE_VALUES


def verbose_enabled():
    """
    Check if verbose mode is enabled by looking for the environment variable.
    """
    x = os.environ.get("TORCH_MAX_BACKEND_VERBOSE", "0").lower()
    py_x = os.environ.get("PYTORCH_MAX_BACKEND_VERBOSE", "0").lower()

    return x in POSITIVE_VALUES or py_x in POSITIVE_VALUES


def fast_eager_enabled():
    """
    Check if the Mojo-extension fast path for max_device eager mode is
    enabled. Defaults to on; set TORCH_MAX_BACKEND_FAST_EAGER=0 to use the
    graph-based eager path for every op.
    """
    x = os.environ.get("TORCH_MAX_BACKEND_FAST_EAGER", "1").lower()

    return x in POSITIVE_VALUES


def debug_graph():
    """
    Check if graph debugging is enabled by looking for the environment variable.
    """
    x = os.environ.get("TORCH_MAX_BACKEND_DEBUG_GRAPH", "0").lower()
    py_x = os.environ.get("PYTORCH_MAX_BACKEND_DEBUG_GRAPH", "0").lower()

    return x in POSITIVE_VALUES or py_x in POSITIVE_VALUES
