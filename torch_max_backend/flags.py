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
