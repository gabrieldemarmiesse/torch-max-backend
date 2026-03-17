import os

IS_RUNNING_TESTS = os.environ.get("TORCH_MAX_BACKEND_TESTING", "0") == "1"
