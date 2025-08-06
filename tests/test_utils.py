import torch

from src.max_torch_backend.utils import get_the_number_of_outputs


class TestUtils:
    def test_single_output(self):
        def fn(x):
            return x + 1

        gm = torch.fx.symbolic_trace(fn)
        num_outputs = get_the_number_of_outputs(gm)
        assert num_outputs == 1

    def test_multiple_outputs(self):
        def fn(x):
            return x + 1, x * 2, x - 1

        gm = torch.fx.symbolic_trace(fn)
        num_outputs = get_the_number_of_outputs(gm)
        assert num_outputs == 3

    def test_two_outputs(self):
        def fn(x, y):
            return x + y, x - y

        gm = torch.fx.symbolic_trace(fn)
        num_outputs = get_the_number_of_outputs(gm)
        assert num_outputs == 2
