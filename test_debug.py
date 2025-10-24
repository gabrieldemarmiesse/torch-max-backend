"""Debug test for convolution_backward"""

import torch
from torch.ops import aten
from torch_max_backend import max_backend

# Test grad_input only
def test_grad_input_only():
    print("Testing grad_input computation...")

    # Setup
    batch_size, in_channels, height, width = 2, 3, 8, 8
    out_channels, kernel_h, kernel_w = 4, 3, 3

    input_tensor = torch.randn(batch_size, in_channels, height, width, dtype=torch.float32, device='cpu')
    weight = torch.randn(out_channels, in_channels, kernel_h, kernel_w, dtype=torch.float32, device='cpu')

    # Forward pass to get grad_output shape
    output = torch.nn.functional.conv2d(input_tensor, weight, stride=1, padding=0)
    print(f"Forward output shape: {output.shape}")

    grad_output = torch.randn_like(output)

    # PyTorch reference
    print("\nComputing with PyTorch (eager)...")
    grad_input_ref, grad_weight_ref, grad_bias_ref = aten.convolution_backward(
        grad_output, input_tensor, weight,
        None,  # no bias
        [1, 1],  # stride
        [0, 0],  # padding
        [1, 1],  # dilation
        False,  # not transposed
        [0, 0],  # output_padding
        1,  # groups
        [True, False, False],  # only compute grad_input
    )
    print(f"PyTorch grad_input shape: {grad_input_ref.shape if grad_input_ref is not None else None}")
    print(f"PyTorch grad_weight: {grad_weight_ref}")
    print(f"PyTorch grad_bias: {grad_bias_ref}")

    # MAX backend
    print("\nCompiling with MAX backend...")

    @torch.compile(backend=max_backend)
    def conv_backward_grad_input(grad_out, inp, wgt):
        result = aten.convolution_backward(
            grad_out, inp, wgt,
            None,  # no bias
            [1, 1],  # stride
            [0, 0],  # padding
            [1, 1],  # dilation
            False,  # not transposed
            [0, 0],  # output_padding
            1,  # groups
            [True, False, False],  # only compute grad_input
        )
        # Extract just grad_input since others are None
        return result[0]

    grad_input_max = conv_backward_grad_input(grad_output, input_tensor, weight)
    print(f"MAX grad_input shape: {grad_input_max.shape}")

    # Compare
    print("\nComparing results...")
    print(f"Shape match: {grad_input_ref.shape == grad_input_max.shape}")

    max_diff = torch.abs(grad_input_ref - grad_input_max).max().item()
    print(f"Max absolute difference: {max_diff}")

    if max_diff < 1e-4:
        print("✓ Test PASSED!")
        return True
    else:
        print("✗ Test FAILED - difference too large")
        return False

def test_grad_bias_only():
    print("\n" + "="*60)
    print("Testing grad_bias computation...")

    # Setup
    batch_size, in_channels, height, width = 2, 3, 8, 8
    out_channels, kernel_h, kernel_w = 4, 3, 3

    input_tensor = torch.randn(batch_size, in_channels, height, width, dtype=torch.float32, device='cpu')
    weight = torch.randn(out_channels, in_channels, kernel_h, kernel_w, dtype=torch.float32, device='cpu')

    # Forward pass to get grad_output shape
    output = torch.nn.functional.conv2d(input_tensor, weight, stride=1, padding=0)
    grad_output = torch.randn_like(output)

    # PyTorch reference
    print("\nComputing with PyTorch (eager)...")
    grad_input_ref, grad_weight_ref, grad_bias_ref = aten.convolution_backward(
        grad_output, input_tensor, weight,
        [out_channels],  # bias_sizes
        [1, 1],  # stride
        [0, 0],  # padding
        [1, 1],  # dilation
        False,  # not transposed
        [0, 0],  # output_padding
        1,  # groups
        [False, False, True],  # only compute grad_bias
    )
    print(f"PyTorch grad_input: {grad_input_ref}")
    print(f"PyTorch grad_weight: {grad_weight_ref}")
    print(f"PyTorch grad_bias shape: {grad_bias_ref.shape if grad_bias_ref is not None else None}")

    # MAX backend
    print("\nCompiling with MAX backend...")

    @torch.compile(backend=max_backend)
    def conv_backward_grad_bias(grad_out, inp, wgt):
        result = aten.convolution_backward(
            grad_out, inp, wgt,
            [out_channels],  # bias_sizes
            [1, 1],  # stride
            [0, 0],  # padding
            [1, 1],  # dilation
            False,  # not transposed
            [0, 0],  # output_padding
            1,  # groups
            [False, False, True],  # only compute grad_bias
        )
        return result[2]  # Extract grad_bias

    grad_bias_max = conv_backward_grad_bias(grad_output, input_tensor, weight)
    print(f"MAX grad_bias shape: {grad_bias_max.shape}")

    # Compare
    print("\nComparing results...")
    print(f"Shape match: {grad_bias_ref.shape == grad_bias_max.shape}")

    max_diff = torch.abs(grad_bias_ref - grad_bias_max).max().item()
    print(f"Max absolute difference: {max_diff}")

    if max_diff < 1e-4:
        print("✓ Test PASSED!")
        return True
    else:
        print("✗ Test FAILED - difference too large")
        return False


def test_grad_input_and_bias():
    print("\n" + "="*60)
    print("Testing grad_input AND grad_bias together...")

    # Setup
    batch_size, in_channels, height, width = 2, 3, 8, 8
    out_channels, kernel_h, kernel_w = 4, 3, 3

    input_tensor = torch.randn(batch_size, in_channels, height, width, dtype=torch.float32, device='cpu')
    weight = torch.randn(out_channels, in_channels, kernel_h, kernel_w, dtype=torch.float32, device='cpu')

    # Forward pass
    output = torch.nn.functional.conv2d(input_tensor, weight, stride=1, padding=0)
    grad_output = torch.randn_like(output)

    # PyTorch reference
    print("\nComputing with PyTorch (eager)...")
    grad_input_ref, grad_weight_ref, grad_bias_ref = aten.convolution_backward(
        grad_output, input_tensor, weight,
        [out_channels],  # bias_sizes
        [1, 1],  # stride
        [0, 0],  # padding
        [1, 1],  # dilation
        False,  # not transposed
        [0, 0],  # output_padding
        1,  # groups
        [True, False, True],  # compute grad_input and grad_bias
    )

    # MAX backend
    print("Compiling with MAX backend...")

    @torch.compile(backend=max_backend)
    def conv_backward_both(grad_out, inp, wgt):
        return aten.convolution_backward(
            grad_out, inp, wgt,
            [out_channels],  # bias_sizes
            [1, 1],  # stride
            [0, 0],  # padding
            [1, 1],  # dilation
            False,  # not transposed
            [0, 0],  # output_padding
            1,  # groups
            [True, False, True],  # compute grad_input and grad_bias
        )

    result_max = conv_backward_both(grad_output, input_tensor, weight)
    grad_input_max = result_max[0]
    grad_bias_max = result_max[2]

    # Compare
    print("\nComparing results...")
    print(f"grad_input shapes match: {grad_input_ref.shape == grad_input_max.shape}")
    print(f"grad_bias shapes match: {grad_bias_ref.shape == grad_bias_max.shape}")

    max_diff_input = torch.abs(grad_input_ref - grad_input_max).max().item()
    max_diff_bias = torch.abs(grad_bias_ref - grad_bias_max).max().item()
    print(f"grad_input max absolute difference: {max_diff_input}")
    print(f"grad_bias max absolute difference: {max_diff_bias}")

    if max_diff_input < 1e-4 and max_diff_bias < 1e-4:
        print("✓ Test PASSED!")
        return True
    else:
        print("✗ Test FAILED - difference too large")
        return False


if __name__ == "__main__":
    passed = 0
    total = 3

    if test_grad_input_only():
        passed += 1
    if test_grad_bias_only():
        passed += 1
    if test_grad_input_and_bias():
        passed += 1

    print("\n" + "="*60)
    print(f"Summary: {passed}/{total} tests passed")
    if passed == total:
        print("✓ All tests PASSED!")
    else:
        print("✗ Some tests FAILED")
