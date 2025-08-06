import torch
from src.max_torch_backend import my_compiler

print("imports done")


def fn(x, y, z):
    return x + y + z, x + torch.abs(z) - torch.cos(y) + 1


fn_compiled = torch.compile(backend=my_compiler)(fn)


a = torch.randn(3).to(device="cuda")
print(a)
b = torch.randn(3).to(device="cuda")
print(b)
c = torch.randn(3).to(device="cuda")
print(c)

outputs_no_compiled = fn(a, b, c)
outputs_compiled = fn_compiled(a, b, c)
print("out_no_compiled:", outputs_no_compiled)
print("out_   compiled:", outputs_compiled)
for out, out_compiled in zip(outputs_no_compiled, outputs_compiled):
    assert torch.allclose(out, out_compiled), "Outputs do not match!"
    assert out.device == out_compiled.device, "Devices do not match!"