import numpy as np
import torch
from torch.nn import functional as F
from max_torch_backend import MaxCompilerBackpropCompatible

device = "cpu"


class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)

    def forward(self, x):
        return self.linear(x)


model = MyModel().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


def train_step(x, y):
    model.train()
    optimizer.zero_grad()
    output = model(x)
    loss = F.mse_loss(output, y)
    loss.backward()
    optimizer.step()
    return loss


a = torch.randn(5, 3).to(device)
b = torch.randn(5, 2).to(device)

# We need to reset the parameters before each test
# to check the model weights afterwards
model.linear.weight.data.fill_(0.01)
model.linear.bias.data.fill_(0.01)

loss_not_compiled = train_step(a, b).cpu().detach().numpy()
weight_not_compiled = model.linear.weight.data.cpu().numpy()
bias_not_compiled = model.linear.bias.data.cpu().numpy()

model.linear.weight.data.fill_(0.01)
model.linear.bias.data.fill_(0.01)

loss_compiled = (
    torch.compile(backend=MaxCompilerBackpropCompatible)(train_step)(a, b)
    .cpu()
    .detach()
    .numpy()
)
weight_compiled = model.linear.weight.data.cpu().numpy()
bias_compiled = model.linear.bias.data.cpu().numpy()

np.testing.assert_allclose(loss_not_compiled, loss_compiled, rtol=5e-2, atol=5e-3)
np.testing.assert_allclose(weight_not_compiled, weight_compiled, rtol=5e-2, atol=5e-3)
np.testing.assert_allclose(bias_not_compiled, bias_compiled, rtol=5e-2, atol=5e-3)
