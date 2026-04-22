import numpy as np
from max.driver import Accelerator
from max.experimental import functional as F
from max.experimental.tensor import Tensor

x = Tensor(
    np.array([0.0, 1.0, 4.0, 9.0, 16.0], dtype=np.float64), device=Accelerator(0)
)

y = F.cos(x)

print(y)
