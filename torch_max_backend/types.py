from max.experimental.tensor import Tensor as MaxEagerTensor
from max.graph import Dim, TensorValue

MaxTensor = TensorValue | MaxEagerTensor
Scalar = int | float | Dim
SymIntType = int | Dim
