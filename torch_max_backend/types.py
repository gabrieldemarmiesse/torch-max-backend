from max.graph import Dim, TensorValue
from max.tensor import Tensor as MaxEagerTensor

MaxTensor = TensorValue | MaxEagerTensor
Scalar = int | float | Dim
SymIntType = int | Dim
