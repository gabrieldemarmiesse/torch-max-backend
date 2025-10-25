from max.experimental.tensor import Tensor as MaxEagerTensor
from max.graph import TensorValue

MaxTensor = TensorValue | MaxEagerTensor
