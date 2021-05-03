from pathlib import Path
from typing import Callable, Tuple, Union

from tensorflow import Tensor

# files n directories
FilePath = Union[str, Path]
Dir = Union[str, Path]

# Tensors and arrays
Tensor2Tensor = Callable[[Tensor], Tensor]
Tensor2TensorTuple = Callable[[Tensor], Tuple[Tensor]]

# Image features
ImageFeatures = Union[Tensor, Tuple[Tensor]]

# Loss
Loss = Callable[[Tensor, Tensor], Tensor]
