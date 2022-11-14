from typing import Callable, Optional, Tuple

import torch

ActivationFn = Callable[[torch.Tensor], torch.Tensor]
ActivationFnArgs = Tuple[Optional[ActivationFn], Optional[ActivationFn]]

VectorTuple = Tuple[torch.Tensor, torch.Tensor]
VectorTupleDim = Tuple[int, int]
VectorOptionalTuple = Tuple[torch.Tensor, Optional[torch.Tensor]]
