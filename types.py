from typing import TypeAlias, TypedDict
import torch

Latent: TypeAlias = TypedDict('Latent', {'samples': torch.Tensor})
BoundingBox: TypeAlias = tuple[torch.Tensor, torch.Tensor]
Dimensions: TypeAlias = tuple[int, int]
