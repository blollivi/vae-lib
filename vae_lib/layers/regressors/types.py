from typing import TypedDict, Optional

from vae_lib.layers.types import MLPParams, LinearParams


class DeepRegressorParams(TypedDict):
    output_dim: int
    mlp_params: Optional[MLPParams]
    linear_params: LinearParams