from typing import TypedDict

from vae_lib.layers.regressors.types import DeepRegressorParams


class DeepGaussianDistributionParams(TypedDict):
    output_dim: int
    mean_regressor_params: DeepRegressorParams
    logvar_regressor_params: DeepRegressorParams


class DeepSpikeSlabDistributionParams(TypedDict):
    output_dim: int
    mean_regressor_params: DeepRegressorParams
    logvar_regressor_params: DeepRegressorParams
    logspike_regressor_params: DeepRegressorParams


class SparseMappingParams(TypedDict):
    input_dim: int
    output_dim: int
    lambda0: float
    lambda1: float
    lambda0_step: float
    a: float
    b: float


class DeepSparseGaussianDistributionParams(TypedDict):
    output_dim: int
    mean_regressor_params: DeepRegressorParams
    logvar_regressor_params: DeepRegressorParams
    sparse_mapping_params: SparseMappingParams


class RecurrentGaussianDistributionParams(TypedDict):
    output_dim: int
    gaussian_regressor_params: DeepGaussianDistributionParams
    n_flow_layers: int
