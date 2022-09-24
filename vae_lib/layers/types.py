from typing import List, Optional, TypedDict  # type: ignore


class MLPParams(TypedDict):
    hidden_units: List[int]
    l1_kernel: Optional[float]
    l2_kernel: Optional[float]
    dropout_rate: Optional[float]


class LinearParams(TypedDict):
    output_dim: int
    l1_kernel: Optional[float]


class DeepRegressorParams(TypedDict):
    output_dim: int
    mlp_params: Optional[MLPParams]
    linear_params: LinearParams


class SparseMappingParams(TypedDict):
    input_dim: int
    output_dim: int
    lambda0: float
    lambda1: float
    lambda0_step: float
    a: float
    b: float


class DeepGaussianDistributionParams(TypedDict):
    output_dim: int
    mu_regressor_params: DeepRegressorParams
    logvar_regressor_params: DeepRegressorParams


class DeepSpikeSlabDistributionParams(TypedDict):
    output_dim: int
    mu_regressor_params: DeepRegressorParams
    logvar_regressor_params: DeepRegressorParams
    logspike_regressor_params: DeepRegressorParams


class DeepSparseGaussianDistributionParams(TypedDict):
    output_dim: int
    mu_regressor_params: DeepRegressorParams
    logvar_regressor_params: DeepRegressorParams
    sparse_mapping_params: SparseMappingParams


class DeepSparseRegressorParams(TypedDict):
    output_dim: int
    mlp_params: Optional[MLPParams]
    linear_params: LinearParams
    sparse_mapping_params: SparseMappingParams
