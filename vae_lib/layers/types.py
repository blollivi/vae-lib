from typing import List, Optional, TypedDict  # type: ignore


class MLPParams(TypedDict):
    hidden_units: List[int]
    l1_kernel: Optional[float]
    l2_kernel: Optional[float]
    dropout_rate: Optional[float]


class LinearParams(TypedDict):
    output_dim: int
    l1_kernel: Optional[float]
    l2_kernel: Optional[float]







