from typing import List, Optional, TypedDict  # type: ignore


class EarlyStoppingParams(TypedDict):
    monitor: str
    tol: float
    patience: int
    restore_best_weights: bool

class BetaSchedulerParams(TypedDict):
    warmup_epochs: int
    delay_epochs: int