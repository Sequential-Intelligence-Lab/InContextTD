from abc import ABC, abstractmethod
from typing import Tuple


# base class for Markov Reward Process
class MRP(ABC):
    def __init__(self, n_states) -> None:
        super().__init__()
        self.n_states = n_states

    @abstractmethod
    def reset(self) -> int:
        pass

    @abstractmethod
    def step(self, state: int) -> Tuple[int, float]:
        pass
