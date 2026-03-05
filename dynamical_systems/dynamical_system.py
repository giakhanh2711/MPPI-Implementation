from abc import ABC, abstractmethod


class DynamicalSystem(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def state_transition(self, x, u, dt):
        pass

    @abstractmethod
    def cost_at_state(self, x):
        pass