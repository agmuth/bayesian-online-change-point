from abc import ABC, abstractmethod


class BaseHazardFunction(ABC):
    @abstractmethod
    def __call__(self, t: float):
        raise NotImplementedError
    
class ExponentialHazardFunction(BaseHazardFunction):
    def __init__(self, scale: float):
        self.scale = scale  # mean time to event

    def __call__(self, t: float):
        return self.scale**-1
    
