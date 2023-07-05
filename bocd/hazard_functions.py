from abc import ABC, abstractmethod
import numpy as np

class BaseHazardFunction(ABC):
    @abstractmethod
    def __call__(self, t: float):
        raise NotImplementedError
    
class ExponentialHazardFunction(BaseHazardFunction):
    def __init__(self, scale: float):
        self.scale = scale  # mean time to event

    def __call__(self, t: float):
        # hazard
        return self.scale**-1
    

class WeibullHazardFunction(BaseHazardFunction):
    # https://en.wikipedia.org/wiki/Discrete_Weibull_distribution
    def __init__(self, shape, scale):
        self.scale = scale
        self.shape = shape
        
    def __call__(self, t: float):
        return (
            np.exp(
                (
                    (t+1) / self.scale
                )**self.shape
                - (
                    t / self.scale
                )**self.shape
            )
            -1
        )