import numpy as np 

from abc import ABC, abstractmethod

# https://en.wikipedia.org/wiki/Conjugate_prior

class BaseConjugatePrior(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def update(self, x_new, *args, **kwargs):
        raise NotImplementedError
    

class BetaPrior(BaseConjugatePrior):
    def __init__(self, shape1_prior: np.ndarray, shape2_prior: np.ndarray):
        self.shape1_prior = np.array(shape1_prior)
        self.shape2_prior = np.array(shape2_prior)
        self.shape1_posterior = np.array(shape1_prior)
        self.shape2_posterior = np.array(shape2_prior)
        
           
    def update(self, x_new: np.ndarray, *args, **kwargs):
        # bernoullie update
        self.shape1_posterior += x_new
        self.shape2_posterior += (1 - x_new)


class GammaConjugatePrior(BaseConjugatePrior):
    def __init__(self, shape_prior: np.ndarray, rate_prior: np.ndarray):
        self.shape_prior = np.array(shape_prior)    
        self.rate_prior = np.array(rate_prior)   
        self.shape_posterior = np.array(shape_prior)    
        self.rate_posterior = np.array(rate_prior)
        
    def update(self, x_new: np.ndarray, *args, **kwargs):
        # poisson update
        self.shape_posterior += x_new
        self.rate_posterior += 1


class NormalGammaConjugatePrior(BaseConjugatePrior):
    def __init__(self, mean_prior: np.ndarray, prec_prior: np.ndarray, shape_prior: np.ndarray, rate_prior: np.ndarray):
        self.mean_prior = np.array(mean_prior)    
        self.prec_prior = np.array(prec_prior)    
        self.shape_prior = np.array(shape_prior)    
        self.rate_prior = np.array(rate_prior)
        
        self.mean_posterior = np.array(mean_prior)    
        self.prec_posterior = np.array(prec_prior)    
        self.shape_posterior = np.array(shape_prior)    
        self.rate_posterior = np.array(rate_prior)
        
        self.n = 0
        self.sum_of_xs = 0
        self.sum_of_xs_squared = 0
        
    
    def update(self, x_new: np.ndarray, *args, **kwargs):
        self.n += 1
        self.sum_of_xs += x_new
        self.sum_of_xs_squared += x_new**2

        x_bar = self.sum_of_xs / self.n
        sum_of_squares = self.sum_of_xs_squared - self.sum_of_xs**2 / self.n

        self.shape_posterior = self.shape_prior + 0.5*self.n
        self.prec_posterior = self.prec_prior + self.n
        
        self.rate_posterior = (
            self.rate_prior
            + 0.5*sum_of_squares 
            + 0.5*self.prec_prior*self.n*(x_bar - self.mean_prior)**2/(self.prec_posterior)
        )
        self.mean_posterior = (self.prec_prior*self.mean_prior + self.sum_of_xs) / (self.prec_posterior)
        
            
        
class MultivariateNormalGammaConjugatePrior(BaseConjugatePrior):
    # http://ericfrazerlock.com/LM_GoryDetails.pdf
    def __init__(self, mean_prior: np.ndarray, prec_prior: np.ndarray, shape_prior: np.ndarray, rate_prior: np.ndarray):        
        self.mean_prior = np.array(mean_prior)    
        self.prec_prior = np.array(prec_prior)    
        self.shape_prior = np.array(shape_prior)    
        self.rate_prior = np.array(rate_prior)
        
        self.mean_posterior = np.array(mean_prior)    
        self.prec_posterior = np.array(prec_prior)    
        self.shape_posterior = np.array(shape_prior)    
        self.rate_posterior = np.array(rate_prior)
        
        self.xs = None
        self.ys = None
        
    
    def update(self, x_new: np.ndarray, y_new: np.ndarray, *args, **kwargs):
        if len(x_new.shape) == 1:
            x_new = np.expand_dims(x_new, 0)
        if len(y_new.shape) == 1:
            y_new = np.expand_dims(y_new, 1)
            
        if self.xs is None:
            self.xs = x_new
            self.ys = y_new
        else:
            self.xs = np.vstack([self.xs, x_new])
            self.ys = np.vstack([self.ys, y_new])
            
        n = self.xs.shape[0]
        xx = self.xs.T @ self.xs
        
        self.prec_posterior = self.prec_prior + xx
        self.mean_posterior = np.linalg.inv(self.prec_posterior) @ (self.prec_prior @ self.mean_prior + self.xs.T @ self.ys)
        
        self.shape_posterior = self.shape_prior + 0.5*n
        self.rate_posterior = (
            self.rate_prior
            + 0.5 * (
                self.mean_prior.T @ self.prec_prior @ self.mean_prior
                + self.ys.T @ self.ys
                - self.mean_posterior.T @ self.prec_posterior @ self.mean_posterior
            )
        )
        
        