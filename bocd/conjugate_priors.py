import numpy as np 

from abc import ABC, abstractmethod

# https://en.wikipedia.org/wiki/Conjugate_prior

class BaseConjugatePrior(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def update(self, x_new):
        raise NotImplementedError
    

class BetaPrior(BaseConjugatePrior):
    def __init__(self, a: np.ndarray, b: np.ndarray):
        self.a_prior = np.array(a)
        self.b_prior = np.array(b)
        self.a_posterior = np.array(a)
        self.b_posterior = np.array(b)
           
    def update(self, x_new: np.ndarray):
        self.a_posterior += x_new
        self.b_posterior += 1 - x_new


class NormalGammaConjugatePrior(BaseConjugatePrior):
    def __init__(self, mean: np.ndarray, prec: np.ndarray, shape: np.ndarray, rate: np.ndarray):
        self.mean_prior = np.array(mean)    
        self.prec_prior = np.array(prec)    
        self.shape_prior = np.array(shape)    
        self.rate_prior = np.array(rate)
        
        self.mean_posterior = np.array(mean)    
        self.prec_posterior = np.array(prec)    
        self.shape_posterior = np.array(shape)    
        self.rate_posterior = np.array(rate)
        
        self.n = 0
        self.sum_of_xs = 0
        self.sum_of_xs_squared = 0
        
    
    def update(self, x_new: np.ndarray):
        self.n += 1
        self.sum_of_xs += x_new
        self.sum_of_xs_squared += x_new**2

        x_bar = self.sum_of_xs / self.n
        sum_of_squares = self.sum_of_xs_squared - self.sum_of_xs**2 / self.n

        self.shape_posterior = self.shape_prior + 0.5*self.n
        self.rate_posterior = (
            self.rate_prior
            + 0.5*sum_of_squares 
            + 0.5*self.prec_prior*self.n*(x_bar - self.mean_prior)**2/(self.prec_prior + self.n)
        )
        self.mean_posterior = (self.prec_prior*self.mean_prior + self.sum_of_xs) / (self.prec_prior + self.n)
        self.prec_posterior = self.prec_prior + self.n
     

class GammaConjugatePrior(BaseConjugatePrior):
    def __init__(self, shape: np.ndarray, rate: np.ndarray):
        self.shape_prior = np.array(shape)    
        self.rate_prior = np.array(rate)   
        self.shape_posterior = np.array(shape)    
        self.rate_posterior = np.array(rate)
        
    def update(self, x_new: np.ndarray):
        self.shape_posterior += x_new
        self.rate_posterior += 1
        

class MultivariateNormalWishartConjugatePrior(BaseConjugatePrior):
    def __init__(self, mean: np.ndarray, prec: np.ndarray, scale: np.ndarray, df: np.ndarray):
        self.mean_prior = np.array(mean)    
        self.prec_prior = np.array(prec)    
        self.scale_prior = np.array(scale)    
        self.df_prior = np.array(df)
        
        self.mean_posterior = np.array(mean)    
        self.prec_posterior = np.array(prec)    
        self.scale_posterior = np.array(scale)    
        self.df_posterior = np.array(df)
        
        self.xs = None
        
        
    def update(self, x_new):
        
        if self.xs is None:
            self.xs = x_new
            self._p = len(x_new)
        else:
            self.xs = np.vstack([self.xs, x_new])
        
        n = self.xs.shape[0]
        x_bar = self.xs.mean(axis=1)
        
        self.mean_posterior = (self.prec_prior*self.mean_prior + self.n*x_bar) / (self.prec_prior+self.n)
        self.prec_posterior = self.prec_prior+self.n
        self.df_posterior = self.df_prior+self.n
        self.scale_posterior = np.linalg.inv(
            np.linalg.inv(self.scale_prior)
            + (self.xs - x_bar)@(self.xs - x_bar).T 
            + self.prec_prior*n/(self.prec_prior+n)
            *(x_bar - self.mean_prior)@(x_bar - self.mean_prior).T
        )
        
        
class MultivariateNormalGammaConjugatePrior(BaseConjugatePrior):
    # http://ericfrazerlock.com/LM_GoryDetails.pdf
    def __init__(self, mean: np.ndarray, prec: np.ndarray, shape: np.ndarray, rate: np.ndarray):
        self.mean_prior = np.array(mean)    
        self.prec_prior = np.array(prec)    
        self.shape_prior = np.array(shape)    
        self.rate_prior = np.array(rate)
        
        self.mean_posterior = np.array(mean)    
        self.prec_posterior = np.array(prec)    
        self.shape_posterior = np.array(shape)    
        self.rate_posterior = np.array(rate)
        
        self.xx = 0
        self.b = 0
        self.upsilon = 0
        self.nu = 0
        
        self.xs = None
        self.ys = None
        
    
    def update(self, x_new: np.ndarray, y_new: np.ndarray):
        
        if self.xs is None:
            self.xs = x_new
            self.ys = y_new
        else:
            self.xs = np.vstack([self.xs, x_new])
            self.ys = np.vstack([self.ys, y_new])
            
        n = self.xs.shape[0]
        xx = self.xs.T @ self.xs
        
        self.prec_posterior = self.prec_prior + xx
        self.mean_posterior = np.linalg.inv(self.prec_posterior) @ (self.prec_prior @ self.mean_prior + self.xs @ self.y)
        
        self.shape_posterior = self.shape_prior + 0.5*n
        self.rate_posterior = (
            self.rate_prior
            + 0.5 * (
                self.mean_prior.T @ self.prec_prior @ self.mean_prior
                + self.ys.T @ self.ys
                - self.mean_posterior @ self.prec_posterior @ self.mean_posterior
            )
        )
        
        