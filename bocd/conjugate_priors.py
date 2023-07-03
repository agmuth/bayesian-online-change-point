import numpy as np 

from abc import ABC, abstractmethod


class BaseConjugatePrior(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def update(self, x_new):
        raise NotImplementedError
    
    @abstractmethod
    def reset(self):
        raise NotImplementedError
    

class MultivariateNormalGammaPrior(BaseConjugatePrior):
    def __init__(self, m, p, alpha, beta):
        self.m_prior = m
        self.p_prior = p
        self.alpha_prior = alpha
        self.beta_prior = beta
        self.reset()
    
    def update(self, x_new, y_new):
        if self.X:
            self.X = np.vstack([self.X, x_new])
        else:
            self.X = np.array(x_new)
        
        if self.y:
            self.y = np.vstack([self.y, y_new])
        else:
            self.y = np.array(y_new)

        self.n = self.X.T @ self.X 
        self.b = np.linalg.inv(self.n) @ self.X.T @ self.y

        self.nu = self.X.shape[0] - np.linalg.matrix_rank(self.n)
        self.upsilon = self.nu**-1 * np.square(self.y - self.X @ self.b).sum()

        self.p_posterior = self.p_prior + self.n
        self.m_posterior = np.linalg.inv(self.p_prior + self.n) @ (self.p_prior @ self.m_prior + self.n @ self.p_prior)
        self.alpha_posterior = self.alpha_prior + 0.5*self.X.shape[0]
        self.beta_posterior = (
            (self.alpha_prior + self.X.shape[0])**-1
            * (
                (self.alpha_prior*self.beta_prior + self.m_prior.T @ self.p_prior @ self.m_prior)
                + (self.nu*self.upsilon +  self.b.T @ self.n @ self.b)
                + (self.m_posterior.T @ self.p_posterior @ self.m_posterior)
            )
        )
    
    def reset(self):
        self.nu = None
        self.upsilon = None 
        self.n = None
        self.b = None
        self.y = None
        self.X = None
        self.m_posterior = self.m_prior
        self.p_posterior = self.p_prior
        self.alpha_posterior = self.alpha_prior
        self.beta_posterior = self.beta_prior

class NormalGammaConjugatePrior(BaseConjugatePrior):
    def __init__(self, m, p, alpha, beta):
        self.m_prior = m
        self.p_prior = p
        self.alpha_prior = alpha
        self.beta_prior = beta
        self.reset()
    
    def update(self, x_new):
        self.n += 1
        self.sum_of_xs += x_new
        self.sum_of_xs_squared += x_new**2

        x_bar = self.sum_of_xs / self.n
        sum_of_squares = self.sum_of_xs_squared - self.sum_of_xs**2 / self.n

        self.alpha_posterior = self.alpha_prior + 0.5*self.n
        self.beta_posterior = (
            self.beta_prior
            + 0.5*sum_of_squares 
            + 0.5*self.p_prior*self.n*(x_bar - self.m_prior)**2/(self.p_prior + self.n)
        )
        self.m_posterior = (self.p_prior*self.m_prior + self.sum_of_xs) / (self.p_prior + self.n)
        self.p_posterior = self.p_prior + self.n
    
    def reset(self):
        self.n = 0
        self.sum_of_xs = 0
        self.sum_of_xs_squared = 0
        self.m_posterior = self.m_prior
        self.p_posterior = self.p_prior
        self.alpha_posterior = self.alpha_prior
        self.beta_posterior = self.beta_prior    
