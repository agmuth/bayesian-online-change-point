import numpy as np
from scipy import stats

from abc import ABC, abstractmethod
from bocd.conjugate_priors import *


class BaseConjugateLikelihood(ABC):
    def update(self, x_new):
        self.conjugate_prior.update(x_new)
        self._update_posterior_predictive()
    
    def posterior_predictive_pdf(self, x):
        return self.posterior_predictive.pdf(x)
    
    def posterior_predictive_ppf(self, x):
        return self.posterior_predictive.ppf(x)
    
    def posterior_predictive_cdf(self, x):
        return self.posterior_predictive.cdf(x)
    
    def posterior_predictive_rvs(self, size):
        return self.posterior_predictive.rvs(size)
    
    @abstractmethod
    def _update_posterior_predictive(self):
        raise NotImplementedError
    
    
class BernoulliConjugateLikelihood(BaseConjugateLikelihood):
    def __init__(self, a: np.ndarray, b: np.ndarray):
        self.conjugate_prior = BetaPrior(a, b)
        self._update_posterior_predictive()
        
    def _update_posterior_predictive(self):
        p = self.conjugate_prior.a / (self.conjugate_prior.a + self.conjugate_prior.b)
        self.posterior_predictive = stats.binom(n=1, p=p)
        

class NormalConjugateLikelihood(BaseConjugateLikelihood):
    def __init__(self, m, p, alpha, beta):
        self.conjugate_prior = NormalGammaConjugatePrior(m, p, alpha, beta)
        self._update_posterior_predictive()

    def _update_posterior_predictive(self):
        df = 2*self.conjugate_prior.alpha_posterior
        loc = self.conjugate_prior.m_posterior
        scale = np.sqrt(
            (self.conjugate_prior.beta_posterior/self.conjugate_prior.alpha_posterior) 
            * ((self.conjugate_prior.p_posterior + 1)/self.conjugate_prior.p_posterior)
        )
        self.posterior_predictive = stats.t(df=df, loc=loc, scale=scale)


class PoissonConjugateLikelihood(BaseConjugateLikelihood):
    def __init__(self, shape: np.ndarray, rate: np.ndarray):
        self.conjugate_prior = GammaConjugatePrior(shape, rate)
        self._update_posterior_predictive()
        
    def update(self, x_new):
        self.conjugate_prior.update(x_new)
        self._update_posterior_predictive()
        
    def _update_posterior_predictive(self):
        n = self.conjugate_prior.shape_posterior
        p = self.conjugate_prior.rate_posterior / (self.conjugate_prior.rate_posterior+1)
        self.posterior_predictive = stats.nbinom(n=n, p=p)
        
        
class MultivariateNormalConjugateLikelihood(BaseConjugateLikelihood):
    def __init__(self, mean: np.ndarray, prec: np.ndarray, scale: np.ndarray, df: np.ndarray):
        self.conjugate_prior = MultivariateNormalWishartConjugatePrior(mean, prec, scale, df)
        self._update_posterior_predictive()
        
    def _update_posterior_predictive(self):
        df = self.conjugate_prior.df_posterior-self.conjugate_prior._p+1
        loc = self.conjugate_prior.mean_posterior
        shape = (
            np.linalg.inv(self.conjugate_prior.scale_posterior)
            * (self.conjugate_prior.prec_posterior+1)
            / (self.conjugate_prior.prec_posterior*df)
        )
        self.posterior_predictive = stats.multivariate_t(loc, shape, df)


