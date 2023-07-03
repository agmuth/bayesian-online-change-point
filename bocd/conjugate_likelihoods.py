import numpy as np
from scipy import stats

from abc import ABC, abstractmethod
from bocd.conjugate_priors import *


class BaseConjugateLikelihood(ABC):
    @abstractmethod
    def __init__(self):
        self.conjugate_prior = None
        self.posterior_predictive = None

    def reset(self):
        self.conjugate_prior.reset()

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
        super().__init__()
        self.conjugate_prior = BetaPrior(a, b)
        self._update_posterior_predictive()
        
    def _update_posterior_predictive(self):
        p = self.conjugate_prior.a / (self.conjugate_prior.a + self.conjugate_prior.b)
        self.posterior_predictive = stats.binom(n=1, p=p)
        

    
class NormalConjugateLikelihood(BaseConjugateLikelihood):
    def __init__(self, m, p, alpha, beta):
        super().__init__()
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


# class NormalRegressionLikelihood(BaseConjugateLikelihood):
#     def __init__(self, m, p, alpha, beta):
#         super().__init__()
#         self.conjugate_prior = MultivariateNormalGammaPrior(m, p, alpha, beta)
#         self._update_posterior_predictive()

#     def _update_posterior_predictive(self):
#         df = self.conjugate_prior.nu
#         loc = self.conjugate_prior.b
#         shape = self.conjugate_prior.nu**-1 * self.conjugate_prior.n
#         self.posterior_predictive = stats.multivariate_t(df=df, loc=loc, shape=shape)


# if __name__ == "__main__":
#     k = 2
#     m = np.zeros(k)
#     p = np.eye(k)
#     alpha = 1
#     beta = 1
#     conjugate_likelihood = NormalRegressionLikelihood(m, p, alpha, beta)
#     x_new = np.array([9.0, 11.0])
#     y_new = np.array([5.])
#     conjugate_likelihood.update(x_new, y_new)

    

