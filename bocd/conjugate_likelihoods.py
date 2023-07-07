from abc import ABC, abstractmethod

import numpy as np
from scipy import stats

from bocd.conjugate_priors import *


class BaseConjugateLikelihood(ABC):
    """Abstract base class for conjugate likelihoods.
    ref for most likelihoods: https://en.wikipedia.org/wiki/Conjugate_prior
    """

    def update(self, x_new):
        """updates conjugate prior"""
        self.update_prior(x_new)
        self.update_posterior_predictive()
        
    def update_prior(self, x_new):
        self.conjugate_prior.update(x_new)

    def posterior_predictive_pdf(self, x):
        """Method to access pdf of posterior predictive."""
        return self.posterior_predictive.pdf(x)

    def posterior_predictive_ppf(self, x):
        """Method to access ppf of posterior predictive."""
        return self.posterior_predictive.ppf(x)

    def posterior_predictive_cdf(self, x):
        """Method to access cdf of posterior predictive."""
        return self.posterior_predictive.cdf(x)

    def posterior_predictive_rvs(self, size):
        """Method to access rvs of posterior predictive."""
        return self.posterior_predictive.rvs(size)

    @abstractmethod
    def update_posterior_predictive(self, *args, **kwargs):
        """Implement/construct rv for current posterior predicitve distn based on current posterior."""
        raise NotImplementedError


class BernoulliConjugateLikelihood(BaseConjugateLikelihood):
    """Bernoulli/Beta conjugate likelihood/prior implementation."""

    def __init__(self, shape1_prior: np.ndarray, shape2_prior: np.ndarray):
        """See docs for `BetaConjugatePrior`."""
        self.conjugate_prior = BetaConjugatePrior(shape1_prior, shape2_prior)
        self.update_posterior_predictive()

    def update_posterior_predictive(self):
        p = self.conjugate_prior.shape1_posterior / (
            self.conjugate_prior.shape1_posterior
            + self.conjugate_prior.shape2_posterior
        )
        self.posterior_predictive = stats.binom(n=1, p=p)
        
    def posterior_predictive_pdf(self, x):
        return self.posterior_predictive.pmf(x)


class PoissonConjugateLikelihood(BaseConjugateLikelihood):
    """Poisson/Gamma conjugate likelihood/prior implementation."""

    def __init__(self, shape_prior: np.ndarray, rate_prior: np.ndarray):
        """See docs for `GammaConjugatePrior`."""
        self.conjugate_prior = GammaConjugatePrior(shape_prior, rate_prior)
        self.update_posterior_predictive()

    def update_posterior_predictive(self, *args, **kwargs):
        n = self.conjugate_prior.shape_posterior
        p = self.conjugate_prior.rate_posterior / (
            self.conjugate_prior.rate_posterior + 1
        )
        self.posterior_predictive = stats.nbinom(n=n, p=p)
    
    def posterior_predictive_pdf(self, x):
        return self.posterior_predictive.pmf(x)


class NormalConjugateLikelihood(BaseConjugateLikelihood):
    """Normal/Normal Gamma conjugate/likelihood implementation."""

    def __init__(
        self,
        mean_prior: np.ndarray,
        prec_prior: np.ndarray,
        shape_prior: np.ndarray,
        rate_prior: np.ndarray,
    ):
        """See docs for `NormalGammaConjugatePrior`."""
        self.conjugate_prior = NormalGammaConjugatePrior(
            mean_prior, prec_prior, shape_prior, rate_prior
        )
        self.update_posterior_predictive()

    def update_posterior_predictive(self):
        df = 2 * self.conjugate_prior.shape_posterior
        loc = self.conjugate_prior.mean_posterior
        scale = np.sqrt(
            (self.conjugate_prior.rate_posterior / self.conjugate_prior.shape_posterior)
            * (
                (self.conjugate_prior.prec_posterior + 1)
                / self.conjugate_prior.prec_posterior
            )
        )
        self.posterior_predictive = stats.t(df=df, loc=loc, scale=scale)


class NormalRegressionConjugateLikelihood(BaseConjugateLikelihood):
    """Linear Regression/Normal Gamma conjugate likelihood/prior implementation.
    NOT MEANT TO BE DIRECTLY USED WITH `BayesianOnlineChangepointDetection`.
    """

    def __init__(
        self,
        mean_prior: np.ndarray,
        prec_prior: np.ndarray,
        shape_prior: np.ndarray,
        rate_prior: np.ndarray,
    ):
        """See docs for `MultivariateNormalGammaConjugatePrior`."""
        self.conjugate_prior = MultivariateNormalGammaConjugatePrior(
            mean_prior, prec_prior, shape_prior, rate_prior
        )

    def update(self, *args, **kwargs):
        raise NotImplementedError
    
    def update_prior(self, x_new: np.ndarray, y_new: np.ndarray):
        """Update conjugate prior

        Parameters
        ----------
        x_new : np.ndarray
            Covariate vector.
        y_new : np.ndarray
            Dependent variable.
        """
        self.conjugate_prior.update(x_new, y_new)
        
        
    def update_posterior_predictive(self, x_new: np.ndarray):
        """Update posterior predictive conditional on new covariates
        ref: http://ericfrazerlock.com/LM_GoryDetails.pdf

        Parameters
        ----------
        x_new : np.ndarray
            New covariates.
        """

        if len(x_new.shape) == 1:
            x_new = np.expand_dims(x_new, 0)
        df = self.conjugate_prior.shape_posterior * 2
        loc = x_new @ self.conjugate_prior.mean_posterior
        shape = (
            self.conjugate_prior.rate_posterior
            / self.conjugate_prior.shape_posterior
            * np.eye(x_new.shape[0])
            + x_new @ np.linalg.inv(self.conjugate_prior.prec_posterior) @ x_new.T
        )
        self.posterior_predictive = stats.multivariate_t(loc, shape, df)



class AutoRegressiveOrderPConjugateLikelihood(NormalRegressionConjugateLikelihood):
    def __init__(
        self,
        mean_prior: np.ndarray,
        prec_prior: np.ndarray,
        shape_prior: np.ndarray,
        rate_prior: np.ndarray,
        p: int,
        x_0: np.ndarray
    ):
        """See docs for `MultivariateNormalGammaConjugatePrior`."""
        super().__init__(mean_prior, prec_prior, shape_prior, rate_prior)
        self.p = p 
        self.x_p = x_0  # p most recent obvs
        
    def update(self, x_new):
        """updates conjugate prior"""
        super().update(x_new=self.x_p, y_new=x_new)
        self.x_p = np.concatenate(x_new, self.x_p[:-1])
        self.update_posterior_predictive()
        
    def update_posterior_predictive(self):
        super().update_posterior_predictive(self.x_p)
    