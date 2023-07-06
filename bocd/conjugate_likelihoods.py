from abc import ABC, abstractmethod

import numpy as np
from scipy import stats

from bocd.conjugate_priors import *


class BaseConjugateLikelihood(ABC):
    def update(self, x_new, *args, **kwargs):
        self.conjugate_prior.update(x_new, *args, **kwargs)

    def posterior_predictive_pdf(self, x):
        return self.posterior_predictive.pdf(x)

    def posterior_predictive_ppf(self, x):
        return self.posterior_predictive.ppf(x)

    def posterior_predictive_cdf(self, x):
        return self.posterior_predictive.cdf(x)

    def posterior_predictive_rvs(self, size):
        return self.posterior_predictive.rvs(size)

    @abstractmethod
    def update_posterior_predictive(self):
        raise NotImplementedError


class BernoulliConjugateLikelihood(BaseConjugateLikelihood):
    def __init__(self, shape1_prior: np.ndarray, shape2_prior: np.ndarray):
        self.conjugate_prior = BetaPrior(shape1_prior, shape2_prior)

    def update_posterior_predictive(self):
        p = self.conjugate_prior.shape1_posterior / (
            self.conjugate_prior.shape1_posterior
            + self.conjugate_prior.shape2_posterior
        )
        self.posterior_predictive = stats.binom(n=1, p=p)


class PoissonConjugateLikelihood(BaseConjugateLikelihood):
    def __init__(self, shape_prior: np.ndarray, rate_prior: np.ndarray):
        self.conjugate_prior = GammaConjugatePrior(shape_prior, rate_prior)

    def update_posterior_predictive(self):
        n = self.conjugate_prior.shape_posterior
        p = self.conjugate_prior.rate_posterior / (
            self.conjugate_prior.rate_posterior + 1
        )
        self.posterior_predictive = stats.nbinom(n=n, p=p)


class NormalConjugateLikelihood(BaseConjugateLikelihood):
    def __init__(
        self,
        mean_prior: np.ndarray,
        prec_prior: np.ndarray,
        shape_prior: np.ndarray,
        rate_prior: np.ndarray,
    ):
        self.conjugate_prior = NormalGammaConjugatePrior(
            mean_prior, prec_prior, shape_prior, rate_prior
        )

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
    def __init__(
        self,
        mean_prior: np.ndarray,
        prec_prior: np.ndarray,
        shape_prior: np.ndarray,
        rate_prior: np.ndarray,
    ):
        self.conjugate_prior = MultivariateNormalGammaConjugatePrior(
            mean_prior, prec_prior, shape_prior, rate_prior
        )

    def update_posterior_predictive(self, x_new: np.ndarray):
        # http://ericfrazerlock.com/LM_GoryDetails.pdf
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
