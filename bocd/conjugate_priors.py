from abc import ABC, abstractmethod

import numpy as np


class BaseConjugatePrior(ABC):
    """Abstract base class for conjugate priors.
    ref for most priors: https://en.wikipedia.org/wiki/Conjugate_prior
    """

    @abstractmethod
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def update(self, x_new, *args, **kwargs):
        raise NotImplementedError


class BetaConjugatePrior(BaseConjugatePrior):
    """Beta Conjugate Prior for Bernoulli likelihood/data."""

    def __init__(self, shape1_prior: np.ndarray, shape2_prior: np.ndarray):
        """init

        Parameters
        ----------
        shape1_prior : np.ndarray
            Prior for first shape param (alpha).
        shape2_prior : np.ndarray
            Prior for second shape param (beta).
        """
        self.shape1_prior = np.array(shape1_prior)
        self.shape2_prior = np.array(shape2_prior)
        self.shape1_posterior = np.array(shape1_prior)
        self.shape2_posterior = np.array(shape2_prior)

    def update(self, x_new: np.ndarray, *args, **kwargs):
        """Perform Bernoulli update.

        Parameters
        ----------
        x_new : np.ndarray
            Single occurance of observed Bernoulli data.
        """
        self.shape1_posterior += x_new
        self.shape2_posterior += 1 - x_new


class GammaConjugatePrior(BaseConjugatePrior):
    """Gamma Conjugate Prior for Poisson likelihood/data."""

    def __init__(self, shape_prior: np.ndarray, rate_prior: np.ndarray):
        """init

        Parameters
        ----------
        shape_prior : np.ndarray
            Prior for shape param.
        rate_prior : np.ndarray
            Prior for rate param.
        """
        self.shape_prior = np.array(shape_prior)
        self.rate_prior = np.array(rate_prior)
        self.shape_posterior = np.array(shape_prior)
        self.rate_posterior = np.array(rate_prior)

    def update(self, x_new: np.ndarray, *args, **kwargs):
        """Perform Poisson update.

        Parameters
        ----------
        x_new : np.ndarray
            Single occurance of observed Poisson data.
        """
        self.shape_posterior += x_new
        self.rate_posterior += 1


class NormalGammaConjugatePrior(BaseConjugatePrior):
    """Normal-Gamma Conjugate Prior for Normal likelihood/data."""

    def __init__(
        self,
        mean_prior: np.ndarray,
        prec_prior: np.ndarray,
        shape_prior: np.ndarray,
        rate_prior: np.ndarray,
    ):
        """init

        Parameters
        ----------
        mean_prior : np.ndarray
            Prior for mean param (normal part).
        prec_prior : np.ndarray
            Prior for precision param (normal part).
        shape_prior : np.ndarray
            Prior for shape param (gamma part).
        rate_prior : np.ndarray
            Prior for rate param (gamma part).
        """
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
        """Perform Normal update.

        Parameters
        ----------
        x_new : np.ndarray
            Single occurance of observed Normal data.
        """
        self.n += 1
        self.sum_of_xs += x_new
        self.sum_of_xs_squared += x_new**2

        x_bar = self.sum_of_xs / self.n
        sum_of_squares = self.sum_of_xs_squared - self.sum_of_xs**2 / self.n

        self.shape_posterior = self.shape_prior + 0.5 * self.n
        self.prec_posterior = self.prec_prior + self.n

        self.rate_posterior = (
            self.rate_prior
            + 0.5 * sum_of_squares
            + 0.5
            * self.prec_prior
            * self.n
            * (x_bar - self.mean_prior) ** 2
            / (self.prec_posterior)
        )
        self.mean_posterior = (self.prec_prior * self.mean_prior + self.sum_of_xs) / (
            self.prec_posterior
        )


class MultivariateNormalGammaConjugatePrior(BaseConjugatePrior):
    """Multivariate Normal Gamma conjugate prior for Bayesian linear regression
    ref: http://ericfrazerlock.com/LM_GoryDetails.pdf
    """

    def __init__(
        self,
        mean_prior: np.ndarray,
        prec_prior: np.ndarray,
        shape_prior: np.ndarray,
        rate_prior: np.ndarray,
    ):
        """init

        Parameters
        ----------
        mean_prior : np.ndarray
            Prior for mean (MVN part) (vector).
        prec_prior : np.ndarray
            Prior for precision (MVN part) (matrix).
        shape_prior : np.ndarray
            Prior for shape param (gamma part).
        rate_prior : np.ndarray
            Prior for rate param (gamma part).
        """
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
        self.mean_posterior = np.linalg.inv(self.prec_posterior) @ (
            self.prec_prior @ self.mean_prior + self.xs.T @ self.ys
        )

        self.shape_posterior = self.shape_prior + 0.5 * n
        self.rate_posterior = self.rate_prior + 0.5 * (
            self.mean_prior.T @ self.prec_prior @ self.mean_prior
            + self.ys.T @ self.ys
            - self.mean_posterior.T @ self.prec_posterior @ self.mean_posterior
        )
