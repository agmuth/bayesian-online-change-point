import numpy as np
from scipy import stats

from bocd.conjugate_likelihoods import *
from bocd.conjugate_priors import *

np.random.seed(583930)


def test_bernoulli_beta_posterior_predictive_rv():
    n_obvs = int(1e3)
    p_true = 1 / 3
    x_news = np.random.binomial(1, p_true, n_obvs)
    x_news = np.expand_dims(x_news, 1)
    model = BernoulliConjugateLikelihood(
        shape1_prior=np.array([1]),
        shape2_prior=np.array([1]),
    )

    for x_new in x_news:
        model.update_prior(x_new)
    model.update_posterior_predictive()
    assert np.isclose(model.posterior_predictive_rvs(n_obvs).mean(), p_true, atol=5e-2)


def test_poisson_gamma_posterior_predictive_rv():
    n_obvs = int(1e5)
    lam_true = 4.5
    x_news = np.random.poisson(lam_true, n_obvs)
    x_news = np.expand_dims(x_news, 1)
    model = PoissonConjugateLikelihood(
        shape_prior=np.array([1]),
        rate_prior=np.array([1]),
    )
    for x_new in x_news:
        model.update_prior(x_new)
    model.update_posterior_predictive()
    assert np.isclose(
        model.posterior_predictive_rvs(n_obvs).mean(), lam_true, atol=1e-2
    )


def test_normal_normal_gamma_posterior_predictive_rv():
    n_obvs = int(1e4)
    loc_true = 5
    prec_true = 2
    x_news = np.random.normal(loc_true, np.sqrt(1 / prec_true), n_obvs)
    x_news = np.expand_dims(x_news, 1)
    model = NormalConjugateLikelihood(
        mean_prior=np.array([0]),
        prec_prior=np.array([1]),
        shape_prior=np.array([1]),
        rate_prior=np.array([1]),
    )
    for x_new in x_news:
        model.update_prior(x_new)
    model.update_posterior_predictive()
    assert np.isclose(
        model.posterior_predictive_rvs(n_obvs).mean(), loc_true, atol=5e-2
    )


def test_normal_regression_multivariate_normal_gamma_posterior_predictive_rv():
    n_obvs = int(1e4)
    p = 1
    x_news = np.hstack([np.ones((n_obvs, 1)), np.random.normal(size=(n_obvs, p))])

    b_true = np.random.normal(size=(p + 1, 1))
    prec_true = 1
    y_news = x_news @ b_true + np.random.normal(
        scale=np.sqrt(1 / prec_true), size=((n_obvs, 1))
    )
    model = NormalRegressionConjugateLikelihood(
        mean_prior=np.expand_dims(np.zeros(p + 1), 1),
        prec_prior=1 * np.eye(p + 1),
        shape_prior=np.array([1]),
        rate_prior=np.array([1]),
    )
    for x_new, y_new in zip(x_news[:-1], y_news[:-1]):
        model.update_prior(x_new, y_new)

    model.update_posterior_predictive(x_news[-1])
    assert np.isclose(
        model.posterior_predictive_rvs(n_obvs).mean(), x_news[-1] @ b_true, atol=1e-1
    )
