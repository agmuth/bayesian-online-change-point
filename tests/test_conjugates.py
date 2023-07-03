import numpy as np
from scipy import stats
from bocd.conjugate_likelihoods import NormalConjugateLikelihood, NormalRegressionLikelihood


def test_normal_conjugate_likelihood_update():
    m = 0
    p = 1
    alpha = 1
    beta = 1
    conjugate_likelihood = NormalConjugateLikelihood(m, p, alpha, beta)
    x_news = np.array([9.0, 11.0])
    conjugate_likelihood.update(x_news[0])
    conjugate_likelihood.update(x_news[1])

    n = len(x_news)
    x_bar = x_news.mean()
    ss = np.square(x_news - x_bar).sum()

    # test update to prior

    assert conjugate_likelihood.conjugate_prior.alpha_posterior == alpha + 0.5*n
    assert conjugate_likelihood.conjugate_prior.beta_posterior == (
        beta
        + 0.5*ss
        + 0.5*p*n*(x_bar-m)**2 / (p+n)
    )
    assert conjugate_likelihood.conjugate_prior.m_posterior == (p*m + n*x_bar) / (p + n)
    assert conjugate_likelihood.conjugate_prior.p_posterior == p + n
    
    rv = stats.t(
        df=2*(1 + 0.5*n),
        loc=(p*m + n*x_bar) / (p + n),
        scale = np.sqrt(
            (
                (
                    beta
                    + 0.5*ss
                    + 0.5*p*n*(x_bar-m)**2 / (p+n)
                ) / (alpha + 0.5*n)
            ) 
            * ((p + n + 1) / (p + n))
        )
    )
    ys = np.linspace(0, 1, 100)
    assert np.allclose(conjugate_likelihood.posterior_predictive.cdf(ys), rv.cdf(ys))    


def test_normal_regression_conjugate_likelihood_update():
    k = 2
    m = np.zeros(k)
    p = np.diag(k)
    alpha = 1
    beta = 1
    conjugate_likelihood = NormalRegressionLikelihood(m, p, alpha, beta)
    x_new = np.array([9.0, 11.0])
    y_new = np.array([5.])
    conjugate_likelihood.update(x_new, y_new)
