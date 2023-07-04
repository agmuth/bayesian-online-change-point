import numpy as np
from scipy import stats
from bocd.conjugate_likelihoods import NormalConjugateLikelihood


def test_normal_conjugate_likelihood_update():
    m = np.array([0.0])
    p = np.array([1.0])
    alpha = np.array([1])
    beta = np.array([1])
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
    
    # test posterior predicitve distn
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
    
    
