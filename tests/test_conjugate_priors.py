import numpy as np

from bocd.conjugate_priors import *


def test_beta_prior_update():
    x_news = np.array([1, 0, 1])
    x_news = np.expand_dims(x_news, 1)
    model = BetaPrior(
        shape1_prior=np.array([1]),
        shape2_prior=np.array([2]),
    )
    for x_new in x_news:
        model.update(x_new)
    assert model.shape1_posterior == 3
    assert model.shape2_posterior == 3
    
    
def test_gamma_prior_update():
    x_news  = np.array([1, 2, 3])
    x_news = np.expand_dims(x_news, 1)
    model = GammaConjugatePrior(
        shape_prior=np.array([1]),
        rate_prior=np.array([1]),
    )
    for x_new in x_news:
        model.update(x_new)
    assert model.shape_posterior == 7
    assert model.rate_posterior == 4
    
    
def test_normal_gamma_prior_update():
    x_news = np.array([-1.5, 2.5])
    x_news = np.expand_dims(x_news, 1)
    model = NormalGammaConjugatePrior(
        mean_prior=np.array([0]),
        prec_prior=np.array([1]),
        shape_prior=np.array([1]),
        rate_prior=np.array([1]),
    )
    for x_new in x_news:
        model.update(x_new)
    assert model.mean_posterior == 1 / (1 + 2)
    assert model.prec_posterior == (1 + 2)
    assert model.shape_posterior == (1 + 0.5*2)
    assert model.rate_posterior == (
        1
        + 0.5*np.square(x_news - x_news.mean()).sum()
        + 0.5*1*2*(x_news.mean() - 0)**2 / (1 + 2)
    )

    
if __name__ == "__main__":
    test_normal_gamma_prior_update()