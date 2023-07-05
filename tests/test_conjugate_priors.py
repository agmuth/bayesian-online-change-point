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


def test_multivariate_normal_gamma_prior_update():
    n = 2
    p = 2
    x_news = np.array(
        [
            [1., 1.04020929],
            [1., 0.98626374]
        ]
    )
    b = np.array([[0.64182395], [0.05373533]])
    y_news = x_news @ b + np.random.normal(scale=0.1, size=(n, 1))
    
    model = MultivariateNormalGammaConjugatePrior(
        mean_prior=np.expand_dims(np.array([0.0, 0.0]), 1),
        prec_prior=1*np.eye(2),
        shape_prior=np.array([1]),
        rate_prior=np.array([1]),
    )
    
    for x_new, y_new in zip(x_news, y_news):
        model.update(x_new, y_new)
        
    assert np.array_equal(x_news, model.xs)
    assert np.array_equal(y_news, model.ys)
    assert np.array_equal(model.prec_posterior, np.eye(2) + x_news.T @ x_news)
    assert np.array_equal(model.mean_posterior, 
            (
                np.linalg.inv(model.prec_posterior) @ (x_news.T @ y_news) 
            )
        )
    
    assert model.shape_posterior == 1 + 0.5*2
    assert model.rate_posterior == (
        1
        + 0.5 * (
            y_news.T @ y_news - model.mean_posterior.T @ model.prec_posterior @ model.mean_posterior
        )
    )
    
    
if __name__ == "__main__":
    test_multivariate_normal_gamma_prior_update()