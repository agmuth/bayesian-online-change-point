import numpy as np

from bocd.conjugate_likelihoods import *
from bocd.cp import BayesianOnlineChangepointDetection
from bocd.hazard_functions import ExponentialHazardFunction

np.random.seed(198882)


def test_bernoulli_cp_run():
    buffer = 8
    run_length = 24
    conjugate_lik = BernoulliConjugateLikelihood(
        shape1_prior=np.array([1]),
        shape2_prior=np.array([1]),
    )
    cp = BayesianOnlineChangepointDetection(
        hazard_func=ExponentialHazardFunction(scale=10),
        conjugate_likelihood=conjugate_lik,
        buffer=buffer,
    )
    x_news = np.random.binomial(1, 0.5, run_length)
    x_news = np.expand_dims(x_news, 1)
    for x_new in x_news:
        cp.update(x_new)
    cp.sample(10)
    assert True


def test_poisson_cp_run():
    buffer = 8
    run_length = 24
    conjugate_lik = PoissonConjugateLikelihood(
        shape_prior=np.array([1]),
        rate_prior=np.array([1]),
    )
    cp = BayesianOnlineChangepointDetection(
        hazard_func=ExponentialHazardFunction(scale=10),
        conjugate_likelihood=conjugate_lik,
        buffer=buffer,
    )
    x_news = np.random.poisson(3, run_length)
    x_news = np.expand_dims(x_news, 1)
    for x_new in x_news:
        cp.update(x_new)
    cp.sample(10)
    assert True


def test_normal_cp_run():
    buffer = 8
    run_length = 24
    conjugate_lik = NormalConjugateLikelihood(
        mean_prior=np.array([0]),
        prec_prior=np.array([1]),
        shape_prior=np.array([1]),
        rate_prior=np.array([1]),
    )
    cp = BayesianOnlineChangepointDetection(
        hazard_func=ExponentialHazardFunction(scale=10),
        conjugate_likelihood=conjugate_lik,
        buffer=buffer,
    )
    x_news = np.random.normal(2, 5, run_length)
    x_news = np.expand_dims(x_news, 1)
    for x_new in x_news:
        cp.update(x_new)
    cp.sample(10)
    assert True


def test_normal_ar_p_run():
    buffer = 8
    run_length = 24
    p = 2
    b_true = np.random.normal(size=(p + 1, 1))
    prec_true = 1
    x_p = np.zeros((1, p))
    conjugate_lik = AutoRegressiveOrderPConjugateLikelihood(
        mean_prior=np.expand_dims(np.zeros(p + 1), 1),
        prec_prior=1 * np.eye(p + 1),
        shape_prior=np.array([1]),
        rate_prior=np.array([1]),
        p=p,
        x_0=x_p,
    )
    cp = BayesianOnlineChangepointDetection(
        hazard_func=ExponentialHazardFunction(scale=10),
        conjugate_likelihood=conjugate_lik,
        buffer=buffer,
    )

    x_p = np.hstack([np.ones((1, 1)), x_p])

    for i in range(run_length):
        x_new = x_p @ b_true + np.random.normal(0, prec_true)
        cp.update(x_new)
        x_p = np.expand_dims(np.hstack([x_p[0][[0]], x_new[0], x_p[0][1:-1]]), 0)
    cp.sample(10)
    assert True


if __name__ == "__main__":
    test_normal_ar_p_run()
