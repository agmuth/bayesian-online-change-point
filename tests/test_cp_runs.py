import numpy as np

from bocd.cp import BayesianOnlineChangepointDetection
from bocd.conjugate_likelihoods import *
from bocd.hazard_functions import ExponentialHazardFunction



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
        buffer=buffer
    )
    x_news = np.random.binomial(1, 0.5, run_length)
    x_news = np.expand_dims(x_news, 1)
    for x_new in x_news:
        cp.update(x_new)
    cp.sample(10)
    assert True
    
    
if __name__ == "__main__":
    test_bernoulli_cp_run()