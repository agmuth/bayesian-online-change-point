from copy import deepcopy

import numpy as np

from bocd.conjugate_likelihoods import BaseConjugateLikelihood
from bocd.hazard_functions import BaseHazardFunction


class BayesianOnlineChangepointDetection:
    """Implementation of Bayesian Online Change Point detection
    ref: https://arxiv.org/pdf/0710.3742.pdf
    """

    def __init__(
        self,
        hazard_func: BaseHazardFunction,
        conjugate_likelihood: BaseConjugateLikelihood,
        buffer=256,
    ):
        """init

        Parameters
        ----------
        hazard_func : BaseHazardFunction
            Hazard function for time to changepoint.
        conjugate_likelihood : BaseConjugateLikelihood
            Initialized `BaseConjugateLikelihood` object.
        buffer : int, optional
            size of initial arrays, by default 256
        """
        # STEP 1: INITIALIZE
        self.time = 0
        self.buffer = buffer
        self.hazard_func = hazard_func
        self.conjugate_likelihood = (
            conjugate_likelihood  # save as reference conjugate likelihood
        )
        self.conjugate_likelihoods = (
            []
        )  # list of conjugate likelihoods indexed based on current run length starting at time t
        self._update_conjugate_likelihoods(None)
        self._init_trellises()

    def _init_trellises(self):
        """init various trellises for prob passing."""
        self._init_conditional_changepoint_probs()
        self._init_run_length_probs_trellis()
        self._init_joint_density_trellis()

    def _init_conditional_changepoint_probs(self):
        """init conditional changepoint probs from hazard func."""
        self.conditional_changepoint_probs = np.array(
            [self.hazard_func(t) for t in range(self.buffer)]
        )

    def _init_run_length_probs_trellis(self):
        """init run length probs trellis."""
        # recursively build up
        self.run_length_probs_trellis = np.zeros(
            (self.buffer, self.buffer)
        )  # cols are time rows are run length
        self.run_length_probs_trellis[0, 0] = 1  # changepoint at time t=0

    def _update_run_length_probs_trellis(self):
        """update run length probs trellis up to current time."""
        # max possible run length at time t-1 is t-1
        # prob that current run at time t is of length 0
        self.run_length_probs_trellis[0, self.time] = np.dot(
            self.run_length_probs_trellis[: self.time, self.time - 1],
            self.conditional_changepoint_probs[
                : self.time
            ],  # could index off of offset here
        )
        # prob that run at time t grew
        self.run_length_probs_trellis[1 : self.time + 1, self.time] = np.multiply(
            self.run_length_probs_trellis[: self.time, self.time - 1],
            (1 - self.conditional_changepoint_probs[: self.time]),  # same here
        )

    def _init_joint_density_trellis(self):
        """init joint density (run length, data) trellis."""
        self.joint_density_trellis = np.zeros((self.buffer, self.buffer))
        self.joint_density_trellis[0, 0] = 1

    def _update_joint_density_trellis(self, predictive_prob_cond_on_run_legth):
        """update joint density (run length, data) trellis to current time."""
        # cols are time rows are run length
        # CASE 1: density/probability that cp occured just before new data and thus current run length is 0
        self.joint_density_trellis[0, self.time] = (
            np.multiply(
                self.joint_density_trellis[
                    0 : self.time, self.time - 1
                ],  # densities/probabilities for all possible run lengths at time t-1
                self.conditional_changepoint_probs[
                    0 : self.time
                ],  # prob each run ended
            ).sum()
            * predictive_prob_cond_on_run_legth[
                -1
            ]  # density/probability base on just prior
        )

        # CASE 2: density/probability that no cp occuered and thus new data is from current run length and that run grew
        prob_new_obvs_came_from_existing_runs = np.multiply(
            # prob new datum came from run from each run
            self.joint_density_trellis[
                0 : self.time, self.time - 1
            ],  # densities/probabilities for all possible run lengths at time t-1
            predictive_prob_cond_on_run_legth[
                : self.time
            ],  # densities/probabilites of observing new datum cond on run length
        )

        self.joint_density_trellis[1 : self.time + 1, self.time] = np.multiply(
            prob_new_obvs_came_from_existing_runs,
            (
                1 - self.conditional_changepoint_probs[: self.time]
            ),  # probability current runs grows
        )

    def _predict_prob_cond_on_run_length(self, x_new):
        """Get density of `x_new` cond on run starting at times 0, 1, . . ., self.time

        Parameters
        ----------
        x_new : np.array like
            Must compatible with underlying conjugate likelihood/prior model.

        Returns
        -------
        np.array
            Vector of conditional probs.
        """
        predictive_prob_cond_on_run_legth = np.zeros(
            self.time + 1
        )  # need time inclusive for new process/cp
        for i in range(self.time):
            predictive_prob_cond_on_run_legth[i] += self.conjugate_likelihoods[
                i
            ].posterior_predictive_pdf(x_new)
        predictive_prob_cond_on_run_legth[
            -1
        ] = self.conjugate_likelihoods[-1].posterior_predictive_pdf(x_new)
        return predictive_prob_cond_on_run_legth

    def _calc_current_run_start_probs(self):
        """Calc the probs of the current run starting at time 0, 1, ..., self.time-1."""
        evidence = self.joint_density_trellis[:, self.time].sum()
        self.current_run_start_probs = (
            self.joint_density_trellis[:, self.time] / evidence
        )
        self.current_run_start_probs = np.nan_to_num(self.current_run_start_probs, 0.0)
        self.current_run_start_probs /= self.current_run_start_probs.sum()

    def _update_conjugate_likelihoods(self, x_new):
        """Update conjugate likelihoods corresponding to run lengths starting at times 0, 1, ..., self.time-1."""
        for i in range(self.time):
            self.conjugate_likelihoods[i].update(
                x_new
            )  # update conjugate-lik for run starting at time i with newest data
            self.conjugate_likelihoods[i].update_posterior_predictive(x_new)
        self.conjugate_likelihoods.append(
            deepcopy(self.conjugate_likelihood)
        )  # add conjugate-lik for run starting now
        self.conjugate_likelihoods[-1].update_posterior_predictive(x_new)

    def _expand_buffer(self):
        """Double array dimensions."""
        self.run_length_probs_trellis = np.vstack(
            [
                np.hstack(
                    [
                        self.run_length_probs_trellis,
                        np.zeros((self.buffer, self.buffer)),
                    ]
                ),
                np.zeros((self.buffer, 2 * self.buffer)),
            ]
        )
        self.joint_density_trellis = np.vstack(
            [
                np.hstack(
                    [self.joint_density_trellis, np.zeros((self.buffer, self.buffer))]
                ),
                np.zeros((self.buffer, 2 * self.buffer)),
            ]
        )
        self.buffer *= 2
        self._init_conditional_changepoint_probs()

    # TODO: remove?
    # def posterior_predictive(self, x_new):
    #     return np.dot(
    #         self._predict_prob_cond_on_run_length(x_new),
    #         self.current_run_start_probs[: self.time],
    #     )

    def sample(self, n: int, *args, **kwargs):
        """Sample from posterior predicitve of underlying conjugate likelihood models conditional on run length.

        Parameters
        ----------
        n : int
            Number of samples to return.

        Returns
        -------
        dict
            Dictionary where key value pairs are `run_length` and samples from the the posterior predictive corresponding to that run length.
        """

        samples = {}
        run_lengths, run_length_counts = np.unique(
            np.random.choice(
                np.arange(self.time + 1),
                replace=True,
                p=self.current_run_start_probs[: self.time + 1],
                size=n,
            ),
            return_counts=True,
        )
        for run_length, count in zip(run_lengths, run_length_counts):
            if count == 0:
                continue
            self.conjugate_likelihoods[run_length].update_posterior_predictive(
                *args, **kwargs
            )
            samples[run_length] = self.conjugate_likelihoods[
                run_length
            ].posterior_predictive_rvs(count)
        return samples

    def update(self, x_new):
        """Runs main `update` loop specified in reference paper.

        Parameters
        ----------
        x_new : np.array like
            Must compatible with underlying conjugate likelihood/prior model.
        """
        # STEP 2: OBSERVE NEW DATAUM
        self.time += 1
        if self.time == self.buffer:
            self._expand_buffer()

        # STEP 3: EVALUATE PREDICTIVE PROBABILITY
        self._update_run_length_probs_trellis()
        predictive_prob_cond_on_run_legth = self._predict_prob_cond_on_run_length(x_new)

        # STEP 4 & 5: CALCULATE GROWTH PROBABILITIES & CALCULATE CHANGEPOINT PROBABILITIES
        self._update_joint_density_trellis(predictive_prob_cond_on_run_legth)

        # STEP 6 & 8: CALCULATE EVIDENCE & DETERMINE RUN LENGTH DISTRIBUTION
        self._calc_current_run_start_probs()

        # STEP 8: UPDATE SUFFICIENT STATS
        self._update_conjugate_likelihoods(x_new)
