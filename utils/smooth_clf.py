# -*- coding: utf-8 -*-
"""
smooth classifier implementation - modified from Cohen et al.
"""

#%% imports

import torch
from scipy.stats import norm, binom_test
import numpy as np
from math import ceil
from statsmodels.stats.proportion import proportion_confint


#%% class - Smooth

class Smooth(object):
    """A smoothed classifier g """

    # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(self, base_classifier: torch.nn.Module, num_classes: int, sigma: float, device: str):
        """
        :param base_classifier: maps from [batch x channel x height x width] to [batch x num_classes]
        :param num_classes:
        :param sigma: the noise level hyperparameter
        :param device: the device to use (cpu or cuda)
        """
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.sigma = sigma
        self.device = device
    # end func

    def certify(self, x: torch.tensor, n0: int, n: int, alpha: float, batch_size: int) -> (int, float):
        """ Monte Carlo algorithm for certifying that g's prediction around x is constant within some L2 radius.
        With probability at least 1 - alpha, the class returned by this method will equal g(x), and g's prediction 
        will be robust within a L2 ball of radius R around x.

        :param x: the input [channel x height x width]
        :param n0: the number of Monte Carlo samples to use for selection
        :param n: the number of Monte Carlo samples to use for estimation
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: (predicted class, certified radius, Pa lower bound)
                 in the case of abstention, the class will be ABSTAIN and the radius 0.
        """
        self.base_classifier.eval()
        # draw samples of f(x + epsilon)
        counts_selection = self._sample_noise(x, n0, batch_size)
        # use these samples to take a guess at the top class
        cAHat = counts_selection.argmax().item()
        # draw more samples of f(x + epsilon)
        counts_estimation = self._sample_noise(x, n, batch_size)
        # use these samples to estimate a lower bound on pA
        nA = counts_estimation[cAHat].item()
        pABar = self._lower_confidence_bound(nA, n, alpha)
        if pABar < 0.5:
            return Smooth.ABSTAIN, 0.0, pABar
        else:
            radius = self.sigma * norm.ppf(pABar)
            return cAHat, radius, pABar
        # end if
    # end func
    
    def certify_nopred(self, x: torch.tensor, n: int, alpha: float, batch_size: int) -> (int, float):
        """ Monte Carlo algorithm for certifying that g's prediction around x is constant within some L2 radius.
        With probability at least 1 - alpha, the class returned by this method will equal g(x), and g's prediction will
        be robust within a L2 ball of radius R around x.
        the difference of this function with certify is that it doesn't use n0 samples to do prediction first, and instead
        uses n samples for both steps; we can prove that this is valid. This saves a small amount of n0 samples

        :param x: the input [channel x height x width]
        :param n: the number of Monte Carlo samples to use for estimation
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: (predicted class, certified radius, Pa lower bound)
                 in the case of abstention, the class will be ABSTAIN and the radius 0.
        """
        self.base_classifier.eval()
        # draw samples of f(x + epsilon)
        counts_selection = self._sample_noise(x, n, batch_size)
        # use these samples to take a guess at the top class
        cAHat = counts_selection.argmax().item()
        counts_estimation = counts_selection
        # use these samples to estimate a lower bound on pA
        nA = counts_estimation[cAHat].item()
        pABar = self._lower_confidence_bound(nA, n, alpha)
        if pABar < 0.5:
            return Smooth.ABSTAIN, 0.0, pABar
        else:
            radius = self.sigma * norm.ppf(pABar)
            return cAHat, radius, pABar
        # end if
    # end func

    def predict(self, x: torch.tensor, n: int, alpha: float, batch_size: int) -> int:
        """ Monte Carlo algorithm for evaluating the prediction of g at x.  With probability at least 1 - alpha, the
        class returned by this method will equal g(x).

        This function uses the hypothesis test described in https://arxiv.org/abs/1610.03944
        for identifying the top category of a multinomial distribution.

        :param x: the input [channel x height x width]
        :param n: the number of Monte Carlo samples to use
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: the predicted class, or ABSTAIN
        """
        self.base_classifier.eval()
        counts = self._sample_noise(x, n, batch_size)
        top2 = counts.argsort()[::-1][:2]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]
        if binom_test(count1, count1 + count2, p=0.5) > alpha:
            return Smooth.ABSTAIN
        else:
            return top2[0]
        # end if
    # end func

    def _sample_noise(self, x: torch.tensor, num: int, batch_size: int) -> np.ndarray:
        """ Sample the base classifier's prediction under noisy corruptions of the input x.

        :param x: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size: the batch size we like for drawing the samples
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """
        with torch.no_grad():
            # init counters for each class
            counts = np.zeros(self.num_classes, dtype = int)
            
            # draw the noisy samples in batches of length batch_size
            for _ in range(ceil(num / batch_size)):
                # the current batch size
                this_batch_size = min(batch_size, num)
                # reduce this batch from the number of samples remaining
                num -= this_batch_size

                # repeat x this_batch_size number of times, in order to add noise afterwards
                # e.g. x -> [x, x, x] -> [x + e1, x + e2, x e3]
                batch = x.repeat((this_batch_size, 1, 1, 1))
                # make gauss noise with sigma
                noise = torch.randn_like(batch, device = self.device) * self.sigma
                # get preds on noisy samples
                predictions = self.base_classifier(batch + noise).argmax(1)
                # update class counts
                counts += self._count_arr(predictions.cpu().numpy(), self.num_classes)
            return counts
        # end with
    # end func

    def _count_arr(self, arr: np.ndarray, length: int) -> np.ndarray:
        # counts the number of occurances of each element in an array
        counts = np.zeros(length, dtype = int)
        for idx in arr:
            counts[idx] += 1
        return counts
    # end func

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.

        This function uses the Clopper-Pearson method.

        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NA, N, alpha = 2 * alpha, method="beta")[0]
    # end func