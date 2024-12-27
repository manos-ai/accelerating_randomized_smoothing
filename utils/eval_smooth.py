#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
smoothing evaluations functions
"""

#%% imports

from typing import Callable, Union, Tuple, List, Dict
#import numpy as np
import tqdm
import pandas as pd
import time

# import nromal distrib
#from scipy.stats import norm, binom_test
#from statsmodels.stats.proportion import proportion_confint

import torch
from torch import nn
#from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Sampler

# own modules
from smooth_clf import Smooth as SmoothClassifier


#%% helper functions

class NthSampleSampler(Sampler):
    # create a sampler to retrieve only each n-th element in a dataloader
    # this runs faster than the original implementation
    def __init__(self, data_source, n):
        self.data_source = data_source
        self.n = n

    def __iter__(self):
        return iter(range(0, len(self.data_source), self.n))

    def __len__(self):
        return len(self.data_source) // self.n
# end class


def predict_model(model: nn.Module, dataset: Dataset, batch_size: int, device: str = 'cuda') -> float:
    """
    Use the model to predict a label for each sample in the provided dataset.
    Parameters
    ----------
    model: nn.Module
        The input model to be used.
    dataset: torch.utils.data.Dataset
        The dataset to predict for.
    batch_size: int
        The batch size.
    device: str
        Torch device to use (cpu or cuda)


    Returns
    -------
    float: the accuracy on the provided dataset.
    """
    # added after the project:
        # init
    model.eval()
    
    # prep loader and calc numb of total batches used
    test_loader = DataLoader(dataset, batch_size = batch_size, shuffle = False)
    num_batches = int(torch.ceil(torch.tensor(len(dataset) / batch_size)).item())
    predictions = []
    targets = []
    
    with torch.no_grad():
        for x, y in tqdm.tqdm(test_loader, total = num_batches):
            targets.append(y)
            x = x.to(device)
            logits = model(x).cpu()
            predictions.append(logits.argmax(-1))
        # end for
    # end with
    
    predictions = torch.cat(predictions)
    targets = torch.cat(targets)
    accuracy = (predictions == targets).float().mean().item()
    return accuracy
# end func


#%% function - evaluate_robustness_smoothing_nopred

def evaluate_robustness_smoothing(base_classifier: nn.Module, sigma: float, dataset: Dataset,
                                  num_samples_1: int = 100, num_samples_2: int = 10000,
                                  alpha: float = 0.05, certification_batch_size: float = 5000, num_classes: int = 10,
                                  device: str = 'cuda', skip: int = 1
                                  ) -> Dict:
    """
    Evaluate the robustness of a smooth classifier based on the input base classifier via randomized smoothing.
    Parameters
    ----------
    base_classifier: nn.Module
        The input base classifier to use in the randomized smoothing process.
    sigma: float
        The variance to use for the Gaussian noise samples.
    dataset: Dataset
        The input dataset to predict on.
    num_samples_1: int
        The number of samples used to determine the most likely class.
    num_samples_2: int
        The number of samples used to perform the certification.
    alpha: float
        The desired confidence level that the top class is indeed the most likely class. E.g. alpha=0.05 means that
        the expected error rate must not be larger than 5%.
    certification_batch_size: int
        The batch size to use during the certification, i.e. how many noise samples to classify in parallel.
    num_classes: int
        The number of classes.
    skip: int
        The datapoints to skip; e.g. we evaluate on a datapoint with index i only if i mod skip = 0.
    device: str
        The device to use (cpu or cuda)

    Returns
    -------
    certif_results: pandas DataFrame
        each row contains data of the form [index, y_real, y_pred, radius, pA], which describe the 
        index of the datapoint examined, the real and predicted class, as well as the radius certified
        by smoothing and the lower confidence bound probability pA
    """
    # init a smoothed classifier
    base_classifier.eval()
    model = SmoothClassifier(base_classifier = base_classifier, sigma = sigma, num_classes = num_classes, device = device)
        
    sampler = NthSampleSampler(dataset, skip)
    test_loader = DataLoader(dataset, batch_size = 1, sampler = sampler)
    
    
    #test_loader = DataLoader(dataset, batch_size = 1, shuffle = False)
    # inits
    cnt = -skip # counts the datapoints
    data = [] # will hold the certification results
    col_names = ['index', 'y_real', 'y_pred', 'radius', 'pA', 'time']
    
    #for x, y in tqdm.tqdm(test_loader, total = len(dataset)):
    for x, y in test_loader:
        cnt += skip
        idx = cnt # index of the current data point
        #if cnt % skip != 0:
            #continue
        # end if
        print(cnt)
        x = x.to(device)
        # try to certify x
        t0 = time.time()
        pred_class, radius, pA = model.certify(x, num_samples_1, num_samples_2, alpha = alpha,
                                           batch_size = certification_batch_size)
        t1 = time.time()
        dt = t1 - t0
        # check certif results
        y_real = int(y)
        res = [idx, y_real, pred_class, radius, pA, dt]
        data.append(res)
    # end for

    # cast results as dataframe
    certif_results = pd.DataFrame(data, columns = col_names)
    return certif_results
# end func


#%% function - evaluate_robustness_smoothing_nopred

def evaluate_robustness_smoothing_nopred(base_classifier: nn.Module, sigma: float, dataset: Dataset,
                                  num_samples_2: int = 10000,
                                  alpha: float = 0.05, certification_batch_size: float = 5000, num_classes: int = 10,
                                  device: str = 'cuda', skip: int = 1
                                  ) -> Dict:
    """
    Evaluate the robustness of a smooth classifier based on the input base classifier via randomized smoothing.
    the difference between 'evaluate_robustness_smoothing' is that this function doesn't use n0 samples to
    do prediction first
    
    Parameters
    ----------
    base_classifier: nn.Module
        The input base classifier to use in the randomized smoothing process.
    sigma: float
        The variance to use for the Gaussian noise samples.
    dataset: Dataset
        The input dataset to predict on.
    num_samples_1: int
        The number of samples used to determine the most likely class.
    num_samples_2: int
        The number of samples used to perform the certification.
    alpha: float
        The desired confidence level that the top class is indeed the most likely class. E.g. alpha=0.05 means that
        the expected error rate must not be larger than 5%.
    certification_batch_size: int
        The batch size to use during the certification, i.e. how many noise samples to classify in parallel.
    num_classes: int
        The number of classes.
    skip: int
        The datapoints to skip; e.g. we evaluate on a datapoint with index i only if i mod skip = 0.
    device: str
        The device to use (cpu or cuda)

    Returns
    -------
    certif_results: pandas DataFrame
        each row contains data of the form [index, y_real, y_pred, radius, pA], which describe the 
        index of the datapoint examined, the real and predicted class, as well as the radius certified
        by smoothing and the lower confidence bound probability pA

    """
    # init a smoothed classifier
    base_classifier.eval()
    model = SmoothClassifier(base_classifier = base_classifier, sigma = sigma, num_classes = num_classes, device = device)
        
    sampler = NthSampleSampler(dataset, skip)
    test_loader = DataLoader(dataset, batch_size = 1, sampler = sampler)
    
    
    #test_loader = DataLoader(dataset, batch_size = 1, shuffle = False)
    # inits
    cnt = -skip # counts the datapoints
    data = [] # will hold the certification results
    col_names = ['index', 'y_real', 'y_pred', 'radius', 'pA', 'time']
    
    #for x, y in tqdm.tqdm(test_loader, total = len(dataset)):
    for x, y in test_loader:
        cnt += skip
        idx = cnt # index of the current data point
        #if cnt % skip != 0:
            #continue
        # end if
        print(cnt)
        x = x.to(device)
        # try to certify x
        t0 = time.time()
        pred_class, radius, pA = model.certify_nopred(x, num_samples_2, alpha = alpha,
                                           batch_size = certification_batch_size)
        t1 = time.time()
        dt = t1 - t0
        # check certif results
        y_real = int(y)
        res = [idx, y_real, pred_class, radius, pA, dt]
        data.append(res)
    # end for

    # cast results as dataframe
    certif_results = pd.DataFrame(data, columns = col_names)
    return certif_results
# end func
