#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval RS for various sample sizes n and noise levels sigma (CIFAR-10) on [Salman et al., 2020] models
"""

#%% imports

import tqdm
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torchvision.models.resnet import resnet50
from torchvision import transforms, datasets
from typing import *
import torch
import os
from torch.utils.data import Dataset

# own modules
from ..utils.cifar_resnet import resnet as resnet_cifar
from ..utils import eval_smooth


#%% params

# list of all datasets
DATASETS = ['imagenet', 'cifar10']

# resnet50 - the classic ResNet-50, sized for ImageNet
# cifar_resnet20 - a 20-layer residual network sized for CIFAR
# cifar_resnet110 - a 110-layer residual network sized for CIFAR
ARCHITECTURES = ['resnet50', 'cifar_resnet20', 'cifar_resnet110']

# params for norm layer
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STDDEV = [0.229, 0.224, 0.225]

_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STDDEV = [0.2023, 0.1994, 0.2010]

device = 'cuda'


#%% load data (cifar 10)

trainset = datasets.CIFAR10('../data', train = True, download = True, transform = transforms.Compose([
            transforms.RandomCrop(32, padding = 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]))

testset = datasets.CIFAR10('../data', train = False, download = True, transform = transforms.ToTensor())


#%% get normalization layer

class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.

      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means: List[float], sds: List[float], device: str):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        :param device: cpu or cuda
        """
        super(NormalizeLayer, self).__init__()
        self.device = device
        self.means = torch.tensor(means).to(device)
        self.sds = torch.tensor(sds).to(device)
    # end func

    def forward(self, input: torch.tensor):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return (input - means) / sds
    # end func
# end class


def get_normalize_layer(dataset: str, device: str) -> torch.nn.Module:
    """Return the dataset's normalization layer"""
    if dataset == "imagenet":
        #return NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV, device)
        return InputCenterLayer(_IMAGENET_MEAN, device) # slaman uses this normal for imagenet
    elif dataset == "cifar10":
        return NormalizeLayer(_CIFAR10_MEAN, _CIFAR10_STDDEV, device)
        #return InputCenterLayer(_CIFAR10_MEAN, device) # slaman uses this normal for imagenet
    # end if
# end func


#%% get input center layer

class InputCenterLayer(torch.nn.Module):
    """Centers the channels of a batch of images by subtracting the dataset mean.
      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means: List[float], device: str):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(InputCenterLayer, self).__init__()
        self.device = device
        self.means = torch.tensor(means).to(device)
    # end func

    def forward(self, input: torch.tensor):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return input - means
    # end func
# end class


#%% get architecture to use

def get_architecture(arch: str, dataset: str, device: str) -> torch.nn.Module:
    """ Return a neural network (with random weights)

    :param arch: the architecture - should be in the ARCHITECTURES list above
    :param dataset: the dataset - should be in the datasets.DATASETS list
    :return: a Pytorch module
    """
    if arch == 'resnet50' and dataset == 'imagenet':
        model = resnet50(pretrained = False).to(device)
    elif arch == 'cifar_resnet20':
        model = resnet_cifar(depth = 20, num_classes = 10).to(device)
    elif arch == 'cifar_resnet110':
        model = resnet_cifar(depth = 110, num_classes = 10).to(device)
    normalize_layer = get_normalize_layer(dataset, device)
    return torch.nn.Sequential(normalize_layer, model)
# end func


#%% run all experiments

dset = 'cifar10'
num_classes = 10
arch = 'cifar_resnet110'
device = 'cuda'
skip = 20
certification_batch_size = 5000

alpha = 0.001
sigmas_str = ['0.12', '0.25', '0.50', '1.00']
ns = [25, 50, 100, 250, 500, 1000, 10000]
#ns = [10000]



checkpoint_paths = [
    '../smoothing_models_salman/models/pretrained_models/cifar10/finetune_cifar_from_imagenetPGD2steps/self_training_PGD_4steps/weight_1.0/eps_127/cifar10/resnet110/noise_0.12/checkpoint.pth.tar',
    '../smoothing_models_salman/models/pretrained_models/cifar10/finetune_cifar_from_imagenetPGD2steps/self_training_PGD_6steps/weight_1.0/eps_255/cifar10/resnet110/noise_0.25/checkpoint.pth.tar',
    '../smoothing_models_salman/models/pretrained_models/cifar10/finetune_cifar_from_imagenetPGD2steps/self_training_PGD_8steps/weight_0.1/eps_512/cifar10/resnet110/noise_0.50/checkpoint.pth.tar',
    '../smoothing_models_salman/models/pretrained_models/cifar10/finetune_cifar_from_imagenetPGD2steps/self_training_PGD_4steps/weight_0.5/eps_512/cifar10/resnet110/noise_1.00/checkpoint.pth.tar'
    ]

# save results as dicts
pA_all = {}
rA_all = {}

print('RS smoothing results...')

for i in tqdm.tqdm(range(len(sigmas_str))):
    
    sigma_str = sigmas_str[i]
    sigma = float(sigma_str)
    # make path to correct checkpoint
    the_arch = arch[arch.find('_') + 1:]
    #checkpoint_path = 'models/{}/{}/noise_{}/checkpoint.pth.tar'.format(dset, the_arch, sigma_str)
    checkpoint_path = checkpoint_paths[i]

    # load model
    checkpoint = torch.load(checkpoint_path)
    base_classifier = get_architecture(checkpoint['arch'], dset, device)
    base_classifier.load_state_dict(checkpoint['state_dict'])

    for n in ns:
        num_samples_2 = n
        certif_results_rs = eval_smooth.evaluate_robustness_smoothing_nopred(base_classifier, sigma, testset, 
                                                                   num_samples_2, alpha, 
                                                                   certification_batch_size, num_classes, device, skip)

        # save results
        fname = '../results/results_RS_CIFAR_Salman3_nopred_{}_{}_sigma_{}_N_{}_alpha_{}_certbatch_{}_nclasses_{}_skip_{}.pkl'.format(dset, arch, sigma,
                                                                                                           num_samples_2, alpha, 
                                                                                                           certification_batch_size, num_classes, skip)

        # save it
        # https://www.statology.org/pandas-save-dataframe/
        certif_results_rs.to_pickle(fname)

        # load results
        # certif_results_rs = pd.read_pickle(fname)

        # print averages
        n_data = len(certif_results_rs)
        n_abstains = sum(certif_results_rs['y_pred'] == -1)
        false_preds = sum( certif_results_rs['y_pred'] != certif_results_rs['y_real'] )
        corr = certif_results_rs['y_pred'] == certif_results_rs['y_real']
        aver_radius = (certif_results_rs['radius'] * corr ).mean()
        aver_pA = certif_results_rs['pA'].mean()
        aver_time = certif_results_rs['time'].mean()

        # print
        print('Smoothing results (no pred) for sigma = {}, alpha = {}, N = {}:'.format(sigma, alpha, num_samples_2))
        print('percentage of abstains: ', n_abstains / n_data * 100)
        print('percentage of false predictions: ', false_preds / n_data * 100)
        print('average certified radius: ', aver_radius)
        print('average pA lower bound: ', aver_pA)
        print('average certification time (s): ', aver_time)
        
        # save results to dics
        pA_all[(sigma_str, n)] = aver_pA
        rA_all[(sigma_str, n)] = aver_radius
    # end for
# end for

# pickle all results
with open('../results/pA_all_cifar_salman.pickle', 'wb') as handle:
    pickle.dump(pA_all, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('../results/rA_all_cifar_salman.pickle', 'wb') as handle:
    pickle.dump(rA_all, handle, protocol=pickle.HIGHEST_PROTOCOL)

