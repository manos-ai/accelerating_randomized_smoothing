# Accelerating Randomized Smoothing

This is the code for the paper "Estimating the Robustness Radius for Randomized Smoothing with 100× Sample Efficiency". 


## Requirements
- pytorch
- torchvision
- numpy
- pandas
- matplotlib
- seaborn
- scipy

Any environment with these packages should do (no specific versions needed). 

We rely on the works and trained models of [(Cohen et al., 2019)](https://github.com/locuslab/smoothing) and [(Salman et al., 2019)](https://github.com/Hadisalman/smoothing-adversarial). We optimize the code so that all experiments, including ImageNet, can run on a single GPU, and substantially faster. 

## Getting started

Simply run the notebook ``main.ipynb``, which presents our main results, and replicates our experiments. 

In order to run all experiments from scratch (without relying on our pre-computed data), the following steps are needed:

- **Datasets**: obtain a copy of [ImageNet](https://www.image-net.org/) and preprocess the ``val`` directory to look like the   train directory by running [this script]( https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh). Place the data inside the ``data`` folder.
- **Models**: Download the trained [models](https://drive.google.com/file/d/1h_TpbXm5haY5f-l4--IKylmdz6tvPoR4/view) of (Cohen et al., 2019), and place them inside the ``models`` folder. Optionally, download also the models of [(Salman et al., 2019)](https://github.com/Hadisalman/smoothing-adversarial) and place them in ``smoothing_models_salman``.

After that, ``main.ipynb`` can run all experiments from the beginning, by uncommenting the indicated code blocks.
