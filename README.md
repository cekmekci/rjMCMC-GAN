# Learning the Distribution of an Ensemble of Models Obtained by a Reversible Jump Markov Chain Monte Carlo Algorithm Using Generative Adversarial Networks

[![DOI](https://zenodo.org/badge/669880939.svg)](https://zenodo.org/badge/latestdoi/669880939)

[Tolulope Olugboji](https://scholar.google.com/citations?user=GnxCk8EAAAAJ&hl=en), Enting Zhou, Walter Hennings, [Canberk Ekmekci](https://cekmekci.github.io/) 

[UR Seismo](http://www.sas.rochester.edu/ees/urseismo/), Rochester, NY USA

This repo contains two-dimensional toy problems for which the distribution of an ensemble of models provided by a reversible jump Markov Chain Monte Carlo algorithm is learned using a Wasserstein generative adversarial network.

## Requirements

- PyTorch = 1.13.1
- Scipy = 1.7.3

## Data

An example toy problem is provided inside the "data" folder. Due to space limitations, other two-dimensional toy problems are provided in [this Google Drive folder](https://drive.google.com/file/d/1G7moiSi-FI1KriPFF_4MSIx_bnHEx_mi/view?usp=sharing). 


## Use

You can train a Wasserstein generative adversarial network for a two-dimensional toy problem by running the first two cells of the "main.ipynb" Jupyter notebook. To generate samples from the trained generative adversarial network and to calculate the mean and the standard deviation of the distribution learned by the trained generative adversarial network, you can run the remaining cells of the notebook. With default parameters provided in the notebook, on an NVIDIA A10 GPU, runtime of the training stage is around 5 minutes, and generating 10000 samples from the trained generative adversarial network takes less than 5 seconds. 

## Note

This repository contains a slightly modified version of the code provided in [this Github repository](https://github.com/ETZET/MCMC_GAN). If you have any problems or questions, please feel free to [contact us](http://www.sas.rochester.edu/ees/urseismo/current-members/). 




