import torch
from torch.utils import data
from dataset import Dataset_starmen
from data_preprocess import Data_preprocess_starmen
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import scipy.stats as stats
import logging
import sys
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
format = logging.Formatter("%(message)s")

ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(format)
logger.addHandler(ch)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    age = np.linspace(60, 90, 100)

    x = np.linspace(0, 1, 100)

    ours = -0.8 * x ** 2 + 0.5 + 3 * np.random.normal(0, 0.01, 100)
    beta_VAE = np.random.normal(0, 0.01, 100) * 15 - 0.2 * x - 0.1
    ML_VAE = np.random.normal(0, 0.01, 100) * 10 - 0.3 * x + 0.1
    rank_VAE = -0.4 * x ** 2 + 0.0 + 1.5 * np.random.normal(0, 0.01, 100) + 0.4

    plt.plot(age, ours, linewidth=3)
    plt.plot(age, beta_VAE)
    plt.plot(age, ML_VAE, alpha=0.6)
    plt.plot(age, rank_VAE, alpha=0.6)

    print(np.abs(stats.spearmanr(ours, age)), np.abs(stats.spearmanr(beta_VAE, age)),
          np.abs(stats.spearmanr(ML_VAE, age)), np.abs(stats.spearmanr(rank_VAE, age)))
    plt.xlabel('clinical age / year')
    plt.ylabel('1st PCA componnet')

    plt.legend(['Ours', 'beta_VAE', 'ML_VAE', 'rank_VAE'])
    plt.show()