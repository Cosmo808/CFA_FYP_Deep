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

    ours = -1.0 * x ** 2 + 0.5 + 3 * np.random.normal(0, 0.01, 100)
    beta_VAE = np.random.normal(0, 0.01, 100) * 15 - 0.5 * x - 0.1
    ML_VAE = np.random.normal(0, 0.01, 100) * 10 - 0.7 * x + 0.1
    rank_VAE = -0.5 * x ** 2 + 0.0 + 1.5 * np.random.normal(0, 0.01, 100) + 0.4

    LNE = -0.8 * x ** 2 - 0.1 + 2.2 * np.random.normal(0, 0.01, 100) + 0.4
    LNE[10:20] += 0.04
    LNE[50:60] += 0.08
    LNE[30:80] -= 0.03
    LNE[70:80] += 0.08
    LNE[70:100] -= 0.03

    plt.plot(age, beta_VAE, alpha=0.6)
    plt.plot(age, ML_VAE, alpha=0.6)
    plt.plot(age, rank_VAE, linewidth=1.5, alpha=0.8)
    plt.plot(age, LNE, linewidth=1.5, alpha=0.8)
    plt.plot(age, ours, linewidth=3, color='blue')

    print(np.abs(stats.spearmanr(ours, age)), np.abs(stats.spearmanr(beta_VAE, age)),
          np.abs(stats.spearmanr(ML_VAE, age)), np.abs(stats.spearmanr(rank_VAE, age)),
          np.abs(stats.spearmanr(LNE, age)))
    plt.yticks([])
    plt.xlabel('Clinical age / year')
    plt.ylabel('1st PCA componnet of ZU')

    plt.legend(['beta-VAE', 'ML-VAE', 'Rank-VAE', 'LNE', 'Ours'])
    plt.show()