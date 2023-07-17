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
    age = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 1.75, 2.0, 2.5, 2.75, 3.0, 3.5, 3.75,
                    4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5])
    stable_num = np.array([52, 339, 68, 553, 235, 42, 259, 102, 4, 155, 52, 5,
                           114, 49, 76, 38, 39, 24, 13, 7, 4, 4])
    conversion_num = np.array([10, 104, 23, 182, 153, 37, 103, 104, 3, 56, 56, 7,
                               58, 56, 45, 38, 30, 29, 16, 16, 2, 2])
    stable_acc = np.array([96.1, 92, 80.1, 93.3, 91.4, 76.2, 93.8, 95.1, 75, 93.5,
                           90.4, 100, 92.3, 93.9, 96, 94.7, 94.9, 91.6, 100, 100, 100])
    conversion_acc = np.array([80, 88.5, 91.3, 90.1, 88.2, 81.1, 89.3, 88.5, 66.7, 87.5,
                               91.1, 85.7, 91.4, 92.9, 93.3, 94.7, 96.7, 86.2, 81.3, 81.3, 100, 100])

    fig, axes = plt.subplots(2, 2)
    # plt.subplots_adjust(wspace=0.5, hspace=0.5)

    axe = axes[0][0]
    axe.plot(age, stable_num, '.-')
    axe.set_xlim([0, 9])
    axe.set_xlabel('years from the last known age')
    axe.set_ylabel('the number of prediction tasks')
    axe.set_title('stable CN and AD')

    axe = axes[1][0]
    axe.plot(age, stable_num, '.-')
    axe.set_xlim([0, 9])
    axe.set_xlabel('years from the last known age')
    axe.set_ylabel('the number of prediction tasks')
    axe.set_title('stable CN and AD')

    axe = axes[1][1]
    axe.plot(age, conversion_num, '.-')
    axe.set_xlim([0, 9])
    axe.set_xlabel('years from the last known age')
    axe.set_ylabel('the number of prediction tasks')
    axe.set_title('conversion')

    plt.show()