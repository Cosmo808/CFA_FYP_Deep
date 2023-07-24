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
    stable_cn_num = np.array([34.0, 123.0, 47.0, 273.0, 112.0, 37.0, 167.0, 73.0, 4.0, 104.0,
                              26.0, 5.0, 85.0, 34.0, 60.0, 29.0, 31.0, 21.0, 11.0, 7.0, 4.0, 4.0])
    stable_ad_num = np.array([18.0, 216.0, 21.0, 280.0, 123.0, 5.0, 92.0, 29.0, 0.0, 51.0,
                              26.0, 0.0, 29.0, 15.0, 16.0, 9.0, 8.0, 3.0, 2.0, 0.0, 0.0, 0.0])

    conversion_num = np.array([10, 104, 23, 182, 153, 37, 103, 104, 3, 56, 56, 7,
                               58, 56, 45, 38, 30, 29, 16, 16, 2, 2])

    stable_acc = np.array([94.2, 88.5, 79.4, 90.1, 84.7, 83.3, 91.1, 84.3, 75, 90.2, 92.3, 100,
                           94.7, 87.8, 96.1, 89.5, 97.4, 95.3, 92.3, 100, 100, 100])
    conversion_acc = np.array([80, 88.5, 91.3, 90.1, 88.2, 81.1, 89.3, 88.5, 66.7, 87.5, 91.1,
                               85.7, 91.4, 92.9, 93.3, 94.7, 96.7, 86.2, 81.3, 81.3, 100, 100])

    fig, axes = plt.subplots(2, 2)
    plt.subplots_adjust(wspace=0.2, hspace=0.5)

    axe = axes[0][0]
    axe.plot(age, stable_acc, 'o-', color='black', alpha=0.8, linewidth=2)
    axe.set_xlim([0, 9])
    axe.set_ylim([60, 100])
    axe.set_xlabel('prediction age gap (year)')
    axe.set_ylabel('prediction accuracy (%)')
    axe.set_title('stable CN and AD')

    axe = axes[0][1]
    axe.plot(age, conversion_acc, 'o-', color='black', alpha=0.8, linewidth=2)
    axe.set_xlim([0, 9])
    axe.set_ylim([60, 100])
    axe.set_xlabel('prediction age gap (year)')
    axe.set_ylabel('prediction accuracy (%)')
    axe.set_title('conversion')

    axe = axes[1][0]
    # axe.plot(age, stable_num, 'o-')
    axe.plot(age, stable_cn_num, 'o-', color='darkblue', alpha=0.8, linewidth=2)
    axe.plot(age, stable_ad_num, 'o-', color='maroon', alpha=0.8, linewidth=2)
    # axe.plot(age, stable_num, 'o-', color='black', alpha=1.0, linewidth=3)
    axe.set_xlim([0, 9])
    axe.set_ylim(bottom=0)
    axe.set_xlabel('prediction age gap (year)')
    axe.set_ylabel('the number of prediction tasks')
    axe.set_title('stable CN and AD')
    axe.legend(['CN', 'AD'])

    axe = axes[1][1]
    axe.plot(age, conversion_num, 'o-', color='maroon', alpha=0.8, linewidth=3)
    axe.set_xlim([0, 9])
    axe.set_ylim(bottom=0)
    axe.set_xlabel('prediction age gap (year)')
    axe.set_ylabel('the number of prediction tasks')
    axe.set_title('conversion')
    # axe.legend(['AD'])

    plt.show()