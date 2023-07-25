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
    age = np.array([0.5, 1.,  1.5, 2., 2.5, 3., 3.5, 4., 5., 6., 7., 8.])

    stable_num = np.array([391, 621, 277, 259, 106, 155, 57, 114, 125, 77, 37, 15])
    stable_cn_num = np.array([157.0, 320.0, 149.0, 167.0, 77.0, 104.0, 31.0, 85.0, 94.0, 60.0, 32.0, 15.0])
    stable_ad_num = np.array([234.0, 301.0, 128.0, 92.0, 29.0, 51.0, 26.0, 29.0, 31.0, 17.0, 5.0, 0.0])

    conversion_num = np.array([114, 205, 190, 103, 107, 56, 63, 58, 101, 68, 45, 20])

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