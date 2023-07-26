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
    age = np.array([0.5, 1., 1.5, 2., 3., 4., 5., 6., 7., 8.])

    stable_num = np.array([391, 621, 277, 312, 260, 119, 125, 77, 37, 15])
    stable_cn_num = np.array([157.0, 320.0, 149.0, 205.0, 169.0, 90.0, 94.0, 60.0, 32.0, 15.0])
    stable_ad_num = np.array([234.0, 301.0, 128.0, 107.0, 91.0, 29.0, 31.0, 17.0, 5.0, 0.0])

    conversion_num = np.array([114, 138, 92, 74, 73, 28, 30, 13, 11, 8])

    stable_acc = np.array([
        [93.86189258312021, 91.46537842190016, 87.72563176895306, 94.87179487179486, 95.38461538461539,
         94.9579831932773, 96.0, 97.40259740259741, 100.0, 100.0],
        [93.86189258312021, 91.30434782608695, 87.00361010830325, 94.87179487179486, 95.38461538461539,
         94.9579831932773, 96.0, 97.40259740259741, 100.0, 100.0],
        [93.86189258312021, 90.98228663446055, 86.28158844765343, 94.55128205128204, 94.61538461538461,
         94.9579831932773, 96.0, 97.40259740259741, 100.0, 100.0],
        [96.16368286445012, 96.45732689210949, 96.38989169675091, 95.83333333333334, 96.15384615384616,
         94.9579831932773, 96.0, 97.40259740259741, 100.0, 100.0],
        [94.37340153452686, 93.39774557165862, 93.86281588447653, 91.02564102564102, 94.61538461538461,
         94.11764705882352, 95.19999999999999, 96.1038961038961, 100.0, 100.0],
        [91.30434782608695, 90.66022544283415, 87.00361010830325, 89.42307692307693, 92.3076923076923,
         92.43697478991596, 93.60000000000001, 94.8051948051948, 94.5945945945946, 100.0],
    ])
    conversion_acc = np.array([
        [61.40350877192983, 76.81159420289855, 90.21739130434783, 86.48648648648648, 94.52054794520548,
         96.42857142857143, 96.66666666666667, 100.0, 100.0, 100.0],
        [61.40350877192983, 76.81159420289855, 90.21739130434783, 86.48648648648648, 94.52054794520548,
         96.42857142857143, 96.66666666666667, 100.0, 100.0, 100.0],

    ])
    fig, axes = plt.subplots(2, 2)
    plt.subplots_adjust(wspace=0.2, hspace=0.5)

    axe = axes[0][0]
    axe.plot(age, np.mean(stable_acc, axis=0), 'o-', color='black', alpha=0.8, linewidth=2)
    axe.plot(age, np.min(stable_acc, axis=0), '-', color='black', alpha=0.8, linewidth=1)
    axe.plot(age, np.max(stable_acc, axis=0), '-', color='black', alpha=0.8, linewidth=1)
    axe.set_xlim([0, 8])
    axe.set_ylim([75, 100])
    axe.set_xlabel('prediction age gap (year)')
    axe.set_ylabel('prediction accuracy (%)')
    axe.set_title('stable CN and AD')

    axe = axes[0][1]
    axe.plot(age, np.mean(conversion_acc, axis=0), 'o-', color='black', alpha=0.8, linewidth=2)
    axe.plot(age, np.min(conversion_acc, axis=0), '-', color='black', alpha=0.8, linewidth=1)
    axe.plot(age, np.max(conversion_acc, axis=0), '-', color='black', alpha=0.8, linewidth=1)
    axe.set_xlim([0, 8])
    axe.set_ylim([50, 100])
    axe.set_xlabel('prediction age gap (year)')
    axe.set_ylabel('prediction accuracy (%)')
    axe.set_title('conversion')

    axe = axes[1][0]
    # axe.plot(age, stable_num, 'o-')
    axe.plot(age, stable_cn_num, 'o-', color='darkblue', alpha=0.8, linewidth=2)
    axe.plot(age, stable_ad_num, 'o-', color='maroon', alpha=0.8, linewidth=2)
    # axe.plot(age, stable_num, 'o-', color='black', alpha=1.0, linewidth=3)
    axe.set_xlim([0, 8])
    axe.set_ylim(bottom=0)
    axe.set_xlabel('prediction age gap (year)')
    axe.set_ylabel('the number of prediction tasks')
    axe.set_title('stable CN and AD')
    axe.legend(['CN', 'AD'])

    axe = axes[1][1]
    axe.plot(age, conversion_num, 'o-', color='maroon', alpha=0.8, linewidth=3)
    axe.set_xlim([0, 8])
    axe.set_ylim(bottom=0)
    axe.set_xlabel('prediction age gap (year)')
    axe.set_ylabel('the number of prediction tasks')
    axe.set_title('conversion')
    # axe.legend(['AD'])

    plt.show()