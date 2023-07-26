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

    conversion_num = np.array([114, 205, 190, 156, 166, 65, 101, 68, 45, 20])

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
        [87.71929824561403, 91.70731707317074, 97.89473684210527, 98.71794871794873, 98.19277108433735,
         98.46153846153847, 99.00990099009901, 100.0, 100.0, 100.0],
        [87.71929824561403, 92.19512195121952, 100.0, 98.71794871794873, 98.19277108433735, 100.0, 99.00990099009901,
         100.0, 100.0, 100.0],
        [87.71929824561403, 92.19512195121952, 100.0, 98.71794871794873, 98.19277108433735, 100.0, 99.00990099009901,
         100.0, 100.0, 100.0],
        [88.59649122807018, 82.92682926829268, 83.6842105263158, 80.12820512820514, 75.90361445783132,
         78.46153846153847, 84.15841584158416, 83.82352941176471, 82.22222222222221, 90.0],
        [92.10526315789474, 83.90243902439025, 83.6842105263158, 82.05128205128204, 79.51807228915662,
         81.53846153846153, 86.13861386138613, 86.76470588235294, 88.88888888888889, 100.0],
        [92.98245614035088, 85.85365853658537, 83.6842105263158, 83.33333333333334, 81.92771084337349,
         86.15384615384616, 90.0990099009901, 89.70588235294117, 91.11111111111111, 100.0],
        [93.85964912280701, 88.78048780487805, 83.6842105263158, 85.25641025641025, 89.1566265060241, 93.84615384615384,
         92.07920792079209, 91.17647058823529, 93.33333333333333, 100.0],
        [93.85964912280701, 90.2439024390244, 84.73684210526315, 87.17948717948718, 90.36144578313254,
         95.38461538461539, 96.03960396039604, 95.58823529411765, 95.55555555555556, 100.0],
        [93.85964912280701, 91.70731707317074, 85.78947368421052, 90.38461538461539, 93.37349397590361,
         96.92307692307692, 97.02970297029702, 97.05882352941177, 97.77777777777777, 100.0]
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