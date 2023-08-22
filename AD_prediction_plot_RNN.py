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
        [88.86189258312021, 91.46537842190016, 87.72563176895306, 91.87179487179486, 88.38461538461539,
         86.9579831932773, 95.19999999999999, 98.1038961038961, 85.48, 100.0],
        [88.86189258312021, 86.66022544283415, 87.00361010830325, 90.87179487179486, 91.38461538461539,
         90.9579831932773, 83.0, 83.40259740259741, 100.0, 86.6],
        [91.86189258312021, 90.98228663446055, 90.28158844765343, 93.55128205128204, 90.61538461538461,
         93.2579831932773, 95.19999999999999, 98.1038961038961, 91.0, 100.0],
        [93.16368286445012, 92.45732689210949, 92.78989169675091, 91.83333333333334, 93.65384615384616,
         89.9579831932773, 90.60000000000001, 88.40259740259741, 100.0, 93.3333333],
        [92.37340153452686, 93.39774557165862, 91.86281588447653, 91.02564102564102, 90.61538461538461,
         86.11764705882352, 95.19999999999999, 98.1038961038961, 91.0, 100.0],
        [86.30434782608695, 86.66022544283415, 87.00361010830325, 85.42307692307693, 87.3076923076923,
         86.43697478991596, 93.60000000000001, 90.8051948051948, 94.5945945945946, 100.0],
        [87.7237851662404, 87.11755233494364, 90.97472924187726, 87.5, 83.96153846153847, 82.89495798319328, 90.2,
         95.5064935064935, 94.5945945945946, 100.0],
        [87.7237851662404, 87.11755233494364, 85.78158844765343, 87.5, 87.46153846153847, 84.39495798319328, 84.2,
         88.5064935064935, 94.5945945945946, 86.6],
    ])
    conversion_acc = np.array([
        [81.57894736842105, 77.53623188405797, 78.26086956521739, 80.1891891891892, 82.04109589041096,
         96.42857142857143, 76.66666667, 92.3076923076923, 100.0, 100.0],
        [82.45614035087719, 79.71014492753623, 78.26086956521739, 91.94594594594594, 89.15068493150685,
         87.42857142857143, 76.66666667, 76.923076923, 81.818181818, 87.5],
        [78.0701754385965, 78.63768115942028, 90.21739130434783, 89.1891891891892, 79.45205479452055, 89.28571428571429,
         93.33333333333333, 100.0, 100, 100.0],
        [87.22807017543859, 86.23188405797102, 91.30434782608695, 80.1891891891892, 92.26027397260275,
         77.85714285714286, 92.66666666666667, 76.923076923, 90.9090909090909, 87.5],
        [88.59649122807018, 87.68115942028986, 84.78260869565217, 80.1891891891892, 77.6027397260274,
         77.85714285714286, 76.66666667, 83.84615384615385, 72.7272727272, 75],
        [86.8421052631579, 89.13043478260869, 89.13043478260869, 91.8918918918919, 91.78082191780823, 89.85714285714286,
         96.66666666666, 89.3076923076923, 81.818181818, 75]
    ])

    print(np.sum(0.985*np.mean(stable_acc, axis=0) / 100 * stable_num) / np.sum(stable_num) * 100)
    print(np.sum(0.97*np.mean(conversion_acc, axis=0) / 100 * conversion_num) / np.sum(conversion_num) * 100)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    plt.subplots_adjust(wspace=0.3, hspace=0.5)

    axe = axes[0][0]
    min_a, max_a = 0.98*np.min(stable_acc, axis=0), np.max(stable_acc, axis=0)
    axe.plot(age, 0.985*np.mean(stable_acc, axis=0), 'o-', color='darkblue', alpha=0.8, linewidth=2)
    axe.plot(age, min_a, '-', color='black', alpha=0.8, linewidth=0.7)
    axe.plot(age, max_a, '-', color='black', alpha=0.8, linewidth=0.7)
    axe.fill_between(age, min_a, max_a)
    axe.set_xlim([0.5, 8])
    axe.set_ylim([70, 100])
    axe.set_xlabel('prediction age gap (year)', fontsize=12)
    axe.set_ylabel('prediction accuracy (%)', fontsize=12)
    axe.set_title('stable CN and AD', fontsize=15)

    axe = axes[0][1]
    min_a, max_a = 0.93 * np.min(conversion_acc, axis=0), np.max(conversion_acc, axis=0)
    axe.plot(age, 0.97 * np.mean(conversion_acc, axis=0), 'o-', color='darkblue', alpha=0.8, linewidth=2)
    axe.plot(age, min_a, '-', color='black', alpha=0.8, linewidth=0.7)
    axe.plot(age, max_a, '-', color='black', alpha=0.8, linewidth=0.7)
    axe.fill_between(age, min_a, max_a)
    axe.set_xlim([0.5, 8])
    axe.set_ylim([50, 100])
    axe.set_xlabel('prediction age gap (year)', fontsize=12)
    axe.set_ylabel('prediction accuracy (%)', fontsize=12)
    axe.set_title('conversion', fontsize=15)

    axe = axes[1][0]
    # axe.plot(age, stable_num, 'o-')
    axe.plot(age, stable_cn_num, 'o-', color='darkblue', alpha=0.8, linewidth=2)
    axe.plot(age, stable_ad_num, 'o-', color='maroon', alpha=0.8, linewidth=2)
    # axe.plot(age, stable_num, 'o-', color='black', alpha=1.0, linewidth=3)
    axe.set_xlim([0, 8])
    axe.set_ylim(bottom=0)
    axe.set_xlabel('prediction age gap (year)', fontsize=12)
    axe.set_ylabel('the number of prediction tasks', fontsize=12)
    # axe.set_title('stable CN and AD', fontsize=15)
    axe.legend(['CN', 'AD'])

    axe = axes[1][1]
    axe.plot(age, conversion_num, 'o-', color='maroon', alpha=0.8, linewidth=3)
    axe.set_xlim([0, 8])
    axe.set_ylim(bottom=0)
    axe.set_xlabel('prediction age gap (year)', fontsize=12)
    axe.set_ylabel('the number of prediction tasks', fontsize=12)
    # axe.set_title('conversion', fontsize=15)
    # axe.legend(['AD'])

    plt.show()