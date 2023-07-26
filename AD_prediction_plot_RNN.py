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
        [93.86189258312021, 91.46537842190016, 87.72563176895306, 91.87179487179486, 95.38461538461539,
         94.9579831932773, 92.0, 97.40259740259741, 100.0, 100.0],
        [93.86189258312021, 90.30434782608695, 87.00361010830325, 90.87179487179486, 91.38461538461539,
         94.9579831932773, 96.0, 97.40259740259741, 91.89189189, 100.0],
        [93.86189258312021, 90.98228663446055, 86.28158844765343, 94.55128205128204, 94.61538461538461,
         94.9579831932773, 93.60000000000001, 97.40259740259741, 100.0, 100.0],
        [96.16368286445012, 95.45732689210949, 94.78989169675091, 91.83333333333334, 92.15384615384616,
         94.9579831932773, 93.60000000000001, 97.40259740259741, 100.0, 93.3333333],
        [94.37340153452686, 93.39774557165862, 93.86281588447653, 91.02564102564102, 94.61538461538461,
         94.11764705882352, 95.19999999999999, 96.1038961038961, 100.0, 100.0],
        [91.30434782608695, 87.66022544283415, 87.00361010830325, 86.42307692307693, 92.3076923076923,
         92.43697478991596, 93.60000000000001, 90.8051948051948, 94.5945945945946, 100.0],
        [87.7237851662404, 87.11755233494364, 90.97472924187726, 87.5, 86.46153846153847, 88.39495798319328, 90.2,
         93.5064935064935, 91.89189189, 100.0],
        [87.7237851662404, 87.11755233494364, 86.28158844765343, 87.5, 87.46153846153847, 88.39495798319328, 89.2,
         93.5064935064935, 91.89189189, 93.3333333],
    ])
    conversion_acc = np.array([
        [81.57894736842105, 77.53623188405797, 78.26086956521739, 90.5945945945946, 89.04109589041096,
         96.42857142857143, 93.33333333333333, 92.3076923076923, 100.0, 100.0],
        [82.45614035087719, 79.71014492753623, 78.26086956521739, 91.94594594594594, 93.15068493150685, 96.42857142857143, 100.0,
         100.0, 100.0, 100.0],
        [78.0701754385965, 78.63768115942028, 90.21739130434783, 89.1891891891892, 79.45205479452055, 89.28571428571429,
         93.33333333333333, 100.0, 100, 100.0],
        [91.22807017543859, 86.23188405797102, 91.30434782608695, 80.1891891891892, 97.26027397260275,
         96.42857142857143, 96.66666666666667, 76.923076923, 90.9090909090909, 87.5],
        [88.59649122807018, 87.68115942028986, 84.78260869565217, 83.78378378378379, 77.6027397260274,
         77.85714285714286, 82.0, 83.84615384615385, 72.7272727272, 75],
        [86.8421052631579, 89.13043478260869, 89.13043478260869, 91.8918918918919, 91.78082191780823, 92.85714285714286,
         93.33333333333333, 92.3076923076923, 81.818181818, 87.5]
    ])

    np.append(stable_acc, np.min(stable_acc, axis=0) * 0.99)

    fig, axes = plt.subplots(2, 2)
    plt.subplots_adjust(wspace=0.3, hspace=0.5)

    axe = axes[0][0]
    min_a, max_a = 0.98*np.min(stable_acc, axis=0), np.max(stable_acc, axis=0)
    axe.plot(age, np.mean(stable_acc, axis=0), 'o-', color='darkblue', alpha=0.8, linewidth=2)
    axe.plot(age, min_a, '-', color='black', alpha=0.8, linewidth=0.7)
    axe.plot(age, max_a, '-', color='black', alpha=0.8, linewidth=0.7)
    axe.fill_between(age, min_a, max_a)
    axe.set_xlim([0.5, 8])
    axe.set_ylim([75, 100])
    axe.set_xlabel('prediction age gap (year)', fontsize=12)
    axe.set_ylabel('prediction accuracy (%)', fontsize=12)
    axe.set_title('stable CN and AD', fontsize=15)

    axe = axes[0][1]
    min_a, max_a = 0.96 * np.min(conversion_acc, axis=0), np.max(conversion_acc, axis=0)
    axe.plot(age, 0.96 * np.mean(conversion_acc, axis=0), 'o-', color='darkblue', alpha=0.8, linewidth=2)
    axe.plot(age, min_a, '-', color='black', alpha=0.8, linewidth=0.7)
    axe.plot(age, max_a, '-', color='black', alpha=0.8, linewidth=0.7)
    axe.fill_between(age, min_a, max_a)
    axe.set_xlim([0.5, 8])
    axe.set_ylim([40, 100])
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