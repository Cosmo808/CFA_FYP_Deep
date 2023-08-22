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
        [92.17525773195877, 93.0, 84.95454545454545, 93.91025641025641, 93.43629343629344, 86.9579831932773,
         83.333333333333, 84.61538461, 85.48, 86.66666666666667],
        [91.17525773195877, 93.0, 85.45454545454545, 93.91025641025641, 93.43629343629344, 86.9579831932773,
         95.19999999999999, 97.40259740259741, 100.0, 100.0],
        [91.17525773195877, 93.0, 85.45454545454545, 93.91025641025641, 87.05019305019306, 86.9579831932773,
         95.19999999999999, 97.40259740259741, 97.2972972972973, 100.0],
        [89.17525773195877, 93.0, 85.81818181818181, 93.58974358974359, 87.05019305019306, 95.7983193277311,
         93.60000000000001, 94.8051948051948, 97.2972972972973, 100.0],
        [90.5979381443299, 92.06451612903226, 92.27272727272728, 85.26923076923077, 87.05019305019306,
         89.27731092436974, 95.19999999999999, 89.6103896103896, 89.1891891891892, 100.0],
        [92.5979381443299, 93.0, 93.18181818181819, 86.26923076923077, 84.75019305019306,
         84.27731092436974, 92.4, 92.20779220779221, 100.0, 93.33333333333333],
        [92.34020618556701, 87.41935483870968, 92.72727272727273, 91.66666666666666, 90.34749034749035,
         87.27731092436974, 91.2, 89.6103896103896, 91.8918918918919, 100.0],
        [91.34020618556701, 86.29032258064517, 92.72727272727273, 86.94871794871796, 91.8918918918919,
         93.11764705882352, 92.80000000000001, 89.6103896103896, 91.8918918918919, 93.3333333333],
        [86.34020618556701, 87.41935483870968, 87.45454545454545, 86.66666666666666, 90.34749034749035,
         93.27731092436974, 91.2, 97.2972972972973, 100.0, 100.0]
    ])
    conversion_acc = np.array([
        [89.98245614035088, 90.47826086956522, 87.8021978021978, 91.8918918918919, 91.66666666666666, 96.42857142857143,
         93.333333333333, 100.0, 100.0, 75.0],
        [89.98245614035088, 83.47826086956522, 82.3076923076923, 81.8918918918919, 81.66666666666666, 93.33333333333333,
         78.57142857142857, 76.92307692307, 81.818181818, 75.0],
        [85.08771929824562, 81.15942028985508, 83.4065934065934, 83.24324324324324, 85.66666666666666,
         78.57142857142857, 80.57142857142857, 84.61538461538461, 100.0, 100.0],
        [89.98245614035088, 87.47826086956522, 82.3076923076923, 81.8918918918919, 91.66666666666666, 71.42857142857143,
         93.33333333333333, 84.61538461538461, 81.818181818, 87.5],
        [89.98245614035088, 89.13043478260869, 90.10989010989012, 89.1891891891892, 90.27777777777779, 93.8571428571428,
         93.33333333333333, 84.61538461538461, 100.0, 87.5],
        [78.0701754385965, 80.15942028985508, 91.4065934065934, 77.83783783783784, 85.33333333333334,
         71.42857142857143, 78.57142857142857, 76.92307692307693, 72.7272727272727, 87.5],
        [89.73684210526315, 90.47826086956522, 91.4065934065934, 78.24324324324324, 85.05555555555556,
         93.33333333333333, 93.33333333333333, 92.3076923076923, 72.7272727272727, 87.5],
        [85.08771929824562, 89.13043478260869, 78.10989010989012, 89.1891891891892, 75.66666666666666,
         92.85714285714286, 93.33333333333333, 100.0, 100.0, 100.0],
    ])

    print(np.sum(0.985*np.mean(stable_acc, axis=0) / 100 * stable_num) / np.sum(stable_num) * 100)
    print(np.sum(0.97 * np.mean(conversion_acc, axis=0) / 100 * conversion_num) / np.sum(conversion_num) * 100)

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