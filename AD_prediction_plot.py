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
    age = np.array([0.5, 1., 1.5, 2., 2.5, 3., 4., 5., 6., 7., 8.])

    stable_num = np.array([391, 621, 277, 259, 106, 207, 119, 125, 77, 37, 15])
    stable_cn_num = np.array([157.0, 320.0, 149.0, 167.0, 77.0, 130.0, 90.0, 94.0, 60.0, 32.0, 15.0])
    stable_ad_num = np.array([234.0, 301.0, 128.0, 92.0, 29.0, 77.0, 29.0, 31.0, 17.0, 5.0, 0.0])

    conversion_num = np.array([114, 205, 190, 103, 107, 112, 65, 101, 68, 45, 20])

    stable_acc = np.array([[83.88746803069054, 84.70209339774557, 83.03249097472924, 90.73359073359073, 80.18867924528303, 93.23671497584542, 94.9579831932773, 96.0, 97.40259740259741, 100.0, 100.0],
                           [83.88746803069054, 83.41384863123994, 85.1985559566787, 92.66409266409266, 88.67924528301887, 92.7536231884058, 95.7983193277311, 94.39999999999999, 97.40259740259741, 100.0, 100.0],
                           [83.12020460358056, 81.80354267310788, 83.03249097472924, 88.8030888030888, 87.73584905660378, 86.47342995169082, 94.9579831932773, 93.60000000000001, 92.20779220779221, 97.2972972972973, 100.0],
                           [89.00255754475704, 89.85507246376811, 84.83754512635379, 89.96138996138995, 81.13207547169812, 95.16908212560386, 94.9579831932773, 96.0, 97.40259740259741, 100.0, 100.0],
                           [93.86189258312021, 90.49919484702093, 83.39350180505414, 89.96138996138995, 75.47169811320755, 95.65217391304348, 94.9579831932773, 95.19999999999999, 97.40259740259741, 100.0, 100.0],
                           [93.86189258312021, 90.66022544283415, 83.75451263537906, 90.34749034749035, 75.47169811320755, 95.65217391304348, 94.9579831932773, 95.19999999999999, 97.40259740259741, 100.0, 100.0],
                           [89.00255754475704, 89.85507246376811, 84.83754512635379, 89.96138996138995, 81.13207547169812, 95.16908212560386, 94.9579831932773, 96.0, 97.40259740259741, 100.0, 100.0]])

    conversion_acc = np.array([[88.59649122807018, 90.73170731707317, 88.94736842105263, 91.2621359223301, 90.65420560747664, 89.28571428571429, 92.3076923076923, 94.05940594059405, 97.05882352941177, 97.77777777777777, 85.0],
                               [89.47368421052632, 94.14634146341463, 99.47368421052632, 96.11650485436894, 98.13084112149532, 95.53571428571429, 98.46153846153847, 99.00990099009901, 100.0, 100.0, 100.0],
                               [89.47368421052632, 97.07317073170731, 100.0, 99.02912621359224, 100.0, 99.10714285714286, 100.0, 100.0, 100.0, 100.0, 100.0],
                               [89.47368421052632, 97.5609756097561, 100.0, 99.02912621359224, 100.0, 99.10714285714286, 100.0, 100.0, 100.0, 100.0, 100.0],
                               [76.31578947368422, 88.78048780487805, 95.78947368421052, 95.14563106796116, 100.0, 93.75, 96.92307692307692, 98.01980198019803, 98.52941176470588, 100.0, 100.0]])

    fig, axes = plt.subplots(2, 2)
    plt.subplots_adjust(wspace=0.2, hspace=0.5)

    axe = axes[0][0]
    axe.plot(age, np.mean(stable_acc, axis=0), 'o-', color='black', alpha=0.8, linewidth=2)
    axe.plot(age, np.min(stable_acc, axis=0), '-', color='black', alpha=1.0, linewidth=1)
    axe.plot(age, np.max(stable_acc, axis=0), '-', color='black', alpha=0.8, linewidth=1)
    axe.set_xlim([0, 8])
    axe.set_ylim([60, 100])
    axe.set_xlabel('prediction age gap (year)')
    axe.set_ylabel('prediction accuracy (%)')
    axe.set_title('stable CN and AD')

    axe = axes[0][1]
    axe.plot(age, np.mean(conversion_acc, axis=0), 'o-', color='black', alpha=0.8, linewidth=2)
    axe.set_xlim([0, 8])
    axe.set_ylim([60, 100])
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