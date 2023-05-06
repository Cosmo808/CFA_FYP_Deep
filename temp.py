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

    logger.info(f"Device is {device}")

    # load data
    data_generator = Data_preprocess_ADNI(number=163842)
    data_generator.device = device
    # demo_train, demo_test = data_generator.generate_demo_train_test(fold)
    # thick_train, thick_test, input_dim = data_generator.generate_thick_train_test(fold)
    demo_train, demo_test, thick_train, thick_test = data_generator.generate_orig_data()
    logger.info(f"Loaded {len(demo_train['age']) + len(demo_test['age'])} scans")

    demo = demo_train
    thick = thick_train
    CN = torch.nonzero(demo['label'] == 0)
    MCI = torch.cat((torch.nonzero(demo['label'] == 1), torch.nonzero(demo['label'] == 2)), dim=0)
    AD = torch.nonzero(demo['label'] == 3)

    age_CN = demo['age'][CN].squeeze()
    age_MCI = demo['age'][MCI].squeeze()
    age_AD = demo['age'][AD].squeeze()

    a_CN = np.round(np.arange(min(age_CN), max(age_CN), 0.1), 1)
    a_MCI = np.round(np.arange(min(age_MCI), max(age_MCI), 0.1), 1)
    a_AD = np.round(np.arange(min(age_AD), max(age_AD), 0.1), 1)

    aa_CN, aa_MCI, aa_AD = [], [], []
    cn_temp, mci_temp, ad_temp = [], [], []
    i, imax = 0, 10
    for ac, am, aa in zip(a_CN, a_MCI, a_AD):
        if i == 0:
            cn_temp = torch.nonzero(age_CN == ac)
            mci_temp = torch.nonzero(age_MCI == am)
            ad_temp = torch.nonzero(age_AD == aa)
        else:
            cn_temp = torch.cat((cn_temp, torch.nonzero(age_CN == ac)), dim=0)
            mci_temp = torch.cat((mci_temp, torch.nonzero(age_MCI == am)), dim=0)
            ad_temp = torch.cat((ad_temp, torch.nonzero(age_AD == aa)), dim=0)
        i += 1
        if i == 10:
            i = 0
            aa_CN.append(cn_temp)
            aa_MCI.append(mci_temp)
            aa_AD.append(ad_temp)

    lt, rt = thick['left'], thick['right']
    CN = CN.view(1, -1).squeeze().numpy()
    MCI = np.sort(MCI.view(1, -1).squeeze().numpy())
    AD = AD.view(1, -1).squeeze().numpy()
    lt_CN, lt_MCI, lt_AD = lt[CN], lt[MCI], lt[AD]
    avg_lt_CN = np.zeros(shape=[len(aa_CN), lt_CN.shape[1]])
    avg_lt_MCI = np.zeros(shape=[len(aa_MCI), lt_MCI.shape[1]])
    avg_lt_AD = np.zeros(shape=[len(aa_AD), lt_AD.shape[1]])

    for i, aa in enumerate(aa_CN):
        aa = aa.view(1, -1).squeeze().numpy()
        try:
            avg = np.sum(lt_CN[aa], axis=0) / len(aa)
        except TypeError:
            avg = lt_CN[aa]
        avg_lt_CN[i] = avg

    for i, aa in enumerate(aa_MCI):
        aa = aa.view(1, -1).squeeze().numpy()
        try:
            avg = np.sum(lt_MCI[aa], axis=0) / len(aa)
        except TypeError:
            avg = lt_MCI[aa]
        avg_lt_MCI[i] = avg

    for i, aa in enumerate(aa_AD):
        aa = aa.view(1, -1).squeeze().numpy()
        try:
            avg = np.sum(lt_AD[aa], axis=0) / len(aa)
        except TypeError:
            avg = lt_AD[aa]
        avg_lt_AD[i] = avg

    lt_mat = {'CN': avg_lt_CN, 'MCI': avg_lt_MCI, 'AD': avg_lt_AD}
    scipy.io.savemat('/home/ming/Desktop/lt_avg.mat', lt_mat)