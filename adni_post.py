import numpy as np
import torch
import torch.optim as optim
from torch.utils import data
from torch.autograd import Variable
import logging
import sys
import os
from dataset import Dataset_adni
from data_preprocess import Data_preprocess_ADNI
import scipy
import argparse
import model_adni


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
format = logging.Formatter("%(message)s")

ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(format)
logger.addHandler(ch)

device = torch.device(f"cuda:0")
ratio = 1.0
fold = 0


if __name__ == '__main__':
    logger.info(f"Device is {device}")

    # load model
    autoencoder = torch.load('model/0_fold_all_left_AE_adni', map_location=device)
    autoencoder.eval()

    # load data
    data_generator = Data_preprocess_ADNI(number=40962, label=-1)
    demo_train, demo_test = data_generator.generate_demo_train_test(fold)
    thick_train, thick_test, input_dim = data_generator.generate_thick_train_test(fold)
    logger.info(f"Loaded {len(demo_train['age']) + len(demo_test['age'])} scans")

    print('Generating data loader finished...')

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
    temp = []
    i, imax = 0, 10
    for a in a_CN:
        if i == 0:
            temp = torch.nonzero(age_CN == a)
        else:
            temp = torch.cat((temp, torch.nonzero(age_CN == a)), dim=0)
        i += 1
        if i == imax:
            i = 0
            temp = temp[0] if temp.nelement() != 0 else torch.tensor([-1])
            aa_CN.append(temp)
    i = 0
    for a in a_MCI:
        if i == 0:
            temp = torch.nonzero(age_MCI == a)
        else:
            temp = torch.cat((temp, torch.nonzero(age_MCI == a)), dim=0)
        i += 1
        if i == imax:
            i = 0
            temp = temp[0] if temp.nelement() != 0 else torch.tensor([-1])
            aa_MCI.append(temp)
    i = 0
    for a in a_AD:
        if i == 0:
            temp = torch.nonzero(age_AD == a)
        else:
            temp = torch.cat((temp, torch.nonzero(age_AD == a)), dim=0)
        i += 1
        if i == imax:
            i = 0
            temp = temp[0] if temp.nelement() != 0 else torch.tensor([-1])
            aa_AD.append(temp)

    lt, rt = thick['left'], thick['right']
    CN = CN.view(1, -1).squeeze().numpy()
    MCI = np.sort(MCI.view(1, -1).squeeze().numpy())
    AD = AD.view(1, -1).squeeze().numpy()
    lt_CN, lt_MCI, lt_AD = lt[CN], lt[MCI], lt[AD]

    aa_CN = torch.tensor(aa_CN).view(1, -1).squeeze().numpy()
    aa_MCI = torch.tensor(aa_MCI).view(1, -1).squeeze().numpy()
    aa_AD = torch.tensor(aa_AD).view(1, -1).squeeze().numpy()
    lt_CN, lt_MCI, lt_AD = lt_CN[aa_CN], lt_MCI[aa_MCI], lt_AD[aa_AD]

    input_CN = Variable(torch.tensor(lt_CN)).to(device).float()
    reconstructed_CN, z_CN, zu_CN, zv_CN = autoencoder.forward(input_CN)

    input_MCI = Variable(torch.tensor(lt_MCI)).to(device).float()
    reconstructed_MCI, z_MCI, zu_MCI, zv_MCI = autoencoder.forward(input_MCI)

    input_AD = Variable(torch.tensor(lt_AD)).to(device).float()
    reconstructed_AD, z_AD, zu_AD, zv_AD = autoencoder.forward(input_AD)

    lt_global_trajectory = {'CN': zu_CN, 'MCI': zu_MCI, 'AD': zu_AD}
    scipy.io.savemat('/home/ming/Desktop/lt_global_trajectory.mat', lt_global_trajectory)