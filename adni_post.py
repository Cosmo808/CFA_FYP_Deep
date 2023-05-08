import numpy
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
from sklearn.manifold import TSNE
from scipy.stats import norm
import matplotlib.pyplot as plt
import argparse
import model_adni


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
format = logging.Formatter("%(message)s")

ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(format)
logger.addHandler(ch)

device = torch.device(f"cuda:1")

parser = argparse.ArgumentParser()
parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--lor', type=int, default=0, help='left (0) or right (1)')
input_para = parser.parse_args()
fold = input_para.fold
left_right = 'left' if input_para.lor == 0 else 'right'


if __name__ == '__main__':
    logger.info(f"Device is {device}")
    # load model
    autoencoder = torch.load('model/{}_fold_all_{}_AE_adni'.format(fold, left_right), map_location=device)
    autoencoder.eval()
    # load data
    data_generator = Data_preprocess_ADNI(number=40962, label=-1)
    demo_train, demo_test, thick_train, thick_test = data_generator.generate_orig_data()
    # demo_train, demo_test = data_generator.generate_demo_train_test(fold)

    # thick_train, thick_test, input_dim = data_generator.generate_thick_train_test(fold)
    # logger.info(f"Loaded {len(demo_train['age']) + len(demo_test['age'])} scans")
    #
    # Dataset = Dataset_adni
    # train = Dataset(thick_train['left'], thick_train['right'], demo_train['age'], demo_train['baseline_age'],
    #                 demo_train['label'], demo_train['subject'], demo_train['timepoint'])
    # test = Dataset(thick_test['left'], thick_test['right'], demo_test['age'], demo_test['baseline_age'],
    #                demo_test['label'], demo_test['subject'], demo_test['timepoint'])
    #
    # train_loader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=False,
    #                                            num_workers=0, drop_last=False)
    # test_loader = torch.utils.data.DataLoader(test, batch_size=128, shuffle=False,
    #                                           num_workers=0, drop_last=False)
    print('Generating data loader finished...')

    # get Z, ZU, ZV
    # with torch.no_grad():
    #     Z, ZU, ZV = None, None, None
    #     for data in train_loader:
    #         image = data[autoencoder.left_right]
    #
    #         # self-reconstruction loss
    #         input_ = Variable(image).to(device).float()
    #         reconstructed, z, zu, zv = autoencoder.forward(input_)
    #         self_reconstruction_loss = autoencoder.loss(input_, reconstructed)
    #
    #         # store Z, ZU, ZV
    #         if Z is None:
    #             Z, ZU, ZV = z, zu, zv
    #         else:
    #             Z = torch.cat((Z, z), 0)
    #             ZU = torch.cat((ZU, zu), 0)
    #             ZV = torch.cat((ZV, zv), 0)
    # train_ZV = ZV
    #
    # with torch.no_grad():
    #     Z, ZU, ZV = None, None, None
    #     for data in test_loader:
    #         image = data[autoencoder.left_right]
    #
    #         # self-reconstruction loss
    #         input_ = Variable(image).to(device).float()
    #         reconstructed, z, zu, zv = autoencoder.forward(input_)
    #         self_reconstruction_loss = autoencoder.loss(input_, reconstructed)
    #
    #         # store Z, ZU, ZV
    #         if Z is None:
    #             Z, ZU, ZV = z, zu, zv
    #         else:
    #             Z = torch.cat((Z, z), 0)
    #             ZU = torch.cat((ZU, zu), 0)
    #             ZV = torch.cat((ZV, zv), 0)
    # test_ZV = ZV

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

    print(a_CN, a_MCI, a_AD)