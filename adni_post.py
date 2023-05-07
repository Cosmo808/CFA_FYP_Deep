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

device = torch.device(f"cuda:0")

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
    #
    # thick_train, thick_test, input_dim = data_generator.generate_thick_train_test(fold)
    logger.info(f"Loaded {len(demo_train['age']) + len(demo_test['age'])} scans")

    Dataset = Dataset_adni
    train = Dataset(thick_train['left'], thick_train['right'], demo_train['age'], demo_train['baseline_age'],
                    demo_train['label'], demo_train['subject'], demo_train['timepoint'])
    test = Dataset(thick_test['left'], thick_test['right'], demo_test['age'], demo_test['baseline_age'],
                   demo_test['label'], demo_test['subject'], demo_test['timepoint'])

    train_loader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=False,
                                               num_workers=0, drop_last=False)
    test_loader = torch.utils.data.DataLoader(test, batch_size=128, shuffle=False,
                                              num_workers=0, drop_last=False)
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

    age = demo_train['age']
    a = np.round(np.arange(min(age), max(age), 0.1), 1)

    aa = []
    temp = []
    i, imax = 0, 10
    for aaa in a:
        if i == 0:
            temp = torch.nonzero(age == aaa)
        else:
            temp = torch.cat((temp, torch.nonzero(age == aaa)), dim=0)
        i += 1
        if i == 10:
            i = 0
            aa.append(temp)

    lt, rt = thick_train['left'], thick_train['right']
    avg_lt = np.zeros(shape=[len(aa), lt.shape[1]])
    avg_rt = np.zeros(shape=[len(aa), lt.shape[1]])
    for i, a in enumerate(aa):
        a = a.view(1, -1).squeeze().numpy()
        try:
            avg = np.sum(lt[a], axis=0) / len(a)
        except TypeError:
            avg = lt[a]
        avg_lt[i] = avg
    for i, a in enumerate(aa):
        a = a.view(1, -1).squeeze().numpy()
        try:
            avg = np.sum(rt[a], axis=0) / len(a)
        except TypeError:
            avg = rt[a]
        avg_rt[i] = avg

    lt_mat = {'all_left': avg_lt, 'all_right': avg_rt}
    scipy.io.savemat('/home/ming/Desktop/lt_avg_all.mat', lt_mat)