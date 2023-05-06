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
    with torch.no_grad():
        Z, ZU, ZV = None, None, None
        for data in train_loader:
            image = data[autoencoder.left_right]

            # self-reconstruction loss
            input_ = Variable(image).to(device).float()
            reconstructed, z, zu, zv = autoencoder.forward(input_)
            self_reconstruction_loss = autoencoder.loss(input_, reconstructed)

            # store Z, ZU, ZV
            if Z is None:
                Z, ZU, ZV = z, zu, zv
            else:
                Z = torch.cat((Z, z), 0)
                ZU = torch.cat((ZU, zu), 0)
                ZV = torch.cat((ZV, zv), 0)

    train_ZU, train_ZV = ZU, ZV

    with torch.no_grad():
        Z, ZU, ZV = None, None, None
        for data in test_loader:
            image = data[autoencoder.left_right]

            # self-reconstruction loss
            input_ = Variable(image).to(device).float()
            reconstructed, z, zu, zv = autoencoder.forward(input_)
            self_reconstruction_loss = autoencoder.loss(input_, reconstructed)

            # store Z, ZU, ZV
            if Z is None:
                Z, ZU, ZV = z, zu, zv
            else:
                Z = torch.cat((Z, z), 0)
                ZU = torch.cat((ZU, zu), 0)
                ZV = torch.cat((ZV, zv), 0)

    test_ZU, test_ZV = ZU, ZV

    classifier = model_adni.Classifier(target_num=3)
    optimizer_fn = optim.Adam
    optimizer = optimizer_fn(autoencoder.parameters(), lr=1e-3)
    classifier.train_(train_ZV, demo_train['label'], test_ZV, demo_test['label'], optimizer=optimizer, num_epochs=200)
