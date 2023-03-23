import torch
from torch.utils import data
from dataset import Dataset_starmen
from data_preprocess import Data_preprocess
from torch.autograd import Variable
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
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
    # load model
    autoencoder = torch.load('model/best_starmen', map_location=device)
    autoencoder.eval()

    # load data
    data_generator = Data_preprocess()
    dataset = data_generator.generate_all()
    dataset.requires_grad = False

    Dataset = Dataset_starmen
    all_data = Dataset(dataset['path'], dataset['subject'], dataset['baseline_age'], dataset['age'],
                       dataset['timepoint'], dataset['first_age'])

    data_loader = torch.utils.data.DataLoader(all_data, batch_size=256, shuffle=False,
                                              num_workers=0, drop_last=False, pin_memory=True)

    # get Z, ZU, ZV
    with torch.no_grad():
        Z, ZU, ZV = None, None, None
        for data in data_loader:
            image = torch.tensor([[np.load(path)] for path in data[0]], device=device).float()

            # self-reconstruction loss
            input_ = Variable(image).to(device)
            reconstructed, z, zu, zv = autoencoder.forward(input_)
            self_reconstruction_loss = autoencoder.loss(input_, reconstructed)

            # store Z, ZU, ZV
            if Z is None:
                Z, ZU, ZV = z, zu, zv
            else:
                Z = torch.cat((Z, z), 0)
                ZU = torch.cat((ZU, zu), 0)
                ZV = torch.cat((ZV, zv), 0)

    min_, mean_, max_ = autoencoder.plot_z_distribution(Z, ZU, ZV)
    autoencoder.plot_simu_repre(min_, mean_, max_)
    autoencoder.plot_grad_simu_repre(min_, mean_, max_)