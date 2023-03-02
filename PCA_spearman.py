import torch
from torch.utils import data
import torch.nn.functional as F
from dataset import Dataset_starmen
from data_preprocess import Data_preprocess
from torch.autograd import Variable
import statsmodels.api as sm
import matplotlib
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

    # U, S, V = torch.pca_lowrank(ZU)
    # PCA_ZU = -1 * torch.matmul(ZU, V[:, 0]).cpu().detach().numpy()

    # get psi
    psi = dataset['alpha'] * (dataset['age'] - dataset['baseline_age'])

    # plot same raising stage
    index = []
    for idx, p in enumerate(psi[:200]):
        match = [i for i, p_ in enumerate(psi[:200]) if 1e-5 < np.abs(p_ - p) <= 0.05]
        if len(match) >= 3:
            index.append([idx] + match)

    idx = index[1]
    print(idx)
    fig, axes = plt.subplots(3, len(idx), figsize=(2 * len(idx), 6))
    plt.subplots_adjust(wspace=0, hspace=0)
    image = [np.load(path) for path in dataset.iloc[idx, 0]]
    recon_glo = autoencoder.decoder(ZU[idx])
    recon_indiv = autoencoder.decoder(ZV[idx])
    for i in range(len(idx)):
        axes[0][i].matshow(255 * image[i])
        axes[1][i].matshow(255 * recon_glo[i][0].cpu().detach().numpy())
        axes[2][i].matshow(255 * recon_indiv[i][0].cpu().detach().numpy())
    for axe in axes:
        for ax in axe:
            ax.set_xticks([])
            ax.set_yticks([])

    # calculate mse between global trajectory
    # sim = []
    # for idx in index:
    #     image = [np.load(path) for path in dataset.iloc[idx, 0]]
    #     recon_glo = autoencoder.decoder(ZU[idx])
    #     recon_indiv = autoencoder.decoder(ZV[idx])
    #     mean_ = torch.mean(recon_glo, dim=0, keepdim=True)
    #     simi = autoencoder.loss(recon_glo, mean_[0])
    #     sim.append(float(simi) / 64 / 64)
    # print(np.mean(sim), np.std(sim))

    # Y = psi
    # X = sm.add_constant(PCA_ZU)
    # linear_model = sm.OLS(Y, X)
    # results = linear_model.fit()
    # print(results.summary())
    # print(stats.pearsonr(PCA_ZU, psi))
    # print(stats.spearmanr(PCA_ZU, psi))
    # for x, y in zip(PCA_ZU, psi):
    #     if np.random.rand() < 0.1:
    #         plt.scatter(x, y)

    plt.show()
