import torch
from torch.utils import data
import torch.nn.functional as F
from dataset import Dataset_starmen
from data_preprocess import Data_preprocess
from torch.autograd import Variable
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
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
    data_generator = Data_preprocess()

    # train_recon_ours, test_recon_ours = [], []
    # train_recon_bvae, test_recon_bvae = [], []
    # for fold in range(5):
    #     logger.info(f"##### Fold {fold + 1}/5 #####\n")
    #
    #     # load two models
    #     autoencoder = torch.load('5-fold/ours/{}_fold_starmen'.format(fold), map_location=device)
    #     beta_VAE = torch.load('5-fold/beta_VAE/{}_fold_beta_VAE'.format(fold), map_location=device)
    #     autoencoder.eval()
    #     beta_VAE.eval()
    #
    #     # load train and test data
    #     train_data, test_data = data_generator.generate_train_test(fold)
    #     train_data.requires_grad = False
    #     test_data.requires_grad = False
    #
    #     Dataset = Dataset_starmen
    #     train = Dataset(train_data['path'], train_data['subject'], train_data['baseline_age'], train_data['age'],
    #                     train_data['timepoint'], train_data['first_age'])
    #     test = Dataset(test_data['path'], test_data['subject'], test_data['baseline_age'], test_data['age'],
    #                    test_data['timepoint'], test_data['first_age'])
    #
    #     train_loader = torch.utils.data.DataLoader(train, batch_size=256, shuffle=False,
    #                                                num_workers=0, drop_last=False, pin_memory=True)
    #     test_loader = torch.utils.data.DataLoader(test, batch_size=256, shuffle=False,
    #                                               num_workers=0, drop_last=False, pin_memory=True)
    #
    #     # self-recon loss on train dataset
    #     loss_ours, loss_bvae = 0, 0
    #     batch = 0
    #     with torch.no_grad():
    #         for data in train_loader:
    #             batch += 1
    #
    #             image = torch.tensor([[np.load(path)] for path in data[0]], device=device).float()
    #             input_ = Variable(image).to(device)
    #
    #             # ours
    #             reconstructed, _, _, _ = autoencoder.forward(input_)
    #             self_reconstruction_loss = autoencoder.loss(input_, reconstructed)
    #
    #             # beta VAE
    #             mu, logVar, reconstructed = beta_VAE.forward(input_)
    #             reconstruction_loss, a = beta_VAE.loss(mu, logVar, input_, reconstructed)
    #
    #             loss_ours += float(self_reconstruction_loss)
    #             loss_bvae += float(reconstruction_loss)
    #
    #     train_recon_ours.append(loss_ours / batch / 64 / 64)
    #     train_recon_bvae.append(loss_bvae / batch / 64 / 64)
    #
    #     # self-recon loss on test dataset
    #     loss_ours, loss_bvae = 0, 0
    #     batch = 0
    #     with torch.no_grad():
    #         for data in test_loader:
    #             batch += 1
    #
    #             image = torch.tensor([[np.load(path)] for path in data[0]], device=device).float()
    #             input_ = Variable(image).to(device)
    #
    #             # ours
    #             reconstructed, _, _, _ = autoencoder.forward(input_)
    #             self_reconstruction_loss = autoencoder.loss(input_, reconstructed)
    #
    #             # beta VAE
    #             mu, logVar, reconstructed = beta_VAE.forward(input_)
    #             reconstruction_loss, _ = beta_VAE.loss(mu, logVar, input_, reconstructed)
    #
    #             loss_ours += float(self_reconstruction_loss)
    #             loss_bvae += float(reconstruction_loss)
    #
    #     test_recon_ours.append(loss_ours / batch / 64 / 64)
    #     test_recon_bvae.append(loss_bvae / batch / 64 / 64)
    #
    # print('train ours: ', np.mean(train_recon_ours), np.std(train_recon_ours))
    # print('test ours: ', np.mean(test_recon_ours), np.std(test_recon_ours))
    #
    # print('train bvae: ', np.mean(train_recon_bvae), np.std(train_recon_bvae))
    # print('test bvae: ', np.mean(test_recon_bvae), np.std(test_recon_bvae))

    orthogonality = []
    pearsonr = []
    spearmanr = []
    for fold in range(5):
        logger.info(f"##### Fold {fold + 1}/5 #####\n")

        # load two models
        autoencoder = torch.load('5-fold/ours/{}_fold_starmen'.format(fold), map_location=device)
        # autoencoder = torch.load('model/best_starmen', map_location=device)
        autoencoder.eval()

        # load train and test data
        train_data, test_data = data_generator.generate_train_test(fold)
        train_data.requires_grad = False
        test_data.requires_grad = False

        Dataset = Dataset_starmen
        train = Dataset(train_data['path'], train_data['subject'], train_data['baseline_age'], train_data['age'],
                        train_data['timepoint'], train_data['first_age'])
        test = Dataset(test_data['path'], test_data['subject'], test_data['baseline_age'], test_data['age'],
                       test_data['timepoint'], test_data['first_age'])

        train_loader = torch.utils.data.DataLoader(train, batch_size=256, shuffle=False,
                                                   num_workers=0, drop_last=False, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test, batch_size=256, shuffle=False,
                                                  num_workers=0, drop_last=False, pin_memory=True)

        with torch.no_grad():
            Z, ZU, ZV = None, None, None
            # for data in train_loader:
            #     image = torch.tensor([[np.load(path)] for path in data[0]], device=device).float()
            #
            #     # self-reconstruction loss
            #     input_ = Variable(image).to(device)
            #     reconstructed, z, zu, zv = autoencoder.forward(input_)
            #     self_reconstruction_loss = autoencoder.loss(input_, reconstructed)
            #
            #     # store Z, ZU, ZV
            #     if Z is None:
            #         Z, ZU, ZV = z, zu, zv
            #     else:
            #         Z = torch.cat((Z, z), 0)
            #         ZU = torch.cat((ZU, zu), 0)
            #         ZV = torch.cat((ZV, zv), 0)

            for data in test_loader:
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

        # min_, mean_, max_ = autoencoder.plot_z_distribution(Z, ZU, ZV)
        # autoencoder.plot_simu_repre(min_, mean_, max_)
        # autoencoder.plot_grad_simu_repre(min_, mean_, max_)
        # exit()

        ortho = torch.matmul(ZU, torch.transpose(ZV, 0, 1))
        ortho = torch.det(ortho)
        orthogonality.append(float(ortho))

        U, S, V = torch.pca_lowrank(ZU)
        PCA_ZU = -1 * torch.matmul(ZU, V[:, 0]).cpu().detach().numpy()
        # get psi
        psi = test_data['alpha'] * (test_data['age'] - test_data['baseline_age'])
        pearsonr.append(stats.pearsonr(PCA_ZU, psi)[0])
        spearmanr.append(stats.spearmanr(PCA_ZU, psi)[0])

    print('orthogonality: ', np.mean(orthogonality), np.std(orthogonality))
    print('pearsonr: ', np.mean(pearsonr), np.std(pearsonr))
    print('spearmanr: ', np.mean(spearmanr), np.std(spearmanr))
