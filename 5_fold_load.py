import torch
from torch.utils import data
from dataset import Dataset_starmen
from data_preprocess import Data_preprocess
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.cross_decomposition import PLSRegression
from torch.autograd import Variable
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

    train_recon, test_recon = [], []
    orthogonality, pls = [], []
    pearsonr, spearmanr = [], []
    for fold in range(5):
        logger.info(f"##### Fold {fold + 1}/5 #####\n")

        # load the model
        model_name = 'ML_VAE'
        autoencoder = torch.load('5-fold/{}/{}_fold_{}'.format(model_name, fold, model_name), map_location=device)
        autoencoder.device = device
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

        # self-recon loss on train dataset
        losses = 0
        batches = 0
        ZU, ZV = None, None
        with torch.no_grad():
            autoencoder.Training = False
            for data in train_loader:
                image = torch.tensor([[np.load(path)] for path in data[0]], device=autoencoder.device).float()
                input_ = Variable(image).to(autoencoder.device)
                zs_mu, zs_logVar, zpsi_mu, zpsi_logVar, zs_encoded, zpsi_encoded = autoencoder.forward(input_)
                encoded = torch.cat((zs_encoded, zpsi_encoded), dim=1)
                reconstructed = autoencoder.decoder(encoded)
                reconstruction_loss, zs_kl_loss = autoencoder.loss(zs_mu, zs_logVar, input_, reconstructed)

                loss = reconstruction_loss

                # store ZU, ZV
                # if ZU is None:
                #     ZU, ZV = zpsi_encoded, zs_encoded
                # else:
                #     ZU = torch.cat((ZU, zpsi_encoded), 0)
                #     ZV = torch.cat((ZV, zs_encoded), 0)

                losses += float(loss)
                batches += 1

        train_recon.append(losses / batches / 64 / 64)

        # self-recon loss on test dataset
        losses = 0
        batches = 0
        with torch.no_grad():
            autoencoder.Training = False
            for data in test_loader:
                image = torch.tensor([[np.load(path)] for path in data[0]], device=autoencoder.device).float()
                input_ = Variable(image).to(autoencoder.device)
                zs_mu, zs_logVar, zpsi_mu, zpsi_logVar, zs_encoded, zpsi_encoded = autoencoder.forward(input_)
                encoded = torch.cat((zs_encoded, zpsi_encoded), dim=1)
                reconstructed = autoencoder.decoder(encoded)
                reconstruction_loss, zs_kl_loss = autoencoder.loss(zs_mu, zs_logVar, input_, reconstructed)

                loss = reconstruction_loss

                # store ZU, ZV
                if ZU is None:
                    ZU, ZV = zpsi_encoded, zs_encoded
                else:
                    ZU = torch.cat((ZU, zpsi_encoded), 0)
                    ZV = torch.cat((ZV, zs_encoded), 0)

                losses += float(loss)
                batches += 1

        test_recon.append(losses / batches / 64 / 64)

        # plot latent space
        min_, mean_, max_ = autoencoder.plot_z_distribution(ZU, ZV)
        autoencoder.plot_simu_repre(min_, mean_, max_)
        autoencoder.plot_grad_simu_repre(min_, mean_, max_)

        # calculate orthogonality between ZU and ZV
        if ZU.size()[1:] == ZV.size()[1:]:
            ortho = torch.matmul(ZU, torch.transpose(ZV, 0, 1))
            ortho = torch.det(ortho)
            orthogonality.append(float(ortho))

        # calculate pls correlation between ZU and ZV
        ZU_pls, ZV_pls = ZU.cpu().detach().numpy(), ZV.cpu().detach().numpy()
        pls_model = PLSRegression(n_components=1)
        pls_model.fit(ZV_pls, ZU_pls)
        pred = pls_model.predict(ZV_pls)
        pls_corr = np.corrcoef(pred.reshape(-1), ZU_pls.reshape(-1))[0, 1]
        pls.append(pls_corr)

        # PCA for ZU
        if ZU.size()[-1] > 1:
            _, _, V = torch.pca_lowrank(ZU)
            PCA_ZU = -1 * torch.matmul(ZU, V[:, 0]).cpu().detach().numpy()
        else:
            PCA_ZU = ZU.cpu().detach().numpy().squeeze()
        # get psi
        psi = test_data['alpha'] * (test_data['age'] - test_data['baseline_age'])
        # calculate pearson and spearman correlation
        if ZU.size()[-1] == ZV.size()[-1]:
            _, _, V = torch.pca_lowrank(ZV)
            PCA_ZV = -1 * torch.matmul(ZV, V[:, 0]).cpu().detach().numpy()
            pearsonr.append(max(np.abs(stats.pearsonr(PCA_ZU, psi)[0]), np.abs(stats.pearsonr(PCA_ZV, psi)[0])))
            spearmanr.append(max(np.abs(stats.spearmanr(PCA_ZU, psi)[0]), np.abs(stats.spearmanr(PCA_ZV, psi)[0])))
        else:
            pearsonr.append(np.abs(stats.pearsonr(PCA_ZU, psi)[0]))
            spearmanr.append(np.abs(stats.spearmanr(PCA_ZU, psi)[0]))

    print('recon train: ', np.mean(train_recon), np.std(train_recon))
    print('recon test: ', np.mean(test_recon), np.std(test_recon))
    print('pearsonr: ', np.mean(pearsonr), np.std(pearsonr))
    print('spearmanr: ', np.mean(spearmanr), np.std(spearmanr))
    print('pls: ', np.mean(pls), np.std(pls))
    print('orthogonality: ', np.mean(orthogonality), np.std(orthogonality))
    print(pearsonr, spearmanr)
