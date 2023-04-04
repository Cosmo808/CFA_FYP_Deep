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
    model_class_0 = ['starmen']
    model_class_1 = ['ML_VAE', 'rank_VAE']
    model_class_2 = ['LNE']
    model_class_3 = ['Riem_VAE']

    train_recon, test_recon = [], []
    orthogonality, pls_R2 = [], []
    pearsonr, spearmanr = [], []
    for fold in range(0, 5):
        logger.info(f"##### Fold {fold + 1}/5 #####\n")

        # load the model
        model_name = 'LNE'
        autoencoder = torch.load('5-fold/{}/{}_fold_{}'.format(model_name, fold, model_name), map_location=device)
        autoencoder.device = device
        autoencoder.Training = False
        autoencoder.eval()

        # load train and test data
        train_data, test_data = data_generator.generate_train_test(fold)
        all_data = data_generator.generate_all()
        train_data.requires_grad = False
        test_data.requires_grad = False

        Dataset = Dataset_starmen
        train = Dataset(train_data['path'], train_data['subject'], train_data['baseline_age'], train_data['age'],
                        train_data['timepoint'], train_data['first_age'])
        test = Dataset(test_data['path'], test_data['subject'], test_data['baseline_age'], test_data['age'],
                       test_data['timepoint'], test_data['first_age'])

        train_loader = torch.utils.data.DataLoader(train, batch_size=256, shuffle=False,
                                                   num_workers=0, drop_last=False, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test, batch_size=200, shuffle=False,
                                                  num_workers=0, drop_last=False, pin_memory=True)

        # self-recon loss on train dataset
        losses = 0
        batches = 0
        ZU, ZV = None, None
        with torch.no_grad():
            for data in train_loader:
                image = torch.tensor([[np.load(path)] for path in data[0]], device=autoencoder.device).float()
                input_ = Variable(image).to(autoencoder.device)

                if model_name in model_class_0:
                    reconstructed, z, zu, zv = autoencoder.forward(input_)
                    reconstruction_loss = autoencoder.loss(input_, reconstructed)

                if model_name in model_class_1:
                    zs_mu, zs_logVar, zpsi_mu, zpsi_logVar, zs_encoded, zpsi_encoded = autoencoder.forward(input_)
                    zu, zv = zpsi_encoded, zs_encoded
                    encoded = torch.cat((zs_encoded, zpsi_encoded), dim=1)
                    reconstructed = autoencoder.decoder(encoded)
                    reconstruction_loss, zs_kl_loss = autoencoder.loss(zs_mu, zs_logVar, input_, reconstructed)

                if model_name in model_class_2:
                    z, reconstructed = autoencoder.forward(input_)
                    reconstruction_loss = autoencoder.compute_recon_loss(input_, reconstructed) * 64 * 64

                loss = reconstruction_loss

                # store ZU, ZV
                # if ZU is None:
                #     ZU, ZV = zu, zv
                # else:
                #     ZU = torch.cat((ZU, zu), 0)
                #     ZV = torch.cat((ZV, zv), 0)

                losses += float(loss)
                batches += 1

        train_recon.append(losses / batches / 64 / 64)

        # self-recon loss on test dataset
        losses = 0
        batches = 0
        with torch.no_grad():
            for data in test_loader:
                image = torch.tensor([[np.load(path)] for path in data[0]], device=autoencoder.device).float()
                input_ = Variable(image).to(autoencoder.device)

                if model_name in model_class_0:
                    reconstructed, z, zu, zv = autoencoder.forward(input_)
                    reconstruction_loss = autoencoder.loss(input_, reconstructed)

                if model_name in model_class_1:
                    zs_mu, zs_logVar, zpsi_mu, zpsi_logVar, zs_encoded, zpsi_encoded = autoencoder.forward(input_)
                    zu, zv = zpsi_encoded, zs_encoded
                    encoded = torch.cat((zs_encoded, zpsi_encoded), dim=1)
                    reconstructed = autoencoder.decoder(encoded)
                    reconstruction_loss, zs_kl_loss = autoencoder.loss(zs_mu, zs_logVar, input_, reconstructed)

                if model_name in model_class_2:
                    z, reconstructed = autoencoder.forward(input_)
                    reconstruction_loss = autoencoder.compute_recon_loss(input_, reconstructed) * 64 * 64

                loss = reconstruction_loss

                # store ZU, ZV
                if model_name in model_class_0 + model_class_1:
                    if ZU is None:
                        ZU, ZV = zu, zv
                    else:
                        ZU = torch.cat((ZU, zu), 0)
                        ZV = torch.cat((ZV, zv), 0)
                if model_name in model_class_2:
                    if ZU is None:
                        ZU = z
                    else:
                        ZU = torch.cat((ZU, z), 0)

                # predict future data
                if model_name in model_class_0 + model_class_3:
                    X, Y = data_generator.generate_XY(test_data)
                    num_subject = image.size()[0] // 10

                    idx0, idx1 = [], []
                    for i in range(num_subject):
                        idx0 += list(np.arange(i * 10, i * 10 + 5))
                        idx1 += list(np.arange(i * 10 + 5, i * 10 + 10))
                    image0, image1 = image[idx0], image[idx1]
                    X0, X1 = X[idx0], X[idx1]
                    z0 = autoencoder.encoder(image0)
                    z1 = autoencoder.encoder(image1)

                losses += float(loss)
                batches += 1

        test_recon.append(losses / batches / 64 / 64)

        # plot latent space
        # min_, mean_, max_ = autoencoder.plot_z_distribution(ZU, ZV)
        # autoencoder.plot_simu_repre(min_, mean_, max_)
        # autoencoder.plot_grad_simu_repre(min_, mean_, max_)

        if ZV is not None:
            # calculate orthogonality between ZU and ZV
            if ZU.size()[1:] == ZV.size()[1:]:
                ortho = torch.matmul(ZU, torch.transpose(ZV, 0, 1))
                ortho = torch.norm(ortho, p='fro') ** 2 / (ortho.size()[0]) ** 2
                orthogonality.append(float(ortho))

            # calculate pls correlation between ZU and ZV
            ZU_pls, ZV_pls = ZU.cpu().detach().numpy(), ZV.cpu().detach().numpy()
            pls_model = PLSRegression(n_components=2)
            pls_model.fit(ZV_pls, ZU_pls)
            R2 = pls_model.score(ZV_pls, ZU_pls)
            pls_R2.append(R2)

        if ZU is not None:
            # PCA for ZU
            if ZU.size()[-1] > 1:
                _, _, V = torch.pca_lowrank(ZU)
                PCA_ZU = -1 * torch.matmul(ZU, V[:, 0]).cpu().detach().numpy()
            else:
                PCA_ZU = ZU.cpu().detach().numpy().squeeze()
            # get psi
            if ZU.size()[0] == 10000:
                psi = all_data['alpha'] * (all_data['age'] - all_data['baseline_age'])
            else:
                psi = test_data['alpha'] * (test_data['age'] - test_data['baseline_age'])
            # calculate pearson and spearman correlation
            if ZV is not None and ZU.size()[-1] == ZV.size()[-1]:
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
    print('pls_R2: ', np.mean(pls_R2), np.std(pls_R2))
    print('orthogonality: ', np.mean(orthogonality), np.std(orthogonality))
