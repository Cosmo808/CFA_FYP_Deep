import torch
from torch.utils import data
from dataset import Dataset_starmen
from data_preprocess import Data_preprocess_starmen
import pandas as pd
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

# recon class
ae_disen_recon_class = ['starmen']
vae_disen_recon_class = ['ML_VAE', 'rank_VAE']
vae_no_disen_recon_class = ['beta_VAE', 'Riem_VAE']
model_class = ['LNE']

# pred class
vae_pred_class = ['ML_VAE', 'rank_VAE', 'beta_VAE']
align_pred_class = ['starmen', 'Riem_VAE']


def expand_vector(vec, missing_num, num_subject):
    t_num = 10 - missing_num
    error_num = missing_num - t_num
    for i in range(num_subject):
        vec = torch.cat((vec[:t_num * (i + 1) + error_num * i],
                         torch.mean(vec[:t_num * (i + 1) + error_num * i], 0).repeat(error_num, 1),
                         vec[t_num * (i + 1) + error_num * i:]
                         ), dim=0)
    return vec


def get_reconstruction(input_, model_name):
    if model_name in ae_disen_recon_class:
        reconstructed, z, zu, zv = autoencoder.forward(input_)
        reconstruction_loss = autoencoder.loss(input_, reconstructed)

    if model_name in vae_disen_recon_class:
        zs_mu, zs_logVar, zpsi_mu, zpsi_logVar, zs_encoded, zpsi_encoded = autoencoder.forward(input_)
        zu, zv, z = zpsi_encoded, zs_encoded, None
        encoded = torch.cat((zs_encoded, zpsi_encoded), dim=1)
        reconstructed = autoencoder.decoder(encoded)
        reconstruction_loss, zs_kl_loss = autoencoder.loss(zs_mu, zs_logVar, input_, reconstructed)

    if model_name in model_class:
        zu, zv = None, None
        z, reconstructed = autoencoder.forward(input_)
        reconstruction_loss = autoencoder.compute_recon_loss(input_, reconstructed) * 64 * 64

    if model_name in vae_no_disen_recon_class:
        zu, zv = None, None
        z, logVar, reconstructed = autoencoder.forward(input_)
        reconstruction_loss, _ = autoencoder.loss(z, logVar, reconstructed, input_)
        if model_name == 'Riem_VAE':
            zu = torch.transpose(z[:, 0].repeat(3, 1), 0, 1)
            zv = z[:, 1:]

    return reconstruction_loss, zu, zv, z


def get_pred_loss(image, model_name, missing_num=7):
    autoencoder.eval()
    num_subject = image.size()[0] // 10
    idx0, idx1 = [], []
    for i in range(num_subject):
        idx0 += list(np.arange(i * 10, i * 10 + (10 - missing_num)))
        idx1 += list(np.arange(i * 10 + (10 - missing_num), i * 10 + 10))
    image0, image1 = image[idx0], image[idx1]
    input_0 = Variable(image0).to(autoencoder.device)
    input_1 = Variable(image1).to(autoencoder.device)

    if model_name in align_pred_class:
        # arange data
        age = pd.DataFrame(data[3].cpu().detach(), columns=['age'])
        baseline_age = pd.DataFrame(data[2].cpu().detach(), columns=['baseline_age'])
        data_xy = pd.concat([age, baseline_age], axis=1)
        # get X and Y
        X, Y = data_generator.generate_XY(data_xy)
        X, Y = Variable(X).to(autoencoder.device).float(), Variable(Y).to(autoencoder.device).float()
        X0, X1 = X[idx0], X[idx1]
        Y0, Y1 = Y[idx0], Y[idx1]
        if model_name == 'starmen':
            # get z, zu, zv
            z0 = autoencoder.encoder(input_0)
            zu0, zv0 = torch.matmul(z0, autoencoder.U), torch.matmul(z0, autoencoder.V)
            # get b
            yt = torch.transpose(Y0, 0, 1)
            yty = torch.matmul(yt, Y0)
            yt_zv = torch.matmul(yt, zv0)
            xbeta = torch.matmul(X0, autoencoder.beta)
            yt_z_xbeta = torch.matmul(yt, z0 - xbeta)
            b = torch.matmul(
                torch.inverse((autoencoder.sigma0_2 + autoencoder.sigma2_2) * yty - 2 * autoencoder.sigma0_2
                              * autoencoder.sigma2_2 * torch.eye(yty.size()[0], device=autoencoder.device)),
                autoencoder.sigma2_2 * yt_z_xbeta + autoencoder.sigma0_2 * yt_zv
            )
            # get z1
            z1 = torch.matmul(X1, autoencoder.beta) + torch.matmul(Y1, b)

        if model_name == 'Riem_VAE':
            # get z
            z0, _ = autoencoder.encoder(input_0)
            # get alpha
            alpha = torch.tensor([[a.exp() for a in data[6]]], device=autoencoder.device).float().view(len(idx0) + len(idx1), -1)
            alpha0, alpha1 = alpha[idx0], alpha[idx1]
            # get delta age
            delta_age0 = X0[:, 1].clone().detach().to(autoencoder.device).float().view(alpha0.size())
            delta_age1 = X1[:, 1].clone().detach().to(autoencoder.device).float().view(alpha1.size())
            # calculate fixed
            fixed0 = torch.mul(delta_age0, alpha0) / 50
            fixed1 = torch.mul(delta_age1, alpha1) / 50
            fixed0 = torch.cat((fixed0, torch.zeros([len(idx0), z0.size()[1] - 1]).to(autoencoder.device).float()), dim=1)
            fixed1 = torch.cat((fixed1, torch.zeros([len(idx1), z0.size()[1] - 1]).to(autoencoder.device).float()), dim=1)
            # calculate random
            Y0 = (Y0[:, ::2]).to(autoencoder.device).float()
            Y1 = (Y1[:, ::2]).to(autoencoder.device).float()
            omega = torch.matmul(
                torch.inverse(torch.matmul(torch.transpose(Y0, 0, 1), Y0)),
                torch.matmul(torch.transpose(Y0, 0, 1), z0 - fixed0)
            )
            random = torch.matmul(Y1, omega)
            random[:, 0] = 0.
            # get z1
            z1 = fixed1 + random

        predicted = autoencoder.decoder(z1)
        return torch.sum((predicted - input_1) ** 2) / input_1.shape[0]

    if model_name in vae_pred_class:
        if model_name in vae_disen_recon_class:
            zs_mu, zs_logVar, zpsi_mu, zpsi_logVar, zs_encoded, zpsi_encoded = autoencoder.forward(input_0)
            if missing_num > 5:
                zs_mu, zs_logVar = expand_vector(zs_mu, missing_num, num_subject), expand_vector(zs_logVar, missing_num, num_subject)
                zpsi_mu, zpsi_logVar = expand_vector(zpsi_mu, missing_num, num_subject), expand_vector(zpsi_logVar, missing_num, num_subject)
            zs_encoded = autoencoder.reparametrize(zs_mu, zs_logVar)
            zpsi_encoded = autoencoder.reparametrize(zpsi_mu, zpsi_logVar)
            encoded = torch.cat((zs_encoded, zpsi_encoded), dim=1)

        if model_name in vae_no_disen_recon_class:
            z, logVar, reconstructed = autoencoder.forward(input_0)
            if missing_num > 5:
                z, logVar = expand_vector(z, missing_num, num_subject), expand_vector(logVar, missing_num, num_subject)
            encoded = autoencoder.reparametrize(z, logVar)

        reconstructed = autoencoder.decoder(encoded)
        pred_loss, _ = autoencoder.loss(torch.tensor([0.], device=autoencoder.device),
                                        torch.tensor([0.], device=autoencoder.device), reconstructed, input_1)
        return pred_loss

    return 0.0


if __name__ == '__main__':
    logger.info(f"Device is {device}")
    data_generator = Data_preprocess_starmen()

    train_recon, test_recon = [], []
    orthogonality, pls_R2 = [], []
    pearsonr, spearmanr = [], []
    pred_loss = []
    for fold in range(0, 5):
        logger.info(f"##### Fold {fold + 1}/5 #####\n")

        # load the model
        model_name = 'starmen'
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
                        train_data['timepoint'], train_data['first_age'], train_data['alpha'])
        test = Dataset(test_data['path'], test_data['subject'], test_data['baseline_age'], test_data['age'],
                       test_data['timepoint'], test_data['first_age'], test_data['alpha'])

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

                reconstruction_loss, zu, zv, z = get_reconstruction(input_, model_name)
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
        pred_losses = 0
        with torch.no_grad():
            for data in test_loader:
                batches += 1
                image = torch.tensor([[np.load(path)] for path in data[0]], device=autoencoder.device).float()
                input_ = Variable(image).to(autoencoder.device)

                reconstruction_loss, zu, zv, z = get_reconstruction(input_, model_name)
                loss = reconstruction_loss
                losses += float(loss)

                # store ZU, ZV
                if zv is not None:
                    if ZU is None:
                        ZU, ZV = zu, zv
                    else:
                        ZU = torch.cat((ZU, zu), 0)
                        ZV = torch.cat((ZV, zv), 0)
                else:
                    if ZU is None:
                        ZU = z
                    else:
                        ZU = torch.cat((ZU, z), 0)

                # predict future data
                prediction_loss = get_pred_loss(image, model_name)
                pred_losses += float(prediction_loss)

        test_recon.append(losses / batches / 64 / 64)
        pred_loss.append(pred_losses / batches / 64 / 64)

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
            if model_name == 'Riem_VAE':
                PCA_ZU = ZU[:, 0].cpu().detach().numpy().squeeze()

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
    print('future prediction loss: ', np.mean(pred_loss), np.std(pred_loss))

    print(train_recon)
    print(test_recon)
    print(pred_loss)
    print(spearmanr)