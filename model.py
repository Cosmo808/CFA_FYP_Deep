import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils import data
from time import time
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
import logging
import math
import sys
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
format = logging.Formatter("%(message)s")

ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(format)
logger.addHandler(ch)

dim_z = 4
N, I, p, q = 8000, 800, 3, 2


class AE_starmen(nn.Module):
    def __init__(self):
        super(AE_starmen, self).__init__()
        nn.Module.__init__(self)
        self.name = 'AE_starmen'
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.gamma = 1
        self.lam = 1.0

        self.conv1 = nn.Conv2d(1, 16, 3, stride=2, padding=1)  # 16 x 32 x 32
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)  # 32 x 16 x 16
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)  # 32 x 8 x 8
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.fc10 = nn.Linear(2048, dim_z)
        self.fc11 = nn.Linear(2048, dim_z)

        self.fc3 = nn.Linear(dim_z, 512)
        self.upconv1 = nn.ConvTranspose2d(8, 64, 3, stride=2, padding=1, output_padding=1)  # 64 x 16 x 16
        self.upconv2 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)  # 32 x 32 x 32
        self.upconv3 = nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1)  # 1 x 64 x 64
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(32)

        self.train_loss, self.test_loss = [], []
        self.self_recon, self.cross_recon, self.align = [], [], []
        self.log_p_b = []

        self.X, self.Y = None, None

        self.beta = torch.rand(size=[p, dim_z], device=self.device)
        self.b = torch.normal(mean=0, std=1, size=[q * I, dim_z], device=self.device)
        self.U = torch.diag(torch.tensor([1 for i in range(dim_z // 2)] + [0 for i in range(dim_z - dim_z // 2)],
                                         device=self.device)).float()
        self.V = torch.eye(dim_z, device=self.device) - self.U
        self.sigma0_2, self.sigma1_2, self.sigma2_2 = 1, 0.5, 1
        self.D = torch.eye(q * I, device=self.device).float()

    def encoder(self, image):
        h1 = F.relu(self.bn1(self.conv1(image)))
        h2 = F.relu(self.bn2(self.conv2(h1)))
        h3 = F.relu(self.bn3(self.conv3(h2)))
        z = torch.tanh(self.fc10(h3.view(h3.size()[0], -1)))
        return z

    def decoder(self, encoded):
        h6 = F.relu(self.fc3(encoded)).reshape([encoded.size()[0], 8, 8, 8])
        h7 = F.relu(self.bn4(self.upconv1(h6)))
        h8 = F.relu(self.bn5(self.upconv2(h7)))
        reconstructed = F.relu(self.upconv3(h8))
        return reconstructed

    def forward(self, image0, image1=None):
        z0 = self.encoder(image0)
        zu0 = torch.matmul(z0, self.U)
        zv0 = torch.matmul(z0, self.V)
        if image1 is not None:
            z1 = self.encoder(image1)
            encoded = torch.matmul(z1, self.U) + zv0
            reconstructed = self.decoder(encoded)
            return reconstructed
        else:
            encoded = zu0 + zv0
            reconstructed = self.decoder(encoded)
            return reconstructed, z0, zu0, zv0

    @staticmethod
    def loss(input_, reconstructed):
        recon_loss = torch.sum((reconstructed - input_) ** 2) / input_.shape[0]
        return recon_loss

    def train_(self, data_loader, test, optimizer, num_epochs):

        self.to(self.device)
        best_loss = 1e10
        es = 0

        for epoch in range(num_epochs):

            start_time = time()
            if es == 100:
                break

            logger.info('#### Epoch {}/{} ####'.format(epoch + 1, num_epochs))

            tloss = 0.0
            nb_batches = 0

            s_tp, Z, ZU, ZV = None, None, None, None
            for data in tqdm(data_loader):
                image = torch.tensor([[np.load(path)] for path in data[0]], device=self.device).float()
                optimizer.zero_grad()

                # self-reconstruction loss
                input_ = Variable(image).to(self.device)
                reconstructed, z, zu, zv = self.forward(input_)
                self_reconstruction_loss = self.loss(input_, reconstructed)

                # store Z, ZU, ZV
                subject = torch.tensor([[s for s in data[1]]], device=self.device)
                tp = torch.tensor([[tp for tp in data[4]]], device=self.device)
                st = torch.transpose(torch.cat((subject, tp), 0), 0, 1)
                if s_tp is None:
                    s_tp, Z, ZU, ZV = st, z, zu, zv
                else:
                    s_tp = torch.cat((s_tp, st), 0)
                    Z = torch.cat((Z, z), 0)
                    ZU = torch.cat((ZU, zu), 0)
                    ZV = torch.cat((ZV, zv), 0)

                # cross-reconstruction loss
                baseline_age = data[2]
                delta_age = data[3] - baseline_age
                index0, index1 = self.generate_sample(baseline_age, delta_age)
                image0 = image[index0]
                image1 = image[index1]
                if index0:
                    input0_ = Variable(image0).to(self.device)
                    input1_ = Variable(image1).to(self.device)
                    reconstructed = self.forward(input0_, input1_)
                    cross_reconstruction_loss = self.loss(input0_, reconstructed)

                    self.self_recon.append(self_reconstruction_loss.cpu().detach().numpy())
                    self.cross_recon.append(cross_reconstruction_loss.cpu().detach().numpy())
                    recon_loss = (self_reconstruction_loss + cross_reconstruction_loss) / 2
                else:
                    recon_loss = self_reconstruction_loss

                loss = recon_loss
                loss.backward()
                optimizer.step()
                tloss += float(loss)
                nb_batches += 1

            # comply with generative model
            sort_index1 = s_tp[:, 1].sort()[1]
            sorted_s_tp = s_tp[sort_index1]
            sort_index2 = sorted_s_tp[:, 0].sort()[1]
            Z, ZU, ZV = Z[sort_index1], ZU[sort_index1], ZV[sort_index1]
            Z, ZU, ZV = Z[sort_index2], ZU[sort_index2], ZV[sort_index2]
            min_, mean_, max_ = self.plot_z_distribution(Z, ZU, ZV)
            self.generative_parameter_update(self.X, self.Y, Z, ZU, ZV)

            epoch_loss = tloss / nb_batches
            test_loss = self.evaluate(test)
            self.train_loss.append(epoch_loss)
            self.test_loss.append(test_loss)

            if epoch_loss <= best_loss:
                es = 0
                best_loss = epoch_loss
            else:
                es += 1

            # plot result
            # Save images to check quality as training goes
            self.plot_recon(test)
            self.plot_simu_repre(min_, mean_, max_)
            self.plot_grad_simu_repre(min_, mean_, max_)
            self.plot_loss()
            end_time = time()
            logger.info(
                f"Epoch loss (train/test): {epoch_loss:.4}/{test_loss:.4} take {end_time - start_time:.3} seconds\n")

        print('Complete training')
        return

    def evaluate(self, test):
        self.to(self.device)
        self.training = False
        self.eval()
        dataloader = torch.utils.data.DataLoader(test, batch_size=32, num_workers=0, shuffle=False)
        tloss = 0.0
        nb_batches = 0

        with torch.no_grad():
            for data in dataloader:
                image = torch.tensor([[np.load(path)] for path in data[0]]).float()

                # self-reconstruction loss
                input_ = Variable(image).to(self.device)
                reconstructed, z, zu, zv = self.forward(input_)
                self_reconstruction_loss = self.loss(input_, reconstructed)

                # cross-reconstruction loss
                baseline_age = data[2]
                delta_age = data[3] - baseline_age
                index0, index1 = self.generate_sample(baseline_age, delta_age)
                image0 = image[index0]
                image1 = image[index1]
                if index0:
                    input0_ = Variable(image0).to(self.device)
                    input1_ = Variable(image1).to(self.device)
                    reconstructed = self.forward(input0_, input1_)
                    cross_reconstruction_loss = self.loss(input0_, reconstructed)
                    recon_loss = (self_reconstruction_loss + cross_reconstruction_loss) / 2
                else:
                    recon_loss = self_reconstruction_loss

                loss = recon_loss
                tloss += float(loss)
                nb_batches += 1

        loss = tloss / nb_batches
        self.training = True
        return loss

    @staticmethod
    def generate_sample(baseline_age, age):
        sample = []
        for index, base_a in enumerate(baseline_age):
            match_ba = [i for i, ba in enumerate(baseline_age) if 1e-5 < np.abs(ba - base_a) <= 0.05]
            if match_ba:
                sample.append([index, match_ba])
        result = []
        for index, match in sample:
            match_age = [i for i in match if 1e-5 < np.abs(age[i] - age[index]) <= 0.05]
            for ind in match_age:
                result.append([index, ind])
        index0 = [idx[0] for idx in result]
        index1 = [idx[1] for idx in result]
        return index0, index1

    def plot_recon(self, data, n_subject=3):
        # Plot the reconstruction
        fig, axes = plt.subplots(2 * n_subject, 10, figsize=(20, 4 * n_subject))
        plt.subplots_adjust(wspace=0, hspace=0)
        for j in range(n_subject):
            for i in range(10):
                test_image = torch.tensor(np.load(data[j * 10 + i][0])).resize(1, 1, 64, 64).float()
                test_image = Variable(test_image).to(self.device)
                out, _, _, _ = self.forward(test_image)
                axes[2 * j][i].matshow(255 * test_image[0][0].cpu().detach().numpy())
                axes[2 * j + 1][i].matshow(255 * out[0][0].cpu().detach().numpy())
        for axe in axes:
            for ax in axe:
                ax.set_xticks([])
                ax.set_yticks([])
        plt.savefig('visualization/reconstruction.png', bbox_inches='tight')
        plt.close()

    def plot_simu_repre(self, min_, mean_, max_):
        # Plot simulated data in all directions of the latent space
        # Z
        fig, axes = plt.subplots(dim_z, 11, figsize=(22, 2 * dim_z))
        plt.subplots_adjust(wspace=0, hspace=0)
        for i in range(dim_z):
            arange = np.linspace(min_[0][i], max_[0][i], num=11)
            for idx, j in enumerate(arange):
                simulated_latent = torch.tensor([[mean for mean in mean_[0]]], device=self.device)
                simulated_latent[0][i] = j
                encoded = torch.matmul(simulated_latent, self.U) + torch.matmul(simulated_latent, self.V)
                simulated_img = self.decoder(encoded)
                axes[i][idx].matshow(255 * simulated_img[0][0].cpu().detach().numpy())
        for axe in axes:
            for ax in axe:
                ax.set_xticks([])
                ax.set_yticks([])

        plt.savefig('visualization/simulation_latent_Z.png', bbox_inches='tight')
        plt.close()

        # ZU
        fig, axes = plt.subplots(dim_z, 11, figsize=(22, 2 * dim_z))
        plt.subplots_adjust(wspace=0, hspace=0)
        for i in range(dim_z):
            arange = np.linspace(min_[0][i], max_[0][i], num=11)
            for idx, j in enumerate(arange):
                simulated_latent = torch.tensor([[mean for mean in mean_[0]]], device=self.device)
                simulated_latent[0][i] = j
                encoded = torch.matmul(simulated_latent, self.U)
                simulated_img = self.decoder(encoded)
                axes[i][idx].matshow(255 * simulated_img[0][0].cpu().detach().numpy())
        for axe in axes:
            for ax in axe:
                ax.set_xticks([])
                ax.set_yticks([])

        plt.savefig('visualization/simulation_latent_ZU.png', bbox_inches='tight')
        plt.close()

        # ZV
        fig, axes = plt.subplots(dim_z, 11, figsize=(22, 2 * dim_z))
        plt.subplots_adjust(wspace=0, hspace=0)
        for i in range(dim_z):
            arange = np.linspace(min_[0][i], max_[0][i], num=11)
            for idx, j in enumerate(arange):
                simulated_latent = torch.tensor([[mean for mean in mean_[0]]], device=self.device)
                simulated_latent[0][i] = j
                encoded = torch.matmul(simulated_latent, self.V)
                simulated_img = self.decoder(encoded)
                axes[i][idx].matshow(255 * simulated_img[0][0].cpu().detach().numpy())
        for axe in axes:
            for ax in axe:
                ax.set_xticks([])
                ax.set_yticks([])

        plt.savefig('visualization/simulation_latent_ZV.png', bbox_inches='tight')
        plt.close()
        self.training = True

    def plot_grad_simu_repre(self, min_, mean_, max_):
        # Plot the gradient map of simulated data in all directions of the latent space
        # Z
        fig, axes = plt.subplots(dim_z, 10, figsize=(20, 2 * dim_z))
        plt.subplots_adjust(wspace=0, hspace=0)
        for i in range(dim_z):
            arange = np.linspace(min_[0][i], max_[0][i], num=11)
            for idx, j in enumerate(arange):
                simulated_latent = torch.tensor([[mean for mean in mean_[0]]], device=self.device)
                simulated_latent[0][i] = j
                encoded = torch.matmul(simulated_latent, self.U) + torch.matmul(simulated_latent, self.V)
                simulated_img = self.decoder(encoded)
                if idx == 0:
                    template = simulated_img
                    continue
                grad_img = simulated_img - template
                template = simulated_img
                axes[i][idx - 1].matshow(grad_img[0][0].cpu().detach().numpy(), cmap=matplotlib.cm.get_cmap('bwr'),
                                         norm=matplotlib.colors.CenteredNorm())
        for axe in axes:
            for ax in axe:
                ax.set_xticks([])
                ax.set_yticks([])
        fig.colorbar(
            matplotlib.cm.ScalarMappable(cmap=matplotlib.cm.get_cmap('bwr'), norm=matplotlib.colors.CenteredNorm()),
            cax=fig.add_axes([0.92, 0.15, 0.01, 0.7]))
        plt.savefig('visualization/gradient_simulation_latent_Z.png', bbox_inches='tight')
        plt.close()

        # ZU
        fig, axes = plt.subplots(dim_z, 10, figsize=(20, 2 * dim_z))
        plt.subplots_adjust(wspace=0, hspace=0)
        for i in range(dim_z):
            arange = np.linspace(min_[0][i], max_[0][i], num=11)
            for idx, j in enumerate(arange):
                simulated_latent = torch.tensor([[mean for mean in mean_[0]]], device=self.device)
                simulated_latent[0][i] = j
                encoded = torch.matmul(simulated_latent, self.U)
                simulated_img = self.decoder(encoded)
                if idx == 0:
                    template = simulated_img
                    continue
                grad_img = simulated_img - template
                template = simulated_img
                axes[i][idx - 1].matshow(grad_img[0][0].cpu().detach().numpy(), cmap=matplotlib.cm.get_cmap('bwr'),
                                         norm=matplotlib.colors.CenteredNorm())
        for axe in axes:
            for ax in axe:
                ax.set_xticks([])
                ax.set_yticks([])
        fig.colorbar(
            matplotlib.cm.ScalarMappable(cmap=matplotlib.cm.get_cmap('bwr'), norm=matplotlib.colors.CenteredNorm()),
            cax=fig.add_axes([0.92, 0.15, 0.01, 0.7]))
        plt.savefig('visualization/gradient_simulation_latent_ZU.png', bbox_inches='tight')
        plt.close()

        # ZV
        fig, axes = plt.subplots(dim_z, 10, figsize=(20, 2 * dim_z))
        plt.subplots_adjust(wspace=0, hspace=0)
        for i in range(dim_z):
            arange = np.linspace(min_[0][i], max_[0][i], num=11)
            for idx, j in enumerate(arange):
                simulated_latent = torch.tensor([[mean for mean in mean_[0]]], device=self.device)
                simulated_latent[0][i] = j
                encoded = torch.matmul(simulated_latent, self.V)
                simulated_img = self.decoder(encoded)
                if idx == 0:
                    template = simulated_img
                    continue
                grad_img = simulated_img - template
                template = simulated_img
                axes[i][idx - 1].matshow(grad_img[0][0].cpu().detach().numpy(), cmap=matplotlib.cm.get_cmap('bwr'),
                                         norm=matplotlib.colors.CenteredNorm())
        for axe in axes:
            for ax in axe:
                ax.set_xticks([])
                ax.set_yticks([])
        fig.colorbar(
            matplotlib.cm.ScalarMappable(cmap=matplotlib.cm.get_cmap('bwr'), norm=matplotlib.colors.CenteredNorm()),
            cax=fig.add_axes([0.92, 0.15, 0.01, 0.7]))
        plt.savefig('visualization/gradient_simulation_latent_ZV.png', bbox_inches='tight')
        plt.close()
        self.training = True

    @staticmethod
    def plot_z_distribution(Z, ZU, ZV):
        min_z, mean_z, max_z = [], [], []
        fig, axes = plt.subplots(1, dim_z, figsize=(4 * dim_z, 4))
        plt.subplots_adjust(wspace=0.1, hspace=0)
        for i in range(dim_z):
            z = Z[:, i].cpu().detach().numpy()
            axes[i].hist(z, bins=70, density=True)
            axes[i].set_title('{}-th dim'.format(i + 1))
            axes[i].set_xlabel(f"Min: {np.min(z):.4}\nMean: {np.mean(z):.4}\nMax: {np.max(z):.4}")
            min_z.append(np.min(z))
            mean_z.append(np.mean(z))
            max_z.append(np.max(z))
        for axe in axes:
            axe.set_yticks([])
            axe.set_xlim(left=-1, right=1)
        plt.savefig('visualization/Z_distribution.png', bbox_inches='tight')
        plt.close()

        min_zu, mean_zu, max_zu = [], [], []
        fig, axes = plt.subplots(1, dim_z, figsize=(4 * dim_z, 4))
        plt.subplots_adjust(wspace=0.1, hspace=0)
        for i in range(dim_z):
            zu = ZU[:, i].cpu().detach().numpy()
            axes[i].hist(zu, bins=70, density=True)
            axes[i].set_title('{}-th dim'.format(i + 1))
            axes[i].set_xlabel(f"Min: {np.min(zu):.4}\nMean: {np.mean(zu):.4}\nMax: {np.max(zu):.4}")
            min_zu.append(np.min(zu))
            mean_zu.append(np.mean(zu))
            max_zu.append(np.max(zu))
        for axe in axes:
            axe.set_yticks([])
            axe.set_xlim(left=-1, right=1)
        plt.savefig('visualization/ZU_distribution.png', bbox_inches='tight')
        plt.close()

        min_zv, mean_zv, max_zv = [], [], []
        fig, axes = plt.subplots(1, dim_z, figsize=(4 * dim_z, 4))
        plt.subplots_adjust(wspace=0.1, hspace=0)
        for i in range(dim_z):
            zv = ZV[:, i].cpu().detach().numpy()
            axes[i].hist(zv, bins=70, density=True)
            axes[i].set_title('{}-th dim'.format(i + 1))
            axes[i].set_xlabel(f"Min: {np.min(zv):.4}\nMean: {np.mean(zv):.4}\nMax: {np.max(zv):.4}")
            min_zv.append(np.min(zv))
            mean_zv.append(np.mean(zv))
            max_zv.append(np.max(zv))
        for axe in axes:
            axe.set_yticks([])
            axe.set_xlim(left=-1, right=1)
        plt.savefig('visualization/ZV_distribution.png', bbox_inches='tight')
        plt.close()

        min_ = [min_z, min_zu, min_zv]
        mean_ = [mean_z, mean_zu, mean_zv]
        max_ = [max_z, max_zu, max_zv]
        return min_, mean_, max_

    def plot_loss(self):
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        plt.subplots_adjust(wspace=0.2, hspace=0.1)

        axes[0][0].plot(self.train_loss, 'red')
        axes[0][0].plot(self.test_loss, 'blue')
        axes[0][0].set_title('Training and Testing Loss')
        axes[0][0].legend(['train', 'test'])

        axes[0][1].plot(self.cross_recon, 'blue')
        axes[0][1].plot(self.self_recon, 'red')
        axes[0][1].set_title('Self-Recon & Cross-Recon Loss')
        axes[0][1].legend(['cross-recon', 'self-recon', 'alignment'])
        axes[0][1].set_ylim(bottom=0)

        # axes[1][0].plot(self.align, 'darkviolet')
        # axes[1][0].set_title('Alignment Loss')
        # axes[1][0].set_ylim(bottom=0)

        axes[1][1].plot(self.log_p_b)
        axes[1][1].set_title('log p(b)')

        for axe in axes:
            for ax in axe:
                ax.set_xticks([])
                # ax.set_yticks([])
                ax.set_xlim(left=0)
        plt.savefig('visualization/loss.png', bbox_inches='tight')
        plt.close()

    def generative_parameter_update(self, X, Y, Z, ZU, ZV):
        start_time = time()

        X = Variable(X).to(self.device).float()
        Y = Variable(Y).to(self.device).float()
        Z = Variable(Z).to(self.device).float()
        ZU = Variable(ZU).to(self.device).float()
        ZV = Variable(ZV).to(self.device).float()

        xt = torch.transpose(X, 0, 1)
        yt = torch.transpose(Y, 0, 1)
        zt = torch.transpose(Z, 0, 1)
        xtx = torch.matmul(xt, X)
        xtx_inv = torch.inverse(torch.matmul(xt, X))
        yty = torch.matmul(yt, Y)
        ztz = torch.matmul(zt, Z)

        xt_zu = torch.matmul(xt, ZU)
        yt_zv = torch.matmul(yt, ZV)

        for epoch in range(5):
            # updata beta and b
            H = torch.matmul(torch.matmul(Y, self.D), yt) + self.sigma0_2 * torch.eye(N, device=self.device).float()
            H_inv = torch.inverse(H)
            xt_hi_x = torch.matmul(torch.matmul(xt, H_inv), X)
            xt_hi_z = torch.matmul(torch.matmul(xt, H_inv), Z)
            mat0 = xt_hi_x + 1 / self.sigma1_2 * xtx
            mat1 = xt_hi_z + 1 / self.sigma1_2 * xt_zu
            self.beta = torch.matmul(torch.inverse(mat0), mat1)

            xbeta = torch.matmul(X, self.beta)
            yt_z_xbeta = torch.matmul(yt, Z - xbeta)
            self.b = torch.matmul(
                torch.inverse(
                    (self.sigma0_2 + self.sigma2_2) * yty - 2 * self.sigma0_2 * self.sigma2_2 * torch.inverse(self.D)),
                self.sigma2_2 * yt_z_xbeta + self.sigma0_2 * yt_zv
            )

            # update variance parameter
            xbeta = torch.matmul(X, self.beta)
            yb = torch.matmul(Y, self.b)
            self.sigma0_2 = 1 / (N * dim_z) * torch.pow(torch.norm(Z - xbeta - yb, p='fro'), 2)
            # self.sigma1_2 = 1 / (N * dim_z) * torch.pow(torch.norm(ZU - xbeta, p='fro'), 2)
            self.sigma2_2 = 1 / (N * dim_z) * torch.pow(torch.norm(ZV - yb, p='fro'), 2)

            for i in range(1):
                dbbd = torch.matmul(torch.inverse(self.D),
                                    torch.matmul(self.b,
                                                 torch.matmul(torch.transpose(self.b, 0, 1), torch.inverse(self.D))))
                grad_d = -1 / 2 * (dim_z * torch.inverse(self.D) - dbbd)
                self.D = self.D + 1e-5 * grad_d

            # update U and V
            zt_xbeta = torch.matmul(zt, torch.matmul(X, self.beta))
            zt_yb = torch.matmul(zt, torch.matmul(Y, self.b))
            for i in range(50):
                vvt = torch.matmul(self.V, torch.transpose(self.V, 0, 1))
                uut = torch.matmul(self.U, torch.transpose(self.U, 0, 1))
                self.U = torch.matmul(torch.inverse(ztz + self.sigma1_2 * self.lam * vvt), zt_xbeta)
                self.V = torch.matmul(torch.inverse(ztz + self.sigma2_2 * self.lam * uut), zt_yb)

            xt_zu = torch.matmul(xt, torch.matmul(Z, self.U))
            yt_zv = torch.matmul(yt, torch.matmul(Z, self.V))

        end_time = time()
        bt_dinv_b = torch.matmul(torch.matmul(torch.transpose(self.b, 0, 1), torch.inverse(self.D)), self.b)
        log_pb = -1 / 2 * (dim_z * torch.log(torch.det(self.D)) + torch.trace(bt_dinv_b))
        log_pb = log_pb.cpu().detach().numpy()
        if log_pb != np.inf:
            self.log_p_b.append(log_pb)
        utv = torch.matmul(torch.transpose(self.U, 0, 1), self.V)
        utv_norm_2 = torch.pow(torch.norm(utv, p='fro'), 2).cpu().detach().numpy()
        logger.info(f"||U^T * V||^2 = {utv_norm_2:.4}")


class AE_starmen_wCRL(nn.Module):
    def __init__(self):
        super(AE_starmen_wCRL, self).__init__()
        nn.Module.__init__(self)
        self.name = 'AE_starmen_wCRL'
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.gamma = 1
        self.lam = 1.0

        self.conv1 = nn.Conv2d(1, 16, 3, stride=2, padding=1)  # 16 x 32 x 32
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)  # 32 x 16 x 16
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)  # 32 x 8 x 8
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.fc10 = nn.Linear(2048, dim_z)
        self.fc11 = nn.Linear(2048, dim_z)

        self.fc3 = nn.Linear(dim_z, 512)
        self.upconv1 = nn.ConvTranspose2d(8, 64, 3, stride=2, padding=1, output_padding=1)  # 64 x 16 x 16
        self.upconv2 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)  # 32 x 32 x 32
        self.upconv3 = nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1)  # 1 x 64 x 64
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(32)

        self.train_loss, self.test_loss = [], []
        self.self_recon, self.cross_recon, self.align = [], [], []
        self.log_p_b = []

        self.X, self.Y = None, None

        self.beta = torch.rand(size=[p, dim_z], device=self.device)
        self.b = torch.normal(mean=0, std=1, size=[q * I, dim_z], device=self.device)
        self.U = torch.diag(torch.tensor([1 for i in range(dim_z // 2)] + [0 for i in range(dim_z - dim_z // 2)],
                                         device=self.device)).float()
        self.V = torch.eye(dim_z, device=self.device) - self.U
        self.sigma0_2, self.sigma1_2, self.sigma2_2 = 1, 0.5, 1
        self.D = torch.eye(q * I, device=self.device).float()

    def encoder(self, image):
        h1 = F.relu(self.bn1(self.conv1(image)))
        h2 = F.relu(self.bn2(self.conv2(h1)))
        h3 = F.relu(self.bn3(self.conv3(h2)))
        z = torch.tanh(self.fc10(h3.view(h3.size()[0], -1)))
        return z

    def decoder(self, encoded):
        h6 = F.relu(self.fc3(encoded)).reshape([encoded.size()[0], 8, 8, 8])
        h7 = F.relu(self.bn4(self.upconv1(h6)))
        h8 = F.relu(self.bn5(self.upconv2(h7)))
        reconstructed = F.relu(self.upconv3(h8))
        return reconstructed

    def forward(self, image0, image1=None):
        z0 = self.encoder(image0)
        zu0 = torch.matmul(z0, self.U)
        zv0 = torch.matmul(z0, self.V)
        if image1 is not None:
            z1 = self.encoder(image1)
            encoded = torch.matmul(z1, self.U) + zv0
            reconstructed = self.decoder(encoded)
            return reconstructed
        else:
            encoded = zu0 + zv0
            reconstructed = self.decoder(encoded)
            return reconstructed, z0, zu0, zv0

    @staticmethod
    def loss(input_, reconstructed):
        recon_loss = torch.sum((reconstructed - input_) ** 2) / input_.shape[0]
        return recon_loss

    def train_(self, data_loader, test, optimizer, num_epochs):

        self.to(self.device)
        best_loss = 1e10
        es = 0

        for epoch in range(num_epochs):

            start_time = time()
            if es == 100:
                break

            logger.info('#### Epoch {}/{} ####'.format(epoch + 1, num_epochs))

            tloss = 0.0
            nb_batches = 0

            s_tp, Z, ZU, ZV = None, None, None, None
            for data in data_loader:
                image = torch.tensor([[np.load(path)] for path in data[0]], device=self.device).float()
                optimizer.zero_grad()

                # self-reconstruction loss
                input_ = Variable(image).to(self.device)
                reconstructed, z, zu, zv = self.forward(input_)
                self_reconstruction_loss = self.loss(input_, reconstructed)

                # store Z, ZU, ZV
                subject = torch.tensor([[s for s in data[1]]], device=self.device)
                tp = torch.tensor([[tp for tp in data[4]]], device=self.device)
                st = torch.transpose(torch.cat((subject, tp), 0), 0, 1)
                if s_tp is None:
                    s_tp, Z, ZU, ZV = st, z, zu, zv
                else:
                    s_tp = torch.cat((s_tp, st), 0)
                    Z = torch.cat((Z, z), 0)
                    ZU = torch.cat((ZU, zu), 0)
                    ZV = torch.cat((ZV, zv), 0)

                loss = self_reconstruction_loss
                loss.backward()
                optimizer.step()
                tloss += float(loss)
                nb_batches += 1

            # comply with generative model
            sort_index1 = s_tp[:, 1].sort()[1]
            sorted_s_tp = s_tp[sort_index1]
            sort_index2 = sorted_s_tp[:, 0].sort()[1]
            Z, ZU, ZV = Z[sort_index1], ZU[sort_index1], ZV[sort_index1]
            Z, ZU, ZV = Z[sort_index2], ZU[sort_index2], ZV[sort_index2]
            min_, mean_, max_ = self.plot_z_distribution(Z, ZU, ZV)
            self.generative_parameter_update(self.X, self.Y, Z, ZU, ZV)

            epoch_loss = tloss / nb_batches
            test_loss = self.evaluate(test)
            self.train_loss.append(epoch_loss)
            self.test_loss.append(test_loss)

            if epoch_loss <= best_loss:
                es = 0
                best_loss = epoch_loss
            else:
                es += 1

            # plot result
            # Save images to check quality as training goes
            self.plot_recon(test)
            self.plot_simu_repre(min_, mean_, max_)
            self.plot_grad_simu_repre(min_, mean_, max_)
            self.plot_loss()
            end_time = time()
            logger.info(
                f"Epoch loss (train/test): {epoch_loss:.4}/{test_loss:.4} take {np.round(end_time - start_time, 3)} seconds\n")

        print('Complete training')
        return

    def evaluate(self, test):
        self.to(self.device)
        self.training = False
        self.eval()
        dataloader = torch.utils.data.DataLoader(test, batch_size=32, num_workers=0, shuffle=False)
        tloss = 0.0
        nb_batches = 0

        with torch.no_grad():
            for data in dataloader:
                image = torch.tensor([[np.load(path)] for path in data[0]]).float()

                # self-reconstruction loss
                input_ = Variable(image).to(self.device)
                reconstructed, z, zu, zv = self.forward(input_)
                self_reconstruction_loss = self.loss(input_, reconstructed)

                loss = self_reconstruction_loss
                tloss += float(loss)
                nb_batches += 1

        loss = tloss / nb_batches
        self.training = True
        return loss

    def plot_recon(self, data, n_subject=3):
        # Plot the reconstruction
        fig, axes = plt.subplots(2 * n_subject, 10, figsize=(20, 4 * n_subject))
        plt.subplots_adjust(wspace=0, hspace=0)
        for j in range(n_subject):
            for i in range(10):
                test_image = torch.tensor(np.load(data[j * 10 + i][0])).resize(1, 1, 64, 64).float()
                test_image = Variable(test_image).to(self.device)
                out, _, _, _ = self.forward(test_image)
                axes[2 * j][i].matshow(255 * test_image[0][0].cpu().detach().numpy())
                axes[2 * j + 1][i].matshow(255 * out[0][0].cpu().detach().numpy())
        for axe in axes:
            for ax in axe:
                ax.set_xticks([])
                ax.set_yticks([])
        plt.savefig('visualization/reconstruction.png', bbox_inches='tight')
        plt.close()

    def plot_simu_repre(self, min_, mean_, max_):
        # Plot simulated data in all directions of the latent space
        # Z
        fig, axes = plt.subplots(dim_z, 11, figsize=(22, 2 * dim_z))
        plt.subplots_adjust(wspace=0, hspace=0)
        for i in range(dim_z):
            arange = np.linspace(min_[0][i], max_[0][i], num=11)
            for idx, j in enumerate(arange):
                simulated_latent = torch.tensor([[mean for mean in mean_[0]]], device=self.device)
                simulated_latent[0][i] = j
                encoded = torch.matmul(simulated_latent, self.U) + torch.matmul(simulated_latent, self.V)
                simulated_img = self.decoder(encoded)
                axes[i][idx].matshow(255 * simulated_img[0][0].cpu().detach().numpy())
        for axe in axes:
            for ax in axe:
                ax.set_xticks([])
                ax.set_yticks([])

        plt.savefig('visualization/simulation_latent_Z.png', bbox_inches='tight')
        plt.close()

        # ZU
        fig, axes = plt.subplots(dim_z, 11, figsize=(22, 2 * dim_z))
        plt.subplots_adjust(wspace=0, hspace=0)
        for i in range(dim_z):
            arange = np.linspace(min_[1][i], max_[1][i], num=11)
            for idx, j in enumerate(arange):
                simulated_latent = torch.tensor([[mean for mean in mean_[1]]], device=self.device)
                simulated_latent[0][i] = j
                encoded = torch.matmul(simulated_latent, self.U)
                simulated_img = self.decoder(encoded)
                axes[i][idx].matshow(255 * simulated_img[0][0].cpu().detach().numpy())
        for axe in axes:
            for ax in axe:
                ax.set_xticks([])
                ax.set_yticks([])

        plt.savefig('visualization/simulation_latent_ZU.png', bbox_inches='tight')
        plt.close()

        # ZV
        fig, axes = plt.subplots(dim_z, 11, figsize=(22, 2 * dim_z))
        plt.subplots_adjust(wspace=0, hspace=0)
        for i in range(dim_z):
            arange = np.linspace(min_[2][i], max_[2][i], num=11)
            for idx, j in enumerate(arange):
                simulated_latent = torch.tensor([[mean for mean in mean_[2]]], device=self.device)
                simulated_latent[0][i] = j
                encoded = torch.matmul(simulated_latent, self.V)
                simulated_img = self.decoder(encoded)
                axes[i][idx].matshow(255 * simulated_img[0][0].cpu().detach().numpy())
        for axe in axes:
            for ax in axe:
                ax.set_xticks([])
                ax.set_yticks([])

        plt.savefig('visualization/simulation_latent_ZV.png', bbox_inches='tight')
        plt.close()
        self.training = True

    def plot_grad_simu_repre(self, min_, mean_, max_):
        # Plot the gradient map of simulated data in all directions of the latent space
        # Z
        fig, axes = plt.subplots(dim_z, 10, figsize=(20, 2 * dim_z))
        plt.subplots_adjust(wspace=0, hspace=0)
        for i in range(dim_z):
            arange = np.linspace(min_[0][i], max_[0][i], num=11)
            for idx, j in enumerate(arange):
                simulated_latent = torch.tensor([[mean for mean in mean_[0]]], device=self.device)
                simulated_latent[0][i] = j
                encoded = torch.matmul(simulated_latent, self.U) + torch.matmul(simulated_latent, self.V)
                simulated_img = self.decoder(encoded)
                if idx == 0:
                    template = simulated_img
                    continue
                grad_img = simulated_img - template
                template = simulated_img
                axes[i][idx - 1].matshow(grad_img[0][0].cpu().detach().numpy(), cmap=matplotlib.cm.get_cmap('bwr'),
                                         norm=matplotlib.colors.CenteredNorm())
        for axe in axes:
            for ax in axe:
                ax.set_xticks([])
                ax.set_yticks([])
        fig.colorbar(
            matplotlib.cm.ScalarMappable(cmap=matplotlib.cm.get_cmap('bwr'), norm=matplotlib.colors.CenteredNorm()),
            cax=fig.add_axes([0.92, 0.15, 0.01, 0.7]))
        plt.savefig('visualization/gradient_simulation_latent_Z.png', bbox_inches='tight')
        plt.close()

        # ZU
        fig, axes = plt.subplots(dim_z, 10, figsize=(20, 2 * dim_z))
        plt.subplots_adjust(wspace=0, hspace=0)
        for i in range(dim_z):
            arange = np.linspace(min_[1][i], max_[1][i], num=11)
            for idx, j in enumerate(arange):
                simulated_latent = torch.tensor([[mean for mean in mean_[1]]], device=self.device)
                simulated_latent[0][i] = j
                encoded = torch.matmul(simulated_latent, self.U)
                simulated_img = self.decoder(encoded)
                if idx == 0:
                    template = simulated_img
                    continue
                grad_img = simulated_img - template
                template = simulated_img
                axes[i][idx - 1].matshow(grad_img[0][0].cpu().detach().numpy(), cmap=matplotlib.cm.get_cmap('bwr'),
                                         norm=matplotlib.colors.CenteredNorm())
        for axe in axes:
            for ax in axe:
                ax.set_xticks([])
                ax.set_yticks([])
        fig.colorbar(
            matplotlib.cm.ScalarMappable(cmap=matplotlib.cm.get_cmap('bwr'), norm=matplotlib.colors.CenteredNorm()),
            cax=fig.add_axes([0.92, 0.15, 0.01, 0.7]))
        plt.savefig('visualization/gradient_simulation_latent_ZU.png', bbox_inches='tight')
        plt.close()

        # ZV
        fig, axes = plt.subplots(dim_z, 10, figsize=(20, 2 * dim_z))
        plt.subplots_adjust(wspace=0, hspace=0)
        for i in range(dim_z):
            arange = np.linspace(min_[2][i], max_[2][i], num=11)
            for idx, j in enumerate(arange):
                simulated_latent = torch.tensor([[mean for mean in mean_[2]]], device=self.device)
                simulated_latent[0][i] = j
                encoded = torch.matmul(simulated_latent, self.V)
                simulated_img = self.decoder(encoded)
                if idx == 0:
                    template = simulated_img
                    continue
                grad_img = simulated_img - template
                template = simulated_img
                axes[i][idx - 1].matshow(grad_img[0][0].cpu().detach().numpy(), cmap=matplotlib.cm.get_cmap('bwr'),
                                         norm=matplotlib.colors.CenteredNorm())
        for axe in axes:
            for ax in axe:
                ax.set_xticks([])
                ax.set_yticks([])
        fig.colorbar(
            matplotlib.cm.ScalarMappable(cmap=matplotlib.cm.get_cmap('bwr'), norm=matplotlib.colors.CenteredNorm()),
            cax=fig.add_axes([0.92, 0.15, 0.01, 0.7]))
        plt.savefig('visualization/gradient_simulation_latent_ZV.png', bbox_inches='tight')
        plt.close()
        self.training = True

    @staticmethod
    def plot_z_distribution(Z, ZU, ZV):
        min_z, mean_z, max_z = [], [], []
        fig, axes = plt.subplots(1, dim_z, figsize=(4 * dim_z, 4))
        plt.subplots_adjust(wspace=0.1, hspace=0)
        for i in range(dim_z):
            z = Z[:, i].cpu().detach().numpy()
            axes[i].hist(z, bins=70, density=True)
            axes[i].set_title('{}-th dim'.format(i + 1))
            axes[i].set_xlabel(f"Min: {np.min(z):.4}\nMean: {np.mean(z):.4}\nMax: {np.max(z):.4}")
            min_z.append(np.min(z))
            mean_z.append(np.mean(z))
            max_z.append(np.max(z))
        for axe in axes:
            axe.set_yticks([])
            axe.set_xlim(left=-1, right=1)
        plt.savefig('visualization/Z_distribution.png', bbox_inches='tight')
        plt.close()

        min_zu, mean_zu, max_zu = [], [], []
        fig, axes = plt.subplots(1, dim_z, figsize=(4 * dim_z, 4))
        plt.subplots_adjust(wspace=0.1, hspace=0)
        for i in range(dim_z):
            zu = ZU[:, i].cpu().detach().numpy()
            axes[i].hist(zu, bins=70, density=True)
            axes[i].set_title('{}-th dim'.format(i + 1))
            axes[i].set_xlabel(f"Min: {np.min(zu):.4}\nMean: {np.mean(zu):.4}\nMax: {np.max(zu):.4}")
            min_zu.append(np.min(zu))
            mean_zu.append(np.mean(zu))
            max_zu.append(np.max(zu))
        for axe in axes:
            axe.set_yticks([])
            axe.set_xlim(left=-1, right=1)
        plt.savefig('visualization/ZU_distribution.png', bbox_inches='tight')
        plt.close()

        min_zv, mean_zv, max_zv = [], [], []
        fig, axes = plt.subplots(1, dim_z, figsize=(4 * dim_z, 4))
        plt.subplots_adjust(wspace=0.1, hspace=0)
        for i in range(dim_z):
            zv = ZV[:, i].cpu().detach().numpy()
            axes[i].hist(zv, bins=70, density=True)
            axes[i].set_title('{}-th dim'.format(i + 1))
            axes[i].set_xlabel(f"Min: {np.min(zv):.4}\nMean: {np.mean(zv):.4}\nMax: {np.max(zv):.4}")
            min_zv.append(np.min(zv))
            mean_zv.append(np.mean(zv))
            max_zv.append(np.max(zv))
        for axe in axes:
            axe.set_yticks([])
            axe.set_xlim(left=-1, right=1)
        plt.savefig('visualization/ZV_distribution.png', bbox_inches='tight')
        plt.close()

        min_ = [min_z, min_zu, min_zv]
        mean_ = [mean_z, mean_zu, mean_zv]
        max_ = [max_z, max_zu, max_zv]
        return min_, mean_, max_

    def plot_loss(self):
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        plt.subplots_adjust(wspace=0.2, hspace=0.1)

        axes[0][0].plot(self.train_loss, 'red')
        axes[0][0].plot(self.test_loss, 'blue')
        axes[0][0].set_title('Training and Testing Loss')
        axes[0][0].legend(['train', 'test'])

        axes[0][1].plot(self.cross_recon, 'blue')
        axes[0][1].plot(self.self_recon, 'red')
        axes[0][1].set_title('Self-Recon & Cross-Recon Loss')
        axes[0][1].legend(['cross-recon', 'self-recon', 'alignment'])
        axes[0][1].set_ylim(bottom=0)

        # axes[1][0].plot(self.align, 'darkviolet')
        # axes[1][0].set_title('Alignment Loss')
        # axes[1][0].set_ylim(bottom=0)

        axes[1][1].plot(self.log_p_b)
        axes[1][1].set_title('log p(b)')

        for axe in axes:
            for ax in axe:
                ax.set_xticks([])
                # ax.set_yticks([])
                ax.set_xlim(left=0)
        plt.savefig('visualization/loss.png', bbox_inches='tight')
        plt.close()

    def generative_parameter_update(self, X, Y, Z, ZU, ZV):
        start_time = time()

        X = Variable(X).to(self.device).float()
        Y = Variable(Y).to(self.device).float()
        Z = Variable(Z).to(self.device).float()
        ZU = Variable(ZU).to(self.device).float()
        ZV = Variable(ZV).to(self.device).float()

        xt = torch.transpose(X, 0, 1)
        yt = torch.transpose(Y, 0, 1)
        zt = torch.transpose(Z, 0, 1)
        xtx = torch.matmul(xt, X)
        xtx_inv = torch.inverse(torch.matmul(xt, X))
        yty = torch.matmul(yt, Y)
        ztz = torch.matmul(zt, Z)

        xt_zu = torch.matmul(xt, ZU)
        yt_zv = torch.matmul(yt, ZV)

        for epoch in range(5):
            # updata beta and b
            H = torch.matmul(torch.matmul(Y, self.D), yt) + self.sigma0_2 * torch.eye(N, device=self.device).float()
            H_inv = torch.inverse(H)
            xt_hi_x = torch.matmul(torch.matmul(xt, H_inv), X)
            xt_hi_z = torch.matmul(torch.matmul(xt, H_inv), Z)
            mat0 = xt_hi_x + 1 / self.sigma1_2 * xtx
            mat1 = xt_hi_z + 1 / self.sigma1_2 * xt_zu
            self.beta = torch.matmul(torch.inverse(mat0), mat1)

            xbeta = torch.matmul(X, self.beta)
            yt_z_xbeta = torch.matmul(yt, Z - xbeta)
            self.b = torch.matmul(
                torch.inverse(
                    (self.sigma0_2 + self.sigma2_2) * yty - 2 * self.sigma0_2 * self.sigma2_2 * torch.inverse(self.D)),
                self.sigma2_2 * yt_z_xbeta + self.sigma0_2 * yt_zv
            )

            # update variance parameter
            xbeta = torch.matmul(X, self.beta)
            yb = torch.matmul(Y, self.b)
            self.sigma0_2 = 1 / (N * dim_z) * torch.pow(torch.norm(Z - xbeta - yb, p='fro'), 2)
            # self.sigma1_2 = 1 / (N * dim_z) * torch.pow(torch.norm(ZU - xbeta, p='fro'), 2)
            self.sigma2_2 = 1 / (N * dim_z) * torch.pow(torch.norm(ZV - yb, p='fro'), 2)

            for i in range(1):
                dbbd = torch.matmul(torch.inverse(self.D),
                                    torch.matmul(self.b,
                                                 torch.matmul(torch.transpose(self.b, 0, 1), torch.inverse(self.D))))
                grad_d = -1 / 2 * (dim_z * torch.inverse(self.D) - dbbd)
                self.D = self.D + 1e-5 * grad_d

            # update U and V
            zt_xbeta = torch.matmul(zt, torch.matmul(X, self.beta))
            zt_yb = torch.matmul(zt, torch.matmul(Y, self.b))
            for i in range(50):
                vvt = torch.matmul(self.V, torch.transpose(self.V, 0, 1))
                uut = torch.matmul(self.U, torch.transpose(self.U, 0, 1))
                self.U = torch.matmul(torch.inverse(ztz + self.sigma1_2 * self.lam * vvt), zt_xbeta)
                self.V = torch.matmul(torch.inverse(ztz + self.sigma2_2 * self.lam * uut), zt_yb)

            xt_zu = torch.matmul(xt, torch.matmul(Z, self.U))
            yt_zv = torch.matmul(yt, torch.matmul(Z, self.V))

        end_time = time()
        bt_dinv_b = torch.matmul(torch.matmul(torch.transpose(self.b, 0, 1), torch.inverse(self.D)), self.b)
        log_pb = -1 / 2 * (dim_z * torch.log(torch.det(self.D)) + torch.trace(bt_dinv_b))
        log_pb = log_pb.cpu().detach().numpy()
        if log_pb != np.inf:
            self.log_p_b.append(log_pb)
        utv = torch.matmul(torch.transpose(self.U, 0, 1), self.V)
        utv_norm_2 = torch.pow(torch.norm(utv, p='fro'), 2).cpu().detach().numpy()
        logger.info(f"||U^T * V||^2 = {utv_norm_2:.4}")


class beta_VAE(nn.Module):
    def __init__(self):
        super(beta_VAE, self).__init__()
        nn.Module.__init__(self)
        self.name = 'beta_VAE'
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.beta = 5

        self.conv1 = nn.Conv2d(1, 16, 3, stride=2, padding=1)  # 16 x 32 x 32
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)  # 32 x 16 x 16
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)  # 32 x 8 x 8
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.fc10 = nn.Linear(2048, dim_z)
        self.fc11 = nn.Linear(2048, dim_z)

        self.fc3 = nn.Linear(dim_z, 512)
        self.upconv1 = nn.ConvTranspose2d(8, 64, 3, stride=2, padding=1, output_padding=1)  # 64 x 16 x 16
        self.upconv2 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)  # 32 x 32 x 32
        self.upconv3 = nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1)  # 1 x 64 x 64
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(32)

        self.train_recon_loss, self.test_recon_loss = [], []

    def encoder(self, image):
        h1 = F.relu(self.bn1(self.conv1(image)))
        h2 = F.relu(self.bn2(self.conv2(h1)))
        h3 = F.relu(self.bn3(self.conv3(h2)))
        mu = torch.tanh(self.fc10(h3.view(h3.size()[0], -1)))
        logVar = self.fc11(h3.view(h3.size()[0], -1))
        return mu, logVar

    def decoder(self, encoded):
        h6 = F.relu(self.fc3(encoded)).reshape([encoded.size()[0], 8, 8, 8])
        h7 = F.relu(self.bn4(self.upconv1(h6)))
        h8 = F.relu(self.bn5(self.upconv2(h7)))
        reconstructed = F.relu(self.upconv3(h8))
        return reconstructed

    def reparametrize(self, mu, logVar):
        # Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar / 2).to(self.device)
        eps = torch.normal(mean=torch.tensor([0 for i in range(std.shape[1])]).float(), std=1).to(self.device)
        if self.beta != 0:  # beta VAE
            return mu + eps * std
        else:  # regular AE
            return mu

    def forward(self, image):
        mu, logVar = self.encoder(image)
        if self.training:
            encoded = self.reparametrize(mu, logVar)
        else:
            encoded = mu
        reconstructed = self.decoder(encoded)
        return mu, logVar, reconstructed

    def loss(self, mu, logVar, reconstructed, input_):
        kl_divergence = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp()) / mu.shape[0]
        recon_error = torch.sum((reconstructed - input_) ** 2) / input_.shape[0]
        return recon_error, kl_divergence

    def train_(self, data_loader, test, optimizer, num_epochs):

        self.to(self.device)
        best_loss = 1e10
        es = 0

        for epoch in range(num_epochs):

            start_time = time()
            if es == 100:
                break

            logger.info('Epoch {}/{}'.format(epoch + 1, num_epochs))

            tloss = 0.0
            nb_batches = 0

            for data in data_loader:
                image = torch.tensor([[np.load(path)] for path in data[0]], device=self.device).float()
                optimizer.zero_grad()

                input_ = Variable(image).to(self.device)
                mu, logVar, reconstructed = self.forward(input_)
                reconstruction_loss, kl_loss = self.loss(mu, logVar, input_, reconstructed)
                loss = reconstruction_loss + self.beta * kl_loss
                self.train_recon_loss.append(reconstruction_loss.cpu().detach().numpy())

                loss.backward()
                optimizer.step()
                tloss += float(loss)
                nb_batches += 1

            epoch_loss = tloss / nb_batches
            test_loss = self.evaluate(test)

            if epoch_loss <= best_loss:
                es = 0
                best_loss = epoch_loss
            else:
                es += 1
            end_time = time()
            self.plot_recon(test)
            logger.info(
                f"Recon / KL loss: {reconstruction_loss:.3}/{kl_loss:.3}")
            logger.info(
                f"Epoch loss (train/test): {epoch_loss:.3}/{test_loss:.3} took {end_time - start_time:.2} seconds")

        print('Complete training')
        return

    def evaluate(self, data):
        self.to(self.device)
        self.training = False
        self.eval()
        dataloader = torch.utils.data.DataLoader(data, batch_size=128, num_workers=0, shuffle=False, drop_last=False)
        tloss = 0.0
        nb_batches = 0

        with torch.no_grad():
            for data in dataloader:
                image = torch.tensor([[np.load(path)] for path in data[0]]).float()

                input_ = Variable(image).to(self.device)
                mu, logVar, reconstructed = self.forward(input_)
                reconstruction_loss, kl_loss = self.loss(mu, logVar, input_, reconstructed)
                loss = reconstruction_loss + self.beta * kl_loss
                self.test_recon_loss.append(reconstruction_loss.cpu().detach().numpy())

                tloss += float(loss)
                nb_batches += 1

        loss = tloss / nb_batches
        self.training = True
        return loss

    def plot_recon(self, data, n_subject=3):
        # Plot the reconstruction
        fig, axes = plt.subplots(2 * n_subject, 10, figsize=(20, 4 * n_subject))
        plt.subplots_adjust(wspace=0, hspace=0)
        for j in range(n_subject):
            for i in range(10):
                test_image = torch.tensor(np.load(data[j * 10 + i][0])).resize(1, 1, 64, 64).float()
                test_image = Variable(test_image).to(self.device)
                _, _, out = self.forward(test_image)
                axes[2 * j][i].matshow(255 * test_image[0][0].cpu().detach().numpy())
                axes[2 * j + 1][i].matshow(255 * out[0][0].cpu().detach().numpy())
        for axe in axes:
            for ax in axe:
                ax.set_xticks([])
                ax.set_yticks([])
        plt.savefig('visualization/reconstruction.png', bbox_inches='tight')
        plt.close()


class ML_VAE(nn.Module):
    def __init__(self):
        super(ML_VAE, self).__init__()
        nn.Module.__init__(self)
        self.name = 'ML_VAE'
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.beta = 5

        self.conv1 = nn.Conv2d(1, 16, 3, stride=2, padding=1)  # 16 x 32 x 32
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)  # 32 x 16 x 16
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)  # 32 x 8 x 8
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.fc10 = nn.Linear(2048, dim_z)
        self.fc11 = nn.Linear(2048, dim_z)
        self.fc12 = nn.Linear(2048, dim_z)
        self.fc13 = nn.Linear(2048, dim_z)

        self.fc3 = nn.Linear(dim_z * 2, 512)
        self.upconv1 = nn.ConvTranspose2d(8, 64, 3, stride=2, padding=1, output_padding=1)  # 64 x 16 x 16
        self.upconv2 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)  # 32 x 32 x 32
        self.upconv3 = nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1)  # 1 x 64 x 64
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(32)

        self.train_recon_loss, self.test_recon_loss = [], []

    def encoder(self, image):
        h1 = F.relu(self.bn1(self.conv1(image)))
        h2 = F.relu(self.bn2(self.conv2(h1)))
        h3 = F.relu(self.bn3(self.conv3(h2)))

        # style
        style_mu = torch.tanh(self.fc10(h3.view(h3.size()[0], -1)))
        style_logVar = self.fc11(h3.view(h3.size()[0], -1))

        # class
        class_mu = torch.tanh(self.fc12(h3.view(h3.size()[0], -1)))
        class_logVar = self.fc13(h3.view(h3.size()[0], -1))
        return style_mu, style_logVar, class_mu, class_logVar

    def decoder(self, encoded):
        h6 = F.relu(self.fc3(encoded)).reshape([encoded.size()[0], 8, 8, 8])
        h7 = F.relu(self.bn4(self.upconv1(h6)))
        h8 = F.relu(self.bn5(self.upconv2(h7)))
        reconstructed = F.relu(self.upconv3(h8))
        return reconstructed

    def reparametrize(self, mu, logVar):
        # Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar / 2).to(self.device)
        eps = torch.normal(mean=torch.tensor([0 for i in range(std.shape[1])]).float(), std=1).to(self.device)
        return mu + eps * std

    def forward(self, image):
        style_mu, style_logVar, class_mu, class_logVar = self.encoder(image)
        if self.training:
            style_encoded = self.reparametrize(style_mu, style_logVar)
            class_encoded = self.reparametrize(class_mu, class_logVar)
        else:
            style_encoded = style_mu
            class_encoded = class_mu
        return style_mu, style_logVar, class_mu, class_logVar, style_encoded, class_encoded

    def loss(self, mu, logVar, reconstructed, input_):
        kl_divergence = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp()) / mu.shape[0]
        recon_error = torch.sum((reconstructed - input_) ** 2) / input_.shape[0]
        return recon_error, kl_divergence

    def train_(self, data_loader, test, optimizer, num_epochs):

        self.to(self.device)
        best_loss = 1e10
        es = 0

        for epoch in range(num_epochs):

            start_time = time()
            if es == 100:
                break

            logger.info('Epoch {}/{}'.format(epoch + 1, num_epochs))

            tloss = 0.0
            nb_batches = 0

            ZU, ZV = None, None
            for data in data_loader:
                image = torch.tensor([[np.load(path)] for path in data[0]], device=self.device).float()
                optimizer.zero_grad()

                input_ = Variable(image).to(self.device)
                style_mu, style_logVar, class_mu, class_logVar, style_encoded, class_encoded = self.forward(input_)
                encoded = torch.cat((style_encoded, class_encoded), dim=1)
                reconstructed = self.decoder(encoded)
                reconstruction_loss, style_kl_loss = self.loss(style_mu, style_logVar, input_, reconstructed)
                reconstruction_loss, class_kl_loss = self.loss(class_mu, class_logVar, input_, reconstructed)
                loss = reconstruction_loss + self.beta * (style_kl_loss + class_kl_loss)
                self.train_recon_loss.append(reconstruction_loss.cpu().detach().numpy())

                # store ZU, ZV
                if ZU is None:
                    ZU, ZV = class_encoded, style_encoded
                else:
                    ZU = torch.cat((ZU, class_encoded), 0)
                    ZV = torch.cat((ZV, style_encoded), 0)

                loss.backward()
                optimizer.step()
                tloss += float(loss)
                nb_batches += 1

            epoch_loss = tloss / nb_batches
            test_loss = self.evaluate(test)

            if epoch_loss <= best_loss:
                es = 0
                best_loss = epoch_loss
            else:
                es += 1
            end_time = time()

            # plot the results
            self.plot_recon(test)
            min_, mean_, max_ = self.plot_z_distribution(ZU, ZV)
            self.plot_simu_repre(min_, mean_, max_)
            self.plot_grad_simu_repre(min_, mean_, max_)

            logger.info(
                f"Recon / KL loss: {reconstruction_loss.cpu().detach().numpy():.3}/{(style_kl_loss + class_kl_loss).cpu().detach().numpy():.3}")
            logger.info(
                f"Epoch loss (train/test): {epoch_loss:.3}/{test_loss:.3} took {end_time - start_time:.1} seconds")

        print('Complete training')
        return

    def evaluate(self, data):
        self.to(self.device)
        self.training = False
        self.eval()
        dataloader = torch.utils.data.DataLoader(data, batch_size=128, num_workers=0, shuffle=False, drop_last=False)
        tloss = 0.0
        nb_batches = 0

        with torch.no_grad():
            for data in dataloader:
                image = torch.tensor([[np.load(path)] for path in data[0]]).float()
                input_ = Variable(image).to(self.device)
                style_mu, style_logVar, class_mu, class_logVar, style_encoded, class_encoded = self.forward(input_)
                encoded = torch.cat((style_encoded, class_encoded), dim=1)
                reconstructed = self.decoder(encoded)
                reconstruction_loss, style_kl_loss = self.loss(style_mu, style_logVar, input_, reconstructed)
                reconstruction_loss, class_kl_loss = self.loss(class_mu, class_logVar, input_, reconstructed)
                loss = reconstruction_loss + self.beta * (style_kl_loss + class_kl_loss)
                self.test_recon_loss.append(reconstruction_loss.cpu().detach().numpy())

                tloss += float(loss)
                nb_batches += 1

        loss = tloss / nb_batches
        self.training = True
        return loss

    def plot_recon(self, data, n_subject=3):
        # Plot the reconstruction
        fig, axes = plt.subplots(2 * n_subject, 10, figsize=(20, 4 * n_subject))
        plt.subplots_adjust(wspace=0, hspace=0)
        for j in range(n_subject):
            for i in range(10):
                test_image = torch.tensor(np.load(data[j * 10 + i][0])).resize(1, 1, 64, 64).float()
                test_image = Variable(test_image).to(self.device)
                _, _, _, _, style_encoded, class_encoded = self.forward(test_image)
                encoded = torch.cat((style_encoded, class_encoded), dim=1)
                out = self.decoder(encoded)
                axes[2 * j][i].matshow(255 * test_image[0][0].cpu().detach().numpy())
                axes[2 * j + 1][i].matshow(255 * out[0][0].cpu().detach().numpy())
        for axe in axes:
            for ax in axe:
                ax.set_xticks([])
                ax.set_yticks([])
        plt.savefig('visualization/reconstruction.png', bbox_inches='tight')
        plt.close()

    def plot_simu_repre(self, min_, mean_, max_):
        # Plot simulated data in all directions of the latent space
        # ZU
        fig, axes = plt.subplots(dim_z, 11, figsize=(22, 2 * dim_z))
        plt.subplots_adjust(wspace=0, hspace=0)
        mean_latent = torch.tensor([[mean for mean in mean_[1]]], device=self.device)  # style
        for i in range(dim_z):
            arange = np.linspace(min_[0][i], max_[0][i], num=11)
            for idx, j in enumerate(arange):
                simulated_latent = torch.tensor([[mean for mean in mean_[0]]], device=self.device)
                simulated_latent[0][i] = j
                encoded = torch.cat((mean_latent, simulated_latent), dim=1)
                simulated_img = self.decoder(encoded)
                axes[i][idx].matshow(255 * simulated_img[0][0].cpu().detach().numpy())
        for axe in axes:
            for ax in axe:
                ax.set_xticks([])
                ax.set_yticks([])

        plt.savefig('visualization/simulation_latent_ZU.png', bbox_inches='tight')
        plt.close()

        # ZV
        fig, axes = plt.subplots(dim_z, 11, figsize=(22, 2 * dim_z))
        plt.subplots_adjust(wspace=0, hspace=0)
        mean_latent = torch.tensor([[mean for mean in mean_[0]]], device=self.device)  # class
        for i in range(dim_z):
            arange = np.linspace(min_[1][i], max_[1][i], num=11)
            for idx, j in enumerate(arange):
                simulated_latent = torch.tensor([[mean for mean in mean_[1]]], device=self.device)
                simulated_latent[0][i] = j
                encoded = torch.cat((simulated_latent, mean_latent), dim=1)
                simulated_img = self.decoder(encoded)
                axes[i][idx].matshow(255 * simulated_img[0][0].cpu().detach().numpy())
        for axe in axes:
            for ax in axe:
                ax.set_xticks([])
                ax.set_yticks([])

        plt.savefig('visualization/simulation_latent_ZV.png', bbox_inches='tight')
        plt.close()
        self.training = True

    def plot_grad_simu_repre(self, min_, mean_, max_):
        # Plot the gradient map of simulated data in all directions of the latent space
        # ZU, class
        fig, axes = plt.subplots(dim_z, 10, figsize=(20, 2 * dim_z))
        plt.subplots_adjust(wspace=0, hspace=0)
        mean_latent = torch.tensor([[mean for mean in mean_[1]]], device=self.device)  # style
        for i in range(dim_z):
            arange = np.linspace(min_[0][i], max_[0][i], num=11)
            for idx, j in enumerate(arange):
                simulated_latent = torch.tensor([[mean for mean in mean_[0]]], device=self.device)
                simulated_latent[0][i] = j
                encoded = torch.cat((mean_latent, simulated_latent), dim=1)
                simulated_img = self.decoder(encoded)
                if idx == 0:
                    template = simulated_img
                    continue
                grad_img = simulated_img - template
                template = simulated_img
                axes[i][idx - 1].matshow(grad_img[0][0].cpu().detach().numpy(), cmap=matplotlib.cm.get_cmap('bwr'),
                                         norm=matplotlib.colors.CenteredNorm())
        for axe in axes:
            for ax in axe:
                ax.set_xticks([])
                ax.set_yticks([])
        fig.colorbar(
            matplotlib.cm.ScalarMappable(cmap=matplotlib.cm.get_cmap('bwr'), norm=matplotlib.colors.CenteredNorm()),
            cax=fig.add_axes([0.92, 0.15, 0.01, 0.7]))
        plt.savefig('visualization/gradient_simulation_latent_ZU.png', bbox_inches='tight')
        plt.close()

        # ZV, style
        fig, axes = plt.subplots(dim_z, 10, figsize=(20, 2 * dim_z))
        plt.subplots_adjust(wspace=0, hspace=0)
        mean_latent = torch.tensor([[mean for mean in mean_[0]]], device=self.device)  # class
        for i in range(dim_z):
            arange = np.linspace(min_[1][i], max_[1][i], num=11)
            for idx, j in enumerate(arange):
                simulated_latent = torch.tensor([[mean for mean in mean_[1]]], device=self.device)
                simulated_latent[0][i] = j
                encoded = torch.cat((simulated_latent, mean_latent), dim=1)
                simulated_img = self.decoder(encoded)
                if idx == 0:
                    template = simulated_img
                    continue
                grad_img = simulated_img - template
                template = simulated_img
                axes[i][idx - 1].matshow(grad_img[0][0].cpu().detach().numpy(), cmap=matplotlib.cm.get_cmap('bwr'),
                                         norm=matplotlib.colors.CenteredNorm())
        for axe in axes:
            for ax in axe:
                ax.set_xticks([])
                ax.set_yticks([])
        fig.colorbar(
            matplotlib.cm.ScalarMappable(cmap=matplotlib.cm.get_cmap('bwr'), norm=matplotlib.colors.CenteredNorm()),
            cax=fig.add_axes([0.92, 0.15, 0.01, 0.7]))
        plt.savefig('visualization/gradient_simulation_latent_ZV.png', bbox_inches='tight')
        plt.close()
        self.training = True

    @staticmethod
    def plot_z_distribution(ZU, ZV):
        min_zu, mean_zu, max_zu = [], [], []
        fig, axes = plt.subplots(1, dim_z, figsize=(4 * dim_z, 4))
        plt.subplots_adjust(wspace=0.1, hspace=0)
        for i in range(dim_z):
            zu = ZU[:, i].cpu().detach().numpy()
            axes[i].hist(zu, bins=70, density=True)
            axes[i].set_title('{}-th dim'.format(i + 1))
            axes[i].set_xlabel(f"Min: {np.min(zu):.4}\nMean: {np.mean(zu):.4}\nMax: {np.max(zu):.4}")
            min_zu.append(np.min(zu))
            mean_zu.append(np.mean(zu))
            max_zu.append(np.max(zu))
        for axe in axes:
            axe.set_yticks([])
            # axe.set_xlim(left=-1, right=1)
        plt.savefig('visualization/ZU_distribution.png', bbox_inches='tight')
        plt.close()

        min_zv, mean_zv, max_zv = [], [], []
        fig, axes = plt.subplots(1, dim_z, figsize=(4 * dim_z, 4))
        plt.subplots_adjust(wspace=0.1, hspace=0)
        for i in range(dim_z):
            zv = ZV[:, i].cpu().detach().numpy()
            axes[i].hist(zv, bins=70, density=True)
            axes[i].set_title('{}-th dim'.format(i + 1))
            axes[i].set_xlabel(f"Min: {np.min(zv):.4}\nMean: {np.mean(zv):.4}\nMax: {np.max(zv):.4}")
            min_zv.append(np.min(zv))
            mean_zv.append(np.mean(zv))
            max_zv.append(np.max(zv))
        for axe in axes:
            axe.set_yticks([])
            # axe.set_xlim(left=-1, right=1)
        plt.savefig('visualization/ZV_distribution.png', bbox_inches='tight')
        plt.close()

        min_ = [min_zu, min_zv]
        mean_ = [mean_zu, mean_zv]
        max_ = [max_zu, max_zv]
        return min_, mean_, max_


class rank_VAE(nn.Module):
    def __init__(self):
        super(rank_VAE, self).__init__()
        nn.Module.__init__(self)
        self.name = 'rank_VAE'
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.beta = 5
        self.gamma = 10

        # zs encoder, dim=dimz-1
        self.conv1 = nn.Conv2d(1, 16, 3, stride=2, padding=1)  # 16 x 32 x 32
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)  # 32 x 16 x 16
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)  # 32 x 8 x 8
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.fc10 = nn.Linear(2048, dim_z - 1)
        self.fc11 = nn.Linear(2048, dim_z - 1)

        # zpsi encoder, dim=1
        self.conv4 = nn.Conv2d(1, 16, 3, stride=2, padding=1)  # 16 x 32 x 32
        self.conv5 = nn.Conv2d(16, 32, 3, stride=2, padding=1)  # 32 x 16 x 16
        self.conv6 = nn.Conv2d(32, 32, 3, stride=2, padding=1)  # 32 x 8 x 8
        self.bn4 = nn.BatchNorm2d(16)
        self.bn5 = nn.BatchNorm2d(32)
        self.bn6 = nn.BatchNorm2d(32)
        self.fc20 = nn.Linear(2048, 1)
        self.fc21 = nn.Linear(2048, 1)

        # decoder
        self.fc3 = nn.Linear(dim_z, 512)
        self.upconv1 = nn.ConvTranspose2d(8, 64, 3, stride=2, padding=1, output_padding=1)  # 64 x 16 x 16
        self.upconv2 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)  # 32 x 32 x 32
        self.upconv3 = nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1)  # 1 x 64 x 64
        self.bn7 = nn.BatchNorm2d(64)
        self.bn8 = nn.BatchNorm2d(32)

        self.train_recon_loss, self.test_recon_loss = [], []

    def encoder_zs(self, image):
        h1 = F.relu(self.bn1(self.conv1(image)))
        h2 = F.relu(self.bn2(self.conv2(h1)))
        h3 = F.relu(self.bn3(self.conv3(h2)))

        zs_mu = torch.tanh(self.fc10(h3.view(h3.size()[0], -1)))
        zs_logVar = self.fc11(h3.view(h3.size()[0], -1))
        return zs_mu, zs_logVar

    def encoder_zpsi(self, image):
        h1 = F.relu(self.bn4(self.conv4(image)))
        h2 = F.relu(self.bn5(self.conv5(h1)))
        h3 = F.relu(self.bn6(self.conv6(h2)))

        zpsi_mu = torch.tanh(self.fc20(h3.view(h3.size()[0], -1)))
        zpsi_logVar = self.fc21(h3.view(h3.size()[0], -1))
        return zpsi_mu, zpsi_logVar

    def decoder(self, encoded):
        h6 = F.relu(self.fc3(encoded)).reshape([encoded.size()[0], 8, 8, 8])
        h7 = F.relu(self.bn7(self.upconv1(h6)))
        h8 = F.relu(self.bn8(self.upconv2(h7)))
        reconstructed = F.relu(self.upconv3(h8))
        return reconstructed

    def reparametrize(self, mu, logVar):
        # Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar / 2).to(self.device)
        eps = torch.normal(mean=torch.tensor([0 for i in range(std.shape[1])]).float(), std=1).to(self.device)
        return mu + eps * std

    def forward(self, image):
        zs_mu, zs_logVar = self.encoder_zs(image)
        zpsi_mu, zpsi_logVar = self.encoder_zpsi(image)
        if self.training:
            zs_encoded = self.reparametrize(zs_mu, zs_logVar)
            zpsi_encoded = self.reparametrize(zpsi_mu, zpsi_logVar)
        else:
            zs_encoded = zs_mu
            zpsi_encoded = zpsi_mu
        return zs_mu, zs_logVar, zpsi_mu, zpsi_logVar, zs_encoded, zpsi_encoded

    def loss(self, mu, logVar, reconstructed, input_):
        kl_divergence = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp()) / mu.shape[0]
        recon_error = torch.sum((reconstructed - input_) ** 2) / input_.shape[0]
        return recon_error, kl_divergence

    def rank_loss(self, subject, timepoint, zpsi):
        rank_loss = 0
        num = 0.0

        subject = torch.squeeze(subject)
        timepoint = torch.squeeze(timepoint)
        unique_subject, cnt = torch.unique(subject, return_counts=True)
        select_subject = [s.squeeze() for i, s in enumerate(unique_subject) if cnt[i] >= 2]
        if len(select_subject) == 0:
            return torch.tensor(0.0, device=self.device)
        else:
            select_index = [np.squeeze(np.nonzero(subject == sub)) for sub in select_subject]
            for index in select_index:
                select_tp = timepoint[index].squeeze()
                rank_tp = torch.argsort(select_tp)
                select_zpsi = zpsi[index].squeeze()
                rand_zpsi = torch.argsort(select_zpsi)
                rank_loss += torch.sum((rand_zpsi - rank_tp) ** 2)
                num += len(index)
            return rank_loss / num

    def train_(self, data_loader, test, optimizer, num_epochs):

        self.to(self.device)
        best_loss = 1e10
        es = 0

        for epoch in range(num_epochs):

            start_time = time()
            if es == 100:
                break

            logger.info('Epoch {}/{}'.format(epoch + 1, num_epochs))

            tloss = 0.0
            nb_batches = 0

            ZU, ZV = None, None
            for data in data_loader:
                image = torch.tensor([[np.load(path)] for path in data[0]], device=self.device).float()
                subject = torch.tensor([[s for s in data[1]]], device=self.device)
                timepoint = torch.tensor([[tp for tp in data[4]]], device=self.device)
                optimizer.zero_grad()

                input_ = Variable(image).to(self.device)
                zs_mu, zs_logVar, zpsi_mu, zpsi_logVar, zs_encoded, zpsi_encoded = self.forward(input_)
                encoded = torch.cat((zs_encoded, zpsi_encoded), dim=1)
                reconstructed = self.decoder(encoded)
                reconstruction_loss, zs_kl_loss = self.loss(zs_mu, zs_logVar, input_, reconstructed)
                reconstruction_loss, zpsi_kl_loss = self.loss(zpsi_mu, zpsi_logVar, input_, reconstructed)
                rank_loss = self.rank_loss(subject, timepoint, zpsi_encoded)

                loss = reconstruction_loss + self.beta * (zs_kl_loss + zpsi_kl_loss) + self.gamma * rank_loss
                self.train_recon_loss.append(reconstruction_loss.cpu().detach().numpy())

                # store ZU, ZV
                if ZU is None:
                    ZU, ZV = zpsi_encoded, zs_encoded
                else:
                    ZU = torch.cat((ZU, zpsi_encoded), 0)
                    ZV = torch.cat((ZV, zs_encoded), 0)

                loss.backward()
                optimizer.step()
                tloss += float(loss)
                nb_batches += 1

            epoch_loss = tloss / nb_batches
            test_loss = self.evaluate(test)

            if epoch_loss <= best_loss:
                es = 0
                best_loss = epoch_loss
            else:
                es += 1
            end_time = time()

            # plot the results
            self.plot_recon(test)
            min_, mean_, max_ = self.plot_z_distribution(ZU, ZV)
            self.plot_simu_repre(min_, mean_, max_)
            self.plot_grad_simu_repre(min_, mean_, max_)

            logger.info(f"Recon / KL / Rank: {reconstruction_loss:.3}/{zs_kl_loss + zpsi_kl_loss:.3}/{rank_loss:.3}")
            logger.info(
                f"Epoch loss (train/test): {epoch_loss:.3}/{test_loss:.3} took {end_time - start_time:.1} seconds")

        print('Complete training')
        return

    def evaluate(self, data):
        self.to(self.device)
        self.training = False
        self.eval()
        dataloader = torch.utils.data.DataLoader(data, batch_size=128, num_workers=0, shuffle=False, drop_last=False)
        tloss = 0.0
        nb_batches = 0

        with torch.no_grad():
            for data in dataloader:
                image = torch.tensor([[np.load(path)] for path in data[0]]).float()
                subject = torch.tensor([[s for s in data[1]]], device=self.device)
                timepoint = torch.tensor([[tp for tp in data[4]]], device=self.device)

                input_ = Variable(image).to(self.device)
                zs_mu, zs_logVar, zpsi_mu, zpsi_logVar, zs_encoded, zpsi_encoded = self.forward(input_)
                encoded = torch.cat((zs_encoded, zpsi_encoded), dim=1)
                reconstructed = self.decoder(encoded)
                reconstruction_loss, zs_kl_loss = self.loss(zs_mu, zs_logVar, input_, reconstructed)
                reconstruction_loss, zpsi_kl_loss = self.loss(zpsi_mu, zpsi_logVar, input_, reconstructed)
                rank_loss = self.rank_loss(subject, timepoint, zpsi_encoded)

                loss = reconstruction_loss + self.beta * (zs_kl_loss + zpsi_kl_loss) + self.gamma * rank_loss
                self.test_recon_loss.append(reconstruction_loss.cpu().detach().numpy())

                tloss += float(loss)
                nb_batches += 1

        loss = tloss / nb_batches
        self.training = True
        return loss

    def plot_recon(self, data, n_subject=3):
        # Plot the reconstruction
        fig, axes = plt.subplots(2 * n_subject, 10, figsize=(20, 4 * n_subject))
        plt.subplots_adjust(wspace=0, hspace=0)
        for j in range(n_subject):
            for i in range(10):
                test_image = torch.tensor(np.load(data[j * 10 + i][0])).resize(1, 1, 64, 64).float()
                test_image = Variable(test_image).to(self.device)
                _, _, _, _, style_encoded, class_encoded = self.forward(test_image)
                encoded = torch.cat((style_encoded, class_encoded), dim=1)
                out = self.decoder(encoded)
                axes[2 * j][i].matshow(255 * test_image[0][0].cpu().detach().numpy())
                axes[2 * j + 1][i].matshow(255 * out[0][0].cpu().detach().numpy())
        for axe in axes:
            for ax in axe:
                ax.set_xticks([])
                ax.set_yticks([])
        plt.savefig('visualization/reconstruction.png', bbox_inches='tight')
        plt.close()

    def plot_simu_repre(self, min_, mean_, max_):
        # Plot simulated data in all directions of the latent space
        # ZU
        fig, axes = plt.subplots(1, 11, figsize=(22, 2))
        plt.subplots_adjust(wspace=0, hspace=0)
        mean_latent = torch.tensor([[mean for mean in mean_[1]]], device=self.device)  # zs
        arange = np.linspace(min_[0][0], max_[0][0], num=11)
        for idx, j in enumerate(arange):
            simulated_latent = torch.tensor([[mean for mean in mean_[0]]], device=self.device)
            simulated_latent[0][0] = j
            encoded = torch.cat((mean_latent, simulated_latent), dim=1)
            simulated_img = self.decoder(encoded)
            axes[idx].matshow(255 * simulated_img[0][0].cpu().detach().numpy())
        for axe in axes:
            axe.set_xticks([])
            axe.set_yticks([])

        plt.savefig('visualization/simulation_latent_ZU.png', bbox_inches='tight')
        plt.close()

        # ZV
        fig, axes = plt.subplots(dim_z - 1, 11, figsize=(22, 2 * (dim_z - 1)))
        plt.subplots_adjust(wspace=0, hspace=0)
        mean_latent = torch.tensor([[mean for mean in mean_[0]]], device=self.device)  # zpsi
        for i in range(dim_z - 1):
            arange = np.linspace(min_[1][i], max_[1][i], num=11)
            for idx, j in enumerate(arange):
                simulated_latent = torch.tensor([[mean for mean in mean_[1]]], device=self.device)
                simulated_latent[0][i] = j
                encoded = torch.cat((simulated_latent, mean_latent), dim=1)
                simulated_img = self.decoder(encoded)
                axes[i][idx].matshow(255 * simulated_img[0][0].cpu().detach().numpy())
        for axe in axes:
            for ax in axe:
                ax.set_xticks([])
                ax.set_yticks([])

        plt.savefig('visualization/simulation_latent_ZV.png', bbox_inches='tight')
        plt.close()
        self.training = True

    def plot_grad_simu_repre(self, min_, mean_, max_):
        # Plot the gradient map of simulated data in all directions of the latent space
        # ZU, class
        fig, axes = plt.subplots(1, 10, figsize=(20, 2))
        plt.subplots_adjust(wspace=0, hspace=0)
        mean_latent = torch.tensor([[mean for mean in mean_[1]]], device=self.device)  # zs
        arange = np.linspace(min_[0][0], max_[0][0], num=11)
        for idx, j in enumerate(arange):
            simulated_latent = torch.tensor([[mean for mean in mean_[0]]], device=self.device)
            simulated_latent[0][0] = j
            encoded = torch.cat((mean_latent, simulated_latent), dim=1)
            simulated_img = self.decoder(encoded)
            if idx == 0:
                template = simulated_img
                continue
            grad_img = simulated_img - template
            template = simulated_img
            axes[idx - 1].matshow(grad_img[0][0].cpu().detach().numpy(), cmap=matplotlib.cm.get_cmap('bwr'),
                                  norm=matplotlib.colors.CenteredNorm())
        for axe in axes:
            axe.set_xticks([])
            axe.set_yticks([])
        fig.colorbar(
            matplotlib.cm.ScalarMappable(cmap=matplotlib.cm.get_cmap('bwr'), norm=matplotlib.colors.CenteredNorm()),
            cax=fig.add_axes([0.92, 0.15, 0.01, 0.7]))
        plt.savefig('visualization/gradient_simulation_latent_ZU.png', bbox_inches='tight')
        plt.close()

        # ZV, style
        fig, axes = plt.subplots(dim_z - 1, 10, figsize=(20, 2 * (dim_z - 1)))
        plt.subplots_adjust(wspace=0, hspace=0)
        mean_latent = torch.tensor([[mean for mean in mean_[0]]], device=self.device)  # zpsi
        for i in range(dim_z - 1):
            arange = np.linspace(min_[1][i], max_[1][i], num=11)
            for idx, j in enumerate(arange):
                simulated_latent = torch.tensor([[mean for mean in mean_[1]]], device=self.device)
                simulated_latent[0][i] = j
                encoded = torch.cat((simulated_latent, mean_latent), dim=1)
                simulated_img = self.decoder(encoded)
                if idx == 0:
                    template = simulated_img
                    continue
                grad_img = simulated_img - template
                template = simulated_img
                axes[i][idx - 1].matshow(grad_img[0][0].cpu().detach().numpy(), cmap=matplotlib.cm.get_cmap('bwr'),
                                         norm=matplotlib.colors.CenteredNorm())
        for axe in axes:
            for ax in axe:
                ax.set_xticks([])
                ax.set_yticks([])
        fig.colorbar(
            matplotlib.cm.ScalarMappable(cmap=matplotlib.cm.get_cmap('bwr'), norm=matplotlib.colors.CenteredNorm()),
            cax=fig.add_axes([0.92, 0.15, 0.01, 0.7]))
        plt.savefig('visualization/gradient_simulation_latent_ZV.png', bbox_inches='tight')
        plt.close()
        self.training = True

    @staticmethod
    def plot_z_distribution(ZU, ZV):
        min_zu, mean_zu, max_zu = [], [], []
        fig, axes = plt.subplots(1, 1, figsize=(4, 4))
        zu = ZU.cpu().detach().numpy()
        axes.hist(zu, bins=70, density=True)
        axes.set_title('{}-th dim'.format(1))
        axes.set_xlabel(f"Min: {np.min(zu):.4}\nMean: {np.mean(zu):.4}\nMax: {np.max(zu):.4}")
        min_zu.append(np.min(zu))
        mean_zu.append(np.mean(zu))
        max_zu.append(np.max(zu))
        axes.set_yticks([])
        # axe.set_xlim(left=-1, right=1)
        plt.savefig('visualization/ZU_distribution.png', bbox_inches='tight')
        plt.close()

        min_zv, mean_zv, max_zv = [], [], []
        fig, axes = plt.subplots(1, dim_z - 1, figsize=(4 * (dim_z - 1), 4))
        plt.subplots_adjust(wspace=0.1, hspace=0)
        for i in range(dim_z - 1):
            zv = ZV[:, i].cpu().detach().numpy()
            axes[i].hist(zv, bins=70, density=True)
            axes[i].set_title('{}-th dim'.format(i + 1))
            axes[i].set_xlabel(f"Min: {np.min(zv):.4}\nMean: {np.mean(zv):.4}\nMax: {np.max(zv):.4}")
            min_zv.append(np.min(zv))
            mean_zv.append(np.mean(zv))
            max_zv.append(np.max(zv))
        for axe in axes:
            axe.set_yticks([])
            # axe.set_xlim(left=-1, right=1)
        plt.savefig('visualization/ZV_distribution.png', bbox_inches='tight')
        plt.close()

        min_ = [min_zu, min_zv]
        mean_ = [mean_zu, mean_zv]
        max_ = [max_zu, max_zv]
        return min_, mean_, max_


class LNE(nn.Module):
    def __init__(self):
        super(LNE, self).__init__()
        nn.Module.__init__(self)
        self.name = 'LNE'
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = 128
        self.lambda_proto = 1.0
        self.lambda_dir = 1.0

        self.conv1 = nn.Conv2d(1, 16, 3, stride=2, padding=1)  # 16 x 32 x 32
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)  # 32 x 16 x 16
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)  # 32 x 8 x 8
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.fc10 = nn.Linear(2048, 1024)

        self.upconv1 = nn.ConvTranspose2d(16, 64, 3, stride=2, padding=1, output_padding=1)  # 64 x 16 x 16
        self.upconv2 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)  # 32 x 32 x 32
        self.upconv3 = nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1)  # 1 x 64 x 64
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(32)

        self.train_loss, self.test_loss = [], []

        self.N_km = [I // 5, I // 10, I // 20]
        self.num_nb = 5
        self.sample_idx_list = None
        self.concentration_list = None
        self.prototype_list = None

    def encoder(self, image):
        h1 = F.relu(self.bn1(self.conv1(image)))
        h2 = F.relu(self.bn2(self.conv2(h1)))
        h3 = F.relu(self.bn3(self.conv3(h2)))
        z = self.fc10(h3.view(h3.size()[0], -1))
        return z

    def decoder(self, encoded):
        h6 = encoded.reshape([encoded.size()[0], 16, 8, 8])
        h7 = F.relu(self.bn4(self.upconv1(h6)))
        h8 = F.relu(self.bn5(self.upconv2(h7)))
        reconstructed = F.relu(self.upconv3(h8))
        return reconstructed

    def forward(self, img1, img2=None):
        bs = img1.shape[0]
        if img2 is None:
            zs = self.encoder(img1)
            recons = self.decoder(zs)
            return zs, recons
        else:
            zs = self.encoder(torch.cat([img1, img2], 0))
            recons = self.decoder(zs)
            zs_flatten = zs.view(bs * 2, -1)
            z1, z2 = zs_flatten[:bs], zs_flatten[bs:]
            recon1, recon2 = recons[:bs], recons[bs:]
            return [z1, z2], [recon1, recon2]

    def build_graph_batch(self, zs):
        z1 = zs[0]
        bs = z1.shape[0]
        dis_mx = torch.zeros(bs, bs).to(self.device)
        for i in range(bs):
            for j in range(i + 1, bs):
                dis_mx[i, j] = torch.sum((z1[i] - z1[j]) ** 2)
                dis_mx[j, i] = dis_mx[i, j]
        sigma = (torch.sort(dis_mx)[0][:, -1]) ** 0.5 - (torch.sort(dis_mx)[0][:, 1]) ** 0.5
        adj_mx = torch.exp(-dis_mx / (2 * sigma ** 2))
        if self.num_nb < bs:
            adj_mx_filter = torch.zeros(bs, bs).to(self.device)
            for i in range(bs):
                ks = torch.argsort(dis_mx[i], descending=False)[:self.num_nb + 1]
                adj_mx_filter[i, ks] = adj_mx[i, ks]
                adj_mx_filter[i, i] = 0.
            return adj_mx_filter
        else:
            return adj_mx * (1. - torch.eye(bs, bs).to(self.device))

    def build_graph_dataset(self, zs_all, zs):
        z1_all = zs_all[0]
        z1 = zs[0]
        ds = z1_all.shape[0]
        bs = z1.shape[0]
        dis_mx = torch.zeros(bs, ds).to(self.device)
        for i in range(bs):
            for j in range(ds):
                dis_mx[i, j] = torch.sum((z1[i] - z1_all[j]) ** 2)
        # sigma = (torch.sort(dis_mx)[0][:, -1])**0.5 - (torch.sort(dis_mx)[0][:, 1])**0.5
        adj_mx = torch.exp(-dis_mx / 100)
        # adj_mx = torch.exp(-dis_mx / (2*sigma**2))
        if self.num_nb < bs:
            adj_mx_filter = torch.zeros(bs, ds).to(self.device)
            for i in range(bs):
                ks = torch.argsort(dis_mx[i], descending=False)[:self.num_neighbours + 1]
                adj_mx_filter[i, ks] = adj_mx[i, ks]
            return adj_mx_filter
        else:
            return adj_mx * (1. - torch.eye(bs, bs).to(self.device))

    @staticmethod
    def compute_social_pooling_delta_z_batch(zs, interval, adj_mx):
        z1, z2 = zs[0], zs[1]
        delta_z = (z2 - z1) / interval.unsqueeze(1)  # [bs, ls]
        delta_h = torch.matmul(adj_mx, delta_z) / adj_mx.sum(1, keepdim=True)  # [bs, ls]
        return delta_z, delta_h

    @staticmethod
    def compute_social_pooling_delta_z_dataset(zs_all, interval_all, zs, interval, adj_mx):
        z1, z2 = zs[0], zs[1]
        delta_z = (z2 - z1) / interval.unsqueeze(1)  # [bs, ls]
        z1_all, z2_all = zs_all[0], zs_all[1]
        delta_z_all = (z2_all - z1_all) / interval_all.unsqueeze(1)  # [bs, ls]
        delta_h = torch.matmul(adj_mx, delta_z_all) / adj_mx.sum(1, keepdim=True)  # [bs, ls]
        return delta_z, delta_h

    def minimatch_sampling_strategy(self, cluster_centers_list, cluster_ids_list):
        # compute distance between clusters
        cluster_dis_ids_list = []

        for m in range(len(cluster_centers_list)):
            cluster_centers = cluster_centers_list[m]
            n_km = cluster_centers.shape[0]
            cluster_dis_ids = np.zeros((n_km, n_km))
            for i in range(n_km):
                dis_cn = np.sqrt(np.sum((cluster_centers[i].reshape(1, -1) - cluster_centers) ** 2, 1))
                cluster_dis_ids[i] = np.argsort(dis_cn)
            cluster_dis_ids_list.append(cluster_dis_ids)

        n_batch = np.ceil(I / self.batch_size).astype(int)
        sample_idx_list = []
        for nb in range(n_batch):
            m_idx = np.random.choice(len(cluster_centers_list))  # select round of kmeans
            c_idx = np.random.choice(cluster_centers_list[m_idx].shape[0])  # select a cluster
            sample_idx_batch = []
            n_s_b = 0
            for c_idx_sel in cluster_dis_ids_list[m_idx][
                c_idx]:  # get nbr clusters given distance to selected cluster c_idx
                sample_idx = np.where(cluster_ids_list[m_idx] == c_idx_sel)[0]
                if n_s_b + sample_idx.shape[0] >= self.batch_size:
                    sample_idx_batch.append(np.random.choice(sample_idx, self.batch_size - n_s_b, replace=False))
                    break
                else:
                    sample_idx_batch.append(sample_idx)
                    n_s_b += sample_idx.shape[0]

            sample_idx_batch = np.concatenate(sample_idx_batch, 0)
            sample_idx_list.append(sample_idx_batch)

        sample_idx_list = np.concatenate(sample_idx_list, 0)
        self.sample_idx_list = sample_idx_list[:I]

    @staticmethod
    def compute_recon_loss(x, recon):
        return torch.mean((recon - x) ** 2)

    @staticmethod
    def compute_direction_loss(delta_z, delta_h):
        delta_z_norm = torch.norm(delta_z, dim=1) + 1e-12
        delta_h_norm = torch.norm(delta_h, dim=1) + 1e-12
        cos = torch.sum(delta_z * delta_h, 1) / (delta_z_norm * delta_h_norm)
        return (1. - cos).mean()

    def update_kmeans(self, z1_list, cluster_ids_list, cluster_centers_list):
        z1_list = torch.tensor(z1_list).to(self.device)
        self.prototype_list = [torch.tensor(c).to(self.device) for c in cluster_centers_list]
        self.concentration_list = []
        for m in range(len(self.N_km)):  # for each round of kmeans
            prototypes = self.prototype_list[m]
            cluster_ids = cluster_ids_list[m]
            concentration_m = []
            for c in range(self.N_km[m]):  # for each cluster center
                zs = z1_list[cluster_ids == c]
                n_c = zs.shape[0]
                norm = torch.norm(zs - prototypes[c].view(1, -1), dim=1).sum()
                concentration = norm / (n_c * math.log(n_c + 10))
                concentration_m.append(concentration)
            self.concentration_list.append(torch.tensor(concentration_m).to(self.device))

    def compute_prototype_NCE(self, z1, cluster_ids):
        loss = 0
        for m in range(len(self.N_km)):  # for each round of kmeans
            prototypes_sel = self.prototype_list[m][cluster_ids[m]]
            concentration_sel = self.concentration_list[m][cluster_ids[m]]
            nominator = torch.sum(z1 * prototypes_sel / concentration_sel.view(-1, 1), 1)
            denominator = torch.logsumexp(torch.matmul(z1, torch.transpose(self.prototype_list[m], 0, 1)) /
                                          self.concentration_list[m].view(1, self.N_km[m]), dim=1)
            loss += -(nominator - denominator).mean()
        return loss / (len(self.N_km) * z1.shape[0])

    def train_(self, data_loader, test, optimizer, num_epochs):
        self.to(self.device)
        best_loss = 1e10
        es = 0

        for epoch in range(num_epochs):

            start_time = time()
            if es == 100:
                break

            logger.info('#### Epoch {}/{} ####'.format(epoch + 1, num_epochs))

            tloss = 0.0
            nb_batches = 0

            # k-means for z1
            with torch.no_grad():
                self.eval()
                z1_list = []
                for data in data_loader:
                    image = torch.tensor([[np.load(path)] for path in data[0]], device=self.device).float()
                    z1 = self.encoder(image)
                    z1_list.append(z1.view(image.shape[0], -1))
                z1_list = torch.cat(z1_list).detach().cpu().numpy()
                print('Finished computing z1 for all training samples!')

                cluster_ids_list = []
                cluster_centers_list = []
                for n_km in self.N_km:
                    kmeans = KMeans(n_clusters=n_km, n_init="auto").fit(z1_list)
                    cluster_centers = kmeans.cluster_centers_
                    cluster_ids = kmeans.labels_
                    cluster_ids_list.append(cluster_ids)
                    cluster_centers_list.append(cluster_centers)
                print('Finished K-means clustering')

            self.update_kmeans(z1_list, cluster_ids_list, cluster_centers_list)
            self.minimatch_sampling_strategy(cluster_centers_list, cluster_ids_list)

            # training
            for iter, data in tqdm(enumerate(data_loader)):
                optimizer.zero_grad()
                image = torch.tensor([[np.load(path)] for path in data[0]], device=self.device).float()
                age = torch.tensor([[a for a in data[3]]], device=self.device).float().squeeze()
                bs = image.size()[0]
                idx1 = torch.arange(0, bs - 1)
                idx2 = idx1 + 1

                img1 = image[idx1]
                img2 = image[idx2]
                interval = age[idx2] - age[idx1]
                cluster_ids = [cluster_ids_list[m][iter * self.batch_size:(iter + 1) * self.batch_size] for m in
                               range(len(self.N_km))]
                cluster_ids = [c[:-1] for c in cluster_ids]

                zs, recons = self.forward(img1, img2)
                adj_mx = self.build_graph_batch(zs)
                delta_z, delta_h = self.compute_social_pooling_delta_z_batch(zs, interval, adj_mx)

                loss_recon = 0.5 * (self.compute_recon_loss(img1, recons[0]) + self.compute_recon_loss(img2, recons[1]))
                loss_dir = self.compute_direction_loss(delta_z, delta_h)
                loss_proto = self.compute_prototype_NCE(zs[0], cluster_ids)

                loss = loss_recon + self.lambda_dir * loss_dir + self.lambda_proto * loss_proto
                loss.backward()

                optimizer.step()
                tloss += float(loss)
                nb_batches += 1

            epoch_loss = tloss / nb_batches
            test_loss = self.evaluate(test)
            self.train_loss.append(epoch_loss)
            self.test_loss.append(test_loss)

            if epoch_loss <= best_loss:
                es = 0
                best_loss = epoch_loss
            else:
                es += 1

            end_time = time()
            logger.info(f"Recon / Dir / Proto: {loss_recon:.3}/{loss_dir:.3}/{loss_proto:.3}")
            logger.info(
                f"Epoch loss (train/test): {epoch_loss:.4}/{test_loss:.4} take {end_time - start_time:.3} seconds\n")

    def evaluate(self, data):
        self.to(self.device)
        self.training = False
        self.eval()
        data_loader = torch.utils.data.DataLoader(data, batch_size=self.batch_size, num_workers=0, shuffle=False,
                                                  drop_last=False)
        tloss = 0.0
        nb_batches = 0

        with torch.no_grad():
            z1_list = []
            for data in data_loader:
                image = torch.tensor([[np.load(path)] for path in data[0]], device=self.device).float()
                z1 = self.encoder(image)
                z1_list.append(z1.view(image.shape[0], -1))
            z1_list = torch.cat(z1_list).detach().cpu().numpy()

            cluster_ids_list = []
            cluster_centers_list = []
            for n_km in self.N_km:
                kmeans = KMeans(n_clusters=n_km, n_init="auto").fit(z1_list)
                cluster_centers = kmeans.cluster_centers_
                cluster_ids = kmeans.labels_
                cluster_ids_list.append(cluster_ids)
                cluster_centers_list.append(cluster_centers)

            self.update_kmeans(z1_list, cluster_ids_list, cluster_centers_list)
            self.minimatch_sampling_strategy(cluster_centers_list, cluster_ids_list)

            # training
            for iter, data in enumerate(data_loader):
                image = torch.tensor([[np.load(path)] for path in data[0]], device=self.device).float()
                age = torch.tensor([[a for a in data[3]]], device=self.device).float().squeeze()
                bs = image.size()[0]
                idx1 = torch.arange(0, bs - 1)
                idx2 = idx1 + 1

                img1 = image[idx1]
                img2 = image[idx2]
                interval = age[idx2] - age[idx1]
                cluster_ids = [cluster_ids_list[m][iter * self.batch_size:(iter + 1) * self.batch_size] for m in
                               range(len(self.N_km))]
                cluster_ids = [c[:-1] for c in cluster_ids]

                zs, recons = self.forward(img1, img2)
                adj_mx = self.build_graph_batch(zs)
                delta_z, delta_h = self.compute_social_pooling_delta_z_batch(zs, interval, adj_mx)

                loss_recon = 0.5 * (self.compute_recon_loss(img1, recons[0]) + self.compute_recon_loss(img2, recons[1]))
                loss_dir = self.compute_direction_loss(delta_z, delta_h)
                loss_proto = self.compute_prototype_NCE(zs[0], cluster_ids)

                loss = loss_recon + self.lambda_dir * loss_dir + self.lambda_proto * loss_proto

                tloss += float(loss)
                nb_batches += 1

        loss = tloss / nb_batches
        self.training = True
        return loss


class Riem_VAE(nn.Module):
    def __init__(self):
        super(Riem_VAE, self).__init__()
        nn.Module.__init__(self)
        self.name = 'Riem_VAE'
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.beta = 5.
        self.gamma = 0.1

        self.conv1 = nn.Conv2d(1, 16, 3, stride=2, padding=1)  # 16 x 32 x 32
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)  # 32 x 16 x 16
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)  # 32 x 8 x 8
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.fc10 = nn.Linear(2048, dim_z)
        self.fc11 = nn.Linear(2048, dim_z)

        self.fc3 = nn.Linear(dim_z, 512)
        self.upconv1 = nn.ConvTranspose2d(8, 64, 3, stride=2, padding=1, output_padding=1)  # 64 x 16 x 16
        self.upconv2 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)  # 32 x 32 x 32
        self.upconv3 = nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1)  # 1 x 64 x 64
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(32)

        self.X, self.Y = None, None
        self.omega = torch.normal(mean=0., std=1., size=[N, dim_z], device=self.device).float()
        self.sigma_2 = 0.5

        self.train_recon_loss, self.test_recon_loss = [], []

    def encoder(self, image):
        h1 = F.relu(self.bn1(self.conv1(image)))
        h2 = F.relu(self.bn2(self.conv2(h1)))
        h3 = F.relu(self.bn3(self.conv3(h2)))
        mu = torch.tanh(self.fc10(h3.view(h3.size()[0], -1)))
        logVar = self.fc11(h3.view(h3.size()[0], -1))
        return mu, logVar

    def decoder(self, encoded):
        h6 = F.relu(self.fc3(encoded)).reshape([encoded.size()[0], 8, 8, 8])
        h7 = F.relu(self.bn4(self.upconv1(h6)))
        h8 = F.relu(self.bn5(self.upconv2(h7)))
        reconstructed = F.relu(self.upconv3(h8))
        return reconstructed

    def reparametrize(self, mu, logVar):
        # Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar / 2).to(self.device)
        eps = torch.normal(mean=torch.tensor([0 for i in range(std.shape[1])]).float(), std=1).to(self.device)
        if self.beta != 0:  # beta VAE
            return mu + eps * std
        else:  # regular AE
            return mu

    def forward(self, image):
        mu, logVar = self.encoder(image)
        if self.training:
            encoded = self.reparametrize(mu, logVar)
        else:
            encoded = mu
        reconstructed = self.decoder(encoded)
        return mu, logVar, reconstructed

    def loss(self, mu, logVar, reconstructed, input_):
        kl_divergence = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp()) / mu.shape[0]
        recon_error = torch.sum((reconstructed - input_) ** 2) / input_.shape[0]
        return recon_error, kl_divergence

    def longitudinal_model(self, data):
        with torch.no_grad():
            subject = torch.tensor([[s for s in data[1]]], device=self.device)
            tp = torch.tensor([[tp for tp in data[4]]], device=self.device)
            idx = (subject * 10 + tp).cpu().detach().numpy().squeeze()
            alpha = torch.tensor([[a.exp() for a in data[6]]], device=self.device).float()
            delta = torch.tensor([[a - ba for a, ba in zip(data[3], data[2])]], device=self.device)
            fixed = torch.mul(alpha, delta).squeeze()
            omega = self.omega[idx]
            for i, f in enumerate(fixed):
                omega[i][0] = f
            return omega

    def train_(self, data_loader, test, optimizer, num_epochs):
        self.to(self.device)
        best_loss = 1e10
        es = 0

        for epoch in range(num_epochs):

            start_time = time()
            if es == 100:
                break

            logger.info('Epoch {}/{}'.format(epoch + 1, num_epochs))

            tloss = 0.0
            nb_batches = 0

            s_tp, Z, alpha = None, None, None
            for data in data_loader:
                optimizer.zero_grad()
                image = torch.tensor([[np.load(path)] for path in data[0]], device=self.device).float()
                input_ = Variable(image).to(self.device)

                z, logVar, reconstructed = self.forward(input_)
                reconstruction_loss, kl_loss = self.loss(z, logVar, input_, reconstructed)
                longitudinal = self.longitudinal_model(data)
                alignment_loss = torch.sum((longitudinal - z) ** 2) / z.shape[0]
                loss = reconstruction_loss + self.beta * kl_loss + self.gamma * alignment_loss
                self.train_recon_loss.append(reconstruction_loss.cpu().detach().numpy())

                # store Z, ZU, ZV
                subject = torch.tensor([[s for s in data[1]]], device=self.device)
                tp = torch.tensor([[tp for tp in data[4]]], device=self.device)
                st = torch.transpose(torch.cat((subject, tp), 0), 0, 1)
                a = torch.transpose(torch.tensor([[a.exp() for a in data[6]]]).to(self.device).float(), 0, 1)
                if s_tp is None:
                    s_tp, Z, alpha = st, z, a
                else:
                    s_tp = torch.cat((s_tp, st), 0)
                    Z = torch.cat((Z, z), 0)
                    alpha = torch.cat((alpha, a), 0)

                loss.backward()
                optimizer.step()
                tloss += float(loss)
                nb_batches += 1

            # comply with generative model
            sort_index1 = s_tp[:, 1].sort()[1]
            sorted_s_tp = s_tp[sort_index1]
            sort_index2 = sorted_s_tp[:, 0].sort()[1]
            Z, alpha = Z[sort_index1], alpha[sort_index1]
            Z, alpha = Z[sort_index2], alpha[sort_index2]
            self.update_omega(Z, alpha)

            epoch_loss = tloss / nb_batches
            test_loss = self.evaluate(test)

            if epoch_loss <= best_loss:
                es = 0
                best_loss = epoch_loss
            else:
                es += 1
            end_time = time()
            self.plot_recon(test)
            logger.info(
                f"Recon / KL loss / Align: {reconstruction_loss:.3}/{kl_loss:.3}/{alignment_loss:.3}")
            logger.info(
                f"Epoch loss (train/test): {epoch_loss:.3}/{test_loss:.3} took {end_time - start_time:.2} seconds")

        print('Complete training')
        return

    def evaluate(self, data):
        self.to(self.device)
        self.training = False
        self.eval()
        dataloader = torch.utils.data.DataLoader(data, batch_size=128, num_workers=0, shuffle=False, drop_last=False)
        tloss = 0.0
        nb_batches = 0

        with torch.no_grad():
            for data in dataloader:
                image = torch.tensor([[np.load(path)] for path in data[0]], device=self.device).float()
                input_ = Variable(image).to(self.device)

                z, logVar, reconstructed = self.forward(input_)
                reconstruction_loss, kl_loss = self.loss(z, logVar, input_, reconstructed)
                # longitudinal = self.longitudinal_model(data)
                # alignment_loss = torch.sum((longitudinal - z) ** 2) / z.shape[0]
                loss = reconstruction_loss + self.beta * kl_loss
                self.test_recon_loss.append(reconstruction_loss.cpu().detach().numpy())

                tloss += float(loss)
                nb_batches += 1

        loss = tloss / nb_batches
        self.training = True
        return loss

    def update_omega(self, Z, alpha):
        delta = torch.tensor(self.X[:, 1]).to(self.device).float().view(alpha.size())
        fixed = torch.mul(delta, alpha)
        fixed = torch.cat((fixed, torch.zeros([N, dim_z - 1]).to(self.device).float()), dim=1)
        for i in range(5):
            self.omega = 1 / (1 - self.sigma_2) * (Z - fixed)
            self.sigma_2 = 1 / (N * dim_z) * torch.pow(torch.norm(Z - fixed - self.omega, p='fro'), 2)

    def plot_recon(self, data, n_subject=3):
        # Plot the reconstruction
        fig, axes = plt.subplots(2 * n_subject, 10, figsize=(20, 4 * n_subject))
        plt.subplots_adjust(wspace=0, hspace=0)
        for j in range(n_subject):
            for i in range(10):
                test_image = torch.tensor(np.load(data[j * 10 + i][0])).resize(1, 1, 64, 64).float()
                test_image = Variable(test_image).to(self.device)
                _, _, out = self.forward(test_image)
                axes[2 * j][i].matshow(255 * test_image[0][0].cpu().detach().numpy())
                axes[2 * j + 1][i].matshow(255 * out[0][0].cpu().detach().numpy())
        for axe in axes:
            for ax in axe:
                ax.set_xticks([])
                ax.set_yticks([])
        plt.savefig('visualization/reconstruction.png', bbox_inches='tight')
        plt.close()