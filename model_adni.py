import torch
import torch.nn as nn
from torch.autograd import Variable
from time import time
import numpy as np
from tqdm import tqdm
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

dim_z = 16


class AE_adni(nn.Module):
    def __init__(self, input_dim, left_right=0):
        super(AE_adni, self).__init__()
        nn.Module.__init__(self)
        self.name = 'AE_adni'
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.gamma = 1
        self.lam = 1
        self.input_dim = input_dim
        self.left_right = left_right

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, dim_z),
            nn.Tanh(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(dim_z, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, self.input_dim),
            nn.ReLU()
        )

        self.X, self.Y = None, None
        self.beta, self.b, self.D = None, None, None
        self.U, self.V = None, None
        self.sigma0_2, self.sigma1_2, self.sigma2_2 = None, None, None

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

    def _init_mixed_effect_model(self):
        self.beta = torch.rand(size=[self.X.size()[1], dim_z], device=self.device)
        self.b = torch.normal(mean=0, std=1, size=[self.Y.size()[1], dim_z], device=self.device)
        self.U = torch.diag(torch.tensor([1 for i in range(dim_z // 2)] + [0 for i in range(dim_z - dim_z // 2)], device=self.device)).float()
        self.V = torch.eye(dim_z, device=self.device) - self.U
        self.sigma0_2, self.sigma1_2, self.sigma2_2 = 0.25, 0.5, 0.25
        self.D = torch.eye(self.Y.size()[1], device=self.device).float()

    @staticmethod
    def loss(input_, reconstructed):
        # recon_loss = torch.sum((reconstructed - input_) ** 2) / input_.shape[0]
        recon_loss = torch.mean((reconstructed - input_) ** 2)
        return recon_loss

    def train_(self, train_data_loader, test_data_loader, optimizer, num_epochs):
        self.to(self.device)
        self._init_mixed_effect_model()
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
            for data in tqdm(train_data_loader):
                # data: 0 lthick, 1 rthick, 2 age, 3 baseline_age, 4 label, 5 subject, 6 timepoint
                image = data[self.left_right]
                optimizer.zero_grad()

                # self-reconstruction loss
                input_ = Variable(image).to(self.device).float()
                reconstructed, z, zu, zv = self.forward(input_)
                self_reconstruction_loss = self.loss(input_, reconstructed)

                # store Z, ZU, ZV
                subject = torch.tensor([[s for s in data[5]]], device=self.device)
                tp = torch.tensor([[tp for tp in data[6]]], device=self.device)
                st = torch.transpose(torch.cat((subject, tp), 0), 0, 1)
                if s_tp is None:
                    s_tp, Z, ZU, ZV = st, z, zu, zv
                else:
                    s_tp = torch.cat((s_tp, st), 0)
                    Z = torch.cat((Z, z), 0)
                    ZU = torch.cat((ZU, zu), 0)
                    ZV = torch.cat((ZV, zv), 0)

                # cross-reconstruction loss
                baseline_age = data[3]
                delta_age = data[2] - baseline_age
                index0, index1 = self.generate_sample(baseline_age, delta_age)
                image0 = image[index0]
                image1 = image[index1]
                if index0:
                    input0_ = Variable(image0).to(self.device).float()
                    input1_ = Variable(image1).to(self.device).float()
                    reconstructed = self.forward(input0_, input1_)
                    cross_reconstruction_loss = self.loss(input0_, reconstructed)
                    recon_loss = (self_reconstruction_loss + cross_reconstruction_loss) / 2
                else:
                    recon_loss = self_reconstruction_loss

                loss = recon_loss
                loss.backward()
                optimizer.step()
                tloss += float(loss)
                nb_batches += 1

            # comply with generative model
            # sort_index1 = s_tp[:, 1].sort()[1]
            # sorted_s_tp = s_tp[sort_index1]
            # sort_index2 = sorted_s_tp[:, 0].sort()[1]
            # Z, ZU, ZV = Z[sort_index1], ZU[sort_index1], ZV[sort_index1]
            # Z, ZU, ZV = Z[sort_index2], ZU[sort_index2], ZV[sort_index2]
            self.generative_parameter_update(Z, ZU, ZV)

            epoch_loss = tloss / nb_batches
            test_loss = self.evaluate(test_data_loader)

            if epoch_loss <= best_loss:
                es = 0
                best_loss = epoch_loss
            else:
                es += 1

            end_time = time()
            logger.info(f"Epoch loss (train/test): {epoch_loss:.4}/{test_loss:.4} take {end_time - start_time:.3} seconds\n")

    def evaluate(self, test_data_loader):
        self.to(self.device)
        self.training = False
        self.eval()
        tloss = 0.0
        nb_batches = 0

        with torch.no_grad():
            for data in test_data_loader:
                # data: 0 lthick, 1 rthick, 2 age, 3 baseline_age, 4 label, 5 subject, 6 timepoint
                image = data[self.left_right]

                # self-reconstruction loss
                input_ = Variable(image).to(self.device).float()
                reconstructed, z, zu, zv = self.forward(input_)
                self_reconstruction_loss = self.loss(input_, reconstructed)

                # cross-reconstruction loss
                baseline_age = data[3]
                delta_age = data[2] - baseline_age
                index0, index1 = self.generate_sample(baseline_age, delta_age)
                image0 = image[index0]
                image1 = image[index1]
                if index0:
                    input0_ = Variable(image0).to(self.device).float()
                    input1_ = Variable(image1).to(self.device).float()
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

    def generative_parameter_update(self, Z, ZU, ZV):
        X = Variable(self.X).to(self.device).float()
        Y = Variable(self.Y).to(self.device).float()
        Z = Variable(Z).to(self.device).float()
        ZU = Variable(ZU).to(self.device).float()
        ZV = Variable(ZV).to(self.device).float()
        N = X.size()[0]

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
            temp_mat = (self.sigma0_2 + self.sigma2_2) * yty - 2 * self.sigma0_2 * self.sigma2_2 * torch.inverse(self.D)\
                       + 1e-5 * torch.eye(self.D.size()[0], device=self.device)
            temp_mat = torch.inverse(temp_mat)
            self.b = torch.matmul(temp_mat, self.sigma2_2 * yt_z_xbeta + self.sigma0_2 * yt_zv)

            # update variance parameter
            xbeta = torch.matmul(X, self.beta)
            yb = torch.matmul(Y, self.b)
            # self.sigma0_2 = 1 / (N * dim_z) * torch.pow(torch.norm(Z - xbeta - yb, p='fro'), 2)
            # self.sigma1_2 = 1 / (N * dim_z) * torch.pow(torch.norm(ZU - xbeta, p='fro'), 2)
            # self.sigma2_2 = 1 / (N * dim_z) * torch.pow(torch.norm(ZV - yb, p='fro'), 2)

            for i in range(1):
                dbbd = torch.matmul(torch.inverse(self.D),
                                    torch.matmul(self.b, torch.matmul(torch.transpose(self.b, 0, 1), torch.inverse(self.D))))
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

        bt_dinv_b = torch.matmul(torch.matmul(torch.transpose(self.b, 0, 1), torch.inverse(self.D)), self.b)
        log_pb = -1 / 2 * (dim_z * torch.log(torch.det(self.D)) + torch.trace(bt_dinv_b))
        utv = torch.matmul(torch.transpose(self.U, 0, 1), self.V)
        utv_norm_2 = torch.pow(torch.norm(utv, p='fro'), 2).cpu().detach().numpy()
        logger.info(f"||U^T * V||^2 = {utv_norm_2:.4}, log p(b) = {log_pb:.3}")


class test_AE(nn.Module):
    def __init__(self, input_dim, left_right=0):
        super(test_AE, self).__init__()
        nn.Module.__init__(self)
        self.name = 'test_AE'
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.gamma = 1
        self.input_dim = input_dim
        self.left_right = left_right

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096, dim_z),
            nn.BatchNorm1d(dim_z),
        )

        self.decoder = nn.Sequential(
            nn.Linear(dim_z, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096, self.input_dim),
            nn.ReLU()
        )

    def forward(self, input):
        z = self.encoder(input)
        output = self.decoder(z)
        return output

    @staticmethod
    def loss(input_, reconstructed):
        recon_loss = torch.mean((reconstructed - input_) ** 2)
        return recon_loss

    def train_(self, train_data_loader, a, optimizer, num_epochs):
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

            for data in train_data_loader:
                # data: 0 lthick, 1 rthick, 2 age, 3 baseline_age, 4 label, 5 subject, 6 timepoint
                image = data[self.left_right]
                optimizer.zero_grad()

                # self-reconstruction loss
                input_ = Variable(image).to(self.device).float()
                reconstructed = self.forward(input_)
                self_reconstruction_loss = self.loss(input_, reconstructed)

                loss = self_reconstruction_loss
                loss.backward()
                optimizer.step()
                tloss += float(loss)
                nb_batches += 1

            epoch_loss = tloss / nb_batches

            if epoch_loss <= best_loss:
                es = 0
                best_loss = epoch_loss
            else:
                es += 1

            end_time = time()
            logger.info(f"Epoch loss (train): {epoch_loss:.4} take {end_time - start_time:.3} seconds\n")