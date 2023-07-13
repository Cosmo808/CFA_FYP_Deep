import torch
import torch.nn as nn
import matplotlib.pyplot as plt
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


class adni_utils:
    @staticmethod
    def generate_age_index(age_all):
        min_age, max_age = min(age_all), max(age_all)
        age_list = torch.linspace(min_age, max_age, steps=50)
        index, ages = [], []
        for ind, age in enumerate(age_list):
            for i, a in enumerate(age_all):
                if np.abs(a - age) <= 0.2:
                    index.append(i)
                    ages.append(a)
                    break
        return ages, index

    @staticmethod
    def global_pca_save(ZU, age_list, file_name):
        if not os.path.exists('global_pca'):
            os.mkdir('global_pca')

        _, _, V = torch.pca_lowrank(ZU)
        if file_name == 'rank_VAE':
            PCA_ZU = -1 * ZU.cpu().detach().numpy()
        else:
            PCA_ZU = -1 * torch.matmul(ZU, V[:, 0]).cpu().detach().numpy()

        global_pca_age = {'global_pca': PCA_ZU, 'age': age_list}
        torch.save(global_pca_age, 'global_pca/{}'.format(file_name))

        spearman = np.abs(stats.spearmanr(PCA_ZU, age_list))
        print('Spearman correlation: ', spearman)

    @staticmethod
    def global_pca_plot():
        path = 'global_pca/'
        for root, dirs, files in os.walk(path):
            for file in files:
                global_pca_age = torch.load(os.path.join(path, file))
                Y, X = global_pca_age['global_pca'], global_pca_age['age']
                plt.plot(X, Y)
        plt.legend(files)
        plt.savefig('visualization/global_pca', bbox_inches='tight')
        plt.close()

    @staticmethod
    def merge_loader(*loaders):
        for loader in loaders:
            for data in loader:
                yield data


class RNN_classifier(nn.Module):
    def __init__(self, layers_num, input_dim):
        super(RNN_classifier, self).__init__()
        nn.Module.__init__(self)
        self.name = 'RNN_classifier'

        self.layers_num = layers_num
        self.input_dim = input_dim
        self.weight_x = nn.Parameter(torch.ones(size=[layers_num, input_dim]))
        self.weight_g = nn.Parameter(torch.ones(size=[layers_num, input_dim]))
        self.weight_h = nn.Parameter(torch.ones(size=[layers_num, input_dim]))
        self.weight_u = nn.Parameter(torch.ones(size=[layers_num, input_dim]))
        self.mask = nn.Parameter(torch.ones(size=[layers_num, input_dim]))
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        x_bar, h = torch.zeros(size=x.size(), device=self.device), torch.zeros(size=x.size(), device=self.device)
        x_bar[0] = x[0]
        h[0] = torch.tanh(torch.mul(x[0], self.weight_x[0]))
        x_bar[1] = torch.mul(h[0], self.weight_g[0]) + x[0]

        for t in range(1, self.layers_num):
            x_trans = torch.mul(x[t], self.mask[t]) + torch.mul(x_bar[t], 1 - self.mask[t])
            u = torch.tanh(torch.mul(x_trans, self.weight_x[t]))

            f = torch.sigmoid(torch.mul(h[t-1], self.weight_h[t]) + torch.mul(u, self.weight_u[t]))
            f = torch.mul(f, self.mask[t])

            h[t] = torch.mul(f, h[t-1]) + torch.min(1 - f, u)
            h[t] = h[0] + torch.relu(h[t] - h[0])

            if t < self.layers_num - 1:
                x_bar[t+1] = torch.mul(h[t], self.weight_g[t]) + x_trans

        pred = torch.sigmoid(self.fc(x_bar))
        return pred

    def expand_zv(self, zv, age, label, timepoint):
        zeros = torch.zeros(size=[1, self.input_dim], device=zv.device)
        timepoint = torch.cat((timepoint.squeeze(), 99 * torch.ones(self.layers_num)), dim=0)
        for i in range(self.layers_num):
            if i + 1 >= timepoint[i]:
                continue
            else:
                timepoint = torch.cat((timepoint[:i], torch.tensor([i + 1]), timepoint[i:]), 0)
                age = torch.cat((age[:i], torch.tensor([-1]), age[i:]), 0)
                label = torch.cat((label[:i], torch.tensor([-1]), label[i:]), 0)
                zv = torch.cat((zv[:i], zeros, zv[i:]), 0)
        return zv, age, label, timepoint[:self.layers_num]

    def train_(self, ZV, demo, optimizer, num_epochs):
        self.to(ZV.device)

        for epoch in range(num_epochs):
            optimizer.zero_grad()

            subject = demo['subject'][0]
            zv = ZV[0]
            age = np.array([demo['age'][0]])
            label = np.array([demo['label'][0]])
            timepoint = np.array([demo['timepoint'][0]])

            acc = []  # 1 true, 0 false
            age_diff = []
            for i in range(1, len(demo['age'])):
                if demo['subject'][i] == subject:
                    zv = torch.cat((zv, ZV[i]), 0)
                    age = np.append(age, demo['age'][i])
                    label = np.append(label, demo['label'][i])
                    timepoint = np.append(timepoint, demo['timepoint'][i])
                else:
                    if len(age) >= 4:
                        zv, age, label, timepoint = self.expand_zv(zv, torch.tensor(age), torch.tensor(label),
                                                                   torch.tensor(timepoint))
                        for j in range(2, len(age)):
                            if label[j] == 0:
                                for k in range(2, j + 1):
                                    pred = self.forward(torch.cat(
                                        (zv[:k], torch.zeros([self.layers_num - k, self.input_dim])
                                         ), 0))
                                    loss = pred[j]
                                    loss.backward()
                                    optimizer.step()

                                    age_diff.append(age[j] - age[k])
                                    if loss < 0.5:
                                        acc.append(1)
                                    else:
                                        acc.append(0)
                            elif label[j] == 3:
                                for k in range(2, j + 1):
                                    pred = self.forward(torch.cat(
                                        (zv[:k], torch.zeros([self.layers_num - k, self.input_dim])
                                         ), 0))
                                    loss = 1 - pred[j]
                                    loss.backward()
                                    optimizer.step()

                                    age_diff.append(age[j] - age[k])
                                    if loss > 0.5:
                                        acc.append(1)
                                    else:
                                        acc.append(0)

                    subject = demo['subject'][0]
                    zv = ZV[0]
                    age = np.array([demo['age'][0]])
                    label = np.array([demo['label'][0]])
                    timepoint = np.array([demo['timepoint'][0]])

            accuracy = round(sum(acc) / len(acc), 3)
            print('#### Epoch {}/{}: accuracy {} ####'.format(epoch + 1, num_epochs, accuracy))