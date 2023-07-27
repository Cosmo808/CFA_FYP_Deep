import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from time import time
import scipy.stats as stats
from sklearn.manifold import TSNE
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

    @staticmethod
    def t_SNE(Z, label, legend, name):
        vis_data = TSNE(n_components=2, perplexity=30.0, n_iter=1000).fit_transform(Z.cpu().detach().numpy())
        vis_x = vis_data[:, 0]
        vis_y = vis_data[:, 1]

        fig, ax = plt.subplots(1)
        ax.set_yticklabels([])
        ax.set_xticklabels([])

        scatter = plt.scatter(vis_x, vis_y, marker='.', c=label, cmap=plt.cm.get_cmap("rainbow"))
        plt.axis('off')
        plt.colorbar()
        plt.title('t-SNE analysis')
        plt.legend(handles=scatter.legend_elements()[0], labels=legend)

        plt.savefig('visualization/{}_t-SNE_analysis.png'.format(name), bbox_inches='tight')
        plt.close()


class RNN_classifier(nn.Module):
    def __init__(self, input_dim, layers_num=12):
        super(RNN_classifier, self).__init__()
        nn.Module.__init__(self)
        self.name = 'RNN_classifier'

        self.layers_num = layers_num
        self.input_dim = input_dim
        self.weight_x = nn.Parameter(torch.ones(size=[layers_num, input_dim]))
        self.weight_g = nn.Parameter(torch.ones(size=[layers_num, input_dim]))
        self.weight_u = nn.Parameter(torch.ones(size=[layers_num, input_dim]))
        self.mask = nn.Parameter(torch.ones(size=[layers_num, input_dim]))
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        x_bar = torch.zeros(size=x.size(), device=x.device, requires_grad=False)
        h = torch.zeros(size=x.size(), device=x.device, requires_grad=False)
        x_bar[0] = x[0].detach().clone()
        h[0] = torch.tanh(torch.mul(x[0].detach().clone(), self.weight_x[0]))
        x_bar[1] = torch.mul(h[0].detach().clone(), self.weight_g[0]) + x[0]

        for t in range(1, self.layers_num):
            x_trans = torch.mul(x[t], self.mask[t]) + torch.mul(x_bar[t], 1 - self.mask[t])
            u = torch.tanh(torch.mul(x_trans, self.weight_x[t]))

            f = torch.sigmoid(h[t - 1].detach().clone() + torch.mul(u, self.weight_u[t]))
            f = torch.mul(f, self.mask[t])

            h[t] = torch.mul(f, h[t - 1]) + torch.mul(1 - f, u)
            h[t] = h[0] + torch.relu(h[t] - h[0])

            if t < self.layers_num - 1:
                x_bar[t + 1] = torch.mul(h[t].detach().clone(), self.weight_g[t]) + x_trans.detach().clone()

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

    def train_(self, ZV, ZV_test, demo_all, optimizer, num_epochs):
        self.to(ZV.device)
        demo = demo_all['train']
        for epoch in range(num_epochs):
            start_time = time()
            subject = demo['subject'][0]
            zv = ZV[0].view(1, -1)
            age = np.array([demo['age'][0]])
            label = np.array([demo['label'][0]])
            timepoint = np.array([demo['timepoint'][0]])

            acc, acc_conv = [], []  # 1 true, 0 false
            age_diff, age_diff_conv = [], []
            labels, labels_conv = [], []  # 1 ad, 0 cn
            for i in range(1, len(demo['age'])):
                if demo['subject'][i] == subject:
                    zv = torch.cat((zv, ZV[i].view(1, -1)), 0)
                    age = np.append(age, demo['age'][i])
                    label = np.append(label, demo['label'][i])
                    timepoint = np.append(timepoint, demo['timepoint'][i])
                else:
                    if len(age) >= 4:
                        zv, age, label, timepoint = self.expand_zv(zv, torch.tensor(age), torch.tensor(label),
                                                                   torch.tensor(timepoint))
                        first_conv = True
                        len_conv = len(acc_conv)
                        for j in range(2, len(age)):
                            if label[j] == 0 or label[j] == 3:
                                for k in range(2, j + 1):
                                    if label[k - 1] == -1:
                                        continue
                                    pred = self.forward(torch.cat(
                                        (zv[:k], torch.zeros([self.layers_num - k, self.input_dim], device=zv.device)
                                         ), 0))
                                    if label[j] == 0:
                                        loss = pred[j]
                                    else:
                                        loss = 1 - pred[j]

                                    loss.backward()
                                    optimizer.step()

                                    if label[k - 1] == label[j]:
                                        age_diff.append(age[j] - age[k - 1])
                                        labels.append(label[j] / 3)
                                        if loss < 0.5:
                                            acc.append(1)
                                        else:
                                            acc.append(0)
                                    else:
                                        if first_conv:
                                            age_diff_conv.append(age[j] - age[k - 1])
                                            labels_conv.append(label[j] / 3)
                                            if loss < 0.5:
                                                acc_conv.append(1)
                                            else:
                                                acc_conv.append(0)
                            if len(acc_conv) != len_conv:
                                first_conv = False

                    subject = demo['subject'][i]
                    zv = ZV[i].view(1, -1)
                    age = np.array([demo['age'][i]])
                    label = np.array([demo['label'][i]])
                    timepoint = np.array([demo['timepoint'][i]])

            self.plot_pred(acc, age_diff, labels, 'stable')
            self.plot_pred(acc_conv, age_diff_conv, labels_conv, 'conversion')
            print('stable acc: {}%, conversion acc: {}%'
                  .format(round(sum(acc) / len(acc) * 100, 2), round(sum(acc_conv) / len(acc_conv) * 100, 2)))

            # self.evaluate(ZV_test, demo_all['test'])

            take_time = round(time() - start_time, 2)
            print('Epoch {}/{} take {} seconds'.format(epoch + 1, num_epochs, take_time), '\n')

    def evaluate(self, ZV, demo):
        self.to(ZV.device)
        self.training = False
        self.eval()

        with torch.no_grad():
            subject = demo['subject'][0]
            zv = ZV[0].view(1, -1)
            age = np.array([demo['age'][0]])
            label = np.array([demo['label'][0]])
            timepoint = np.array([demo['timepoint'][0]])

            acc, acc_conv = [], []  # 1 true, 0 false
            age_diff, age_diff_conv = [], []
            labels, labels_conv = [], []  # 1 ad, 0 cn
            for i in range(1, len(demo['age'])):
                if demo['subject'][i] == subject:
                    zv = torch.cat((zv, ZV[i].view(1, -1)), 0)
                    age = np.append(age, demo['age'][i])
                    label = np.append(label, demo['label'][i])
                    timepoint = np.append(timepoint, demo['timepoint'][i])
                else:
                    if len(age) >= 4:
                        zv, age, label, timepoint = self.expand_zv(zv, torch.tensor(age), torch.tensor(label),
                                                                   torch.tensor(timepoint))
                        first_conv = True
                        len_conv = len(acc_conv)
                        for j in range(2, len(age)):
                            if label[j] == 0 or label[j] == 3:
                                for k in range(2, j + 1):
                                    if label[k - 1] == -1:
                                        continue
                                    pred = self.forward(torch.cat(
                                        (zv[:k], torch.zeros([self.layers_num - k, self.input_dim], device=zv.device)
                                         ), 0))
                                    if label[j] == 0:
                                        loss = pred[j]
                                    else:
                                        loss = 1 - pred[j]

                                    if label[k - 1] == label[j]:
                                        age_diff.append(age[j] - age[k - 1])
                                        labels.append(label[j] / 3)
                                        if loss < 0.5:
                                            acc.append(1)
                                        else:
                                            acc.append(0)
                                    else:
                                        if first_conv:
                                            age_diff_conv.append(age[j] - age[k - 1])
                                            labels_conv.append(label[j] / 3)
                                            if loss < 0.5:
                                                acc_conv.append(1)
                                            else:
                                                acc_conv.append(0)
                            if len(acc_conv) != len_conv:
                                first_conv = False

                    subject = demo['subject'][i]
                    zv = ZV[i].view(1, -1)
                    age = np.array([demo['age'][i]])
                    label = np.array([demo['label'][i]])
                    timepoint = np.array([demo['timepoint'][i]])

        self.training = False
        self.plot_pred(acc, age_diff, labels, 'stable')
        self.plot_pred(acc_conv, age_diff_conv, labels_conv, 'conversion')
        stable_acc = round(sum(acc) / len(acc) * 100, 2)
        try:
            conv_acc = round(sum(acc_conv) / len(acc_conv) * 100, 2)
        except ZeroDivisionError:
            conv_acc = 0.
        print('stable acc: {}%, conversion acc: {}%'.format(stable_acc, conv_acc))

    @staticmethod
    def plot_pred(acc, age_diff, labels, name):
        print(name)
        acc, age_diff, labels = np.array(acc), np.round(np.array(age_diff), 2), np.array(labels)
        ind = np.argsort(age_diff)
        acc, age_diff, labels = acc[ind], age_diff[ind], labels[ind]

        age_diff[age_diff == 0.25] = 0.5
        age_diff[age_diff == 0.75] = 1.0
        age_diff[age_diff == 1.75] = 1.5
        age_diff[age_diff == 2.75] = 2.5
        age_diff[age_diff == 3.5] = 3.0
        age_diff[age_diff == 3.75] = 4.0
        age_diff[age_diff == 4.5] = 5.0
        age_diff[age_diff == 5.5] = 6.0
        age_diff[age_diff == 6.5] = 7.0
        age_diff[age_diff == 7.5] = 8.0
        age_diff[age_diff == 8.5] = 8.0

        idx = np.where(age_diff == 2.5)[0]
        idx0, idx1 = idx[:len(idx) // 2], idx[len(idx) // 2:]
        age_diff[idx0] = 2.0
        age_diff[idx1] = 3.0

        unique_age = np.unique(age_diff)
        accuracy, num, ad_num, cn_num = [], [], [], []
        for age in unique_age:
            index = np.where(age_diff == age)
            pred = acc[index]
            try:
                accuracy.append(np.sum(pred) / len(pred) * 100)
            except ZeroDivisionError:
                accuracy.append(0.)
            label = labels[index]
            num.append(len(pred))
            ad_num.append(sum(label))
            cn_num.append(len(label) - sum(label))

        print('acc: ', accuracy, '\n')


class FC_classifier(nn.Module):
    def __init__(self, input_dim):
        super(FC_classifier, self).__init__()
        nn.Module.__init__(self)
        self.name = 'FC_classifier'

        self.input_dim = input_dim
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, zv, X, predicted_age, params):
        Y = X[:, :2].clone()
        baseline_age = X[0, -1]
        beta, sigma0_2, sigma1_2, sigma2_2 = params['beta'], params['sigma0_2'], params['sigma1_2'], params['sigma2_2']
        # get b
        yt = torch.transpose(Y, 0, 1)
        yty = torch.matmul(yt, Y)
        yt_zv = torch.matmul(yt, zv)
        xbeta = torch.matmul(X, beta)
        yt_z_xbeta = torch.matmul(yt, zv - xbeta)
        b = torch.matmul(
            torch.inverse(
                (sigma0_2 + sigma2_2) * yty - 2 * sigma0_2 * sigma2_2 * torch.eye(yty.size()[0], device=zv.device)),
            sigma2_2 * yt_z_xbeta + sigma0_2 * yt_zv
        )
        # predict zv
        x_pred = torch.tensor([[1, predicted_age - baseline_age, baseline_age]], device=zv.device)
        y_pred = torch.tensor([[1, predicted_age - baseline_age]], device=zv.device)
        zv_pred = torch.matmul(x_pred, beta) + torch.matmul(y_pred, b)

        return torch.sigmoid(self.fc(zv_pred))

    def train_(self, ZV, ZV_test, demo_all, optimizer, num_epochs):
        self.to(ZV.device)
        demo = demo_all['train']
        X, Y = demo_all['X'], demo_all['Y']
        for epoch in range(num_epochs):
            start_time = time()
            subject = demo['subject'][0]
            zv = ZV[0].view(1, -1)
            age = np.array([demo['age'][0]])
            label = np.array([demo['label'][0]])
            timepoint = np.array([demo['timepoint'][0]])
            x, y = X[0].view(1, -1), Y[0].view(1, -1)

            acc, acc_conv = [], []  # 1 true, 0 false
            age_diff, age_diff_conv = [], []
            labels, labels_conv = [], []  # 1 ad, 0 cn
            for i in range(1, len(demo['age'])):
                if demo['subject'][i] == subject:
                    zv = torch.cat((zv, ZV[i].view(1, -1)), 0)
                    age = np.append(age, demo['age'][i])
                    label = np.append(label, demo['label'][i])
                    timepoint = np.append(timepoint, demo['timepoint'][i])
                    x = torch.cat((x, X[i].view(1, -1)), dim=0)
                    y = torch.cat((y, Y[i].view(1, -1)), dim=0)
                else:
                    if len(age) >= 4:
                        first_conv = True
                        len_conv = len(acc_conv)
                        for j in range(2, len(age)):
                            if label[j] == 0 or label[j] == 3:
                                for k in range(2, j + 1):
                                    if label[k - 1] == -1:
                                        continue
                                    pred = self.forward(zv[:k], x[:k], age[j], demo_all['params'])
                                    if label[j] == 0:
                                        loss = pred
                                    else:
                                        loss = 1 - pred

                                    loss.backward()
                                    optimizer.step()

                                    if label[k - 1] == label[j]:
                                        age_diff.append(age[j] - age[k - 1])
                                        labels.append(label[j] / 3)
                                        if loss < 0.5:
                                            acc.append(1)
                                        else:
                                            acc.append(0)
                                    else:
                                        if first_conv:
                                            age_diff_conv.append(age[j] - age[k - 1])
                                            labels_conv.append(label[j] / 3)
                                            if loss < 0.5:
                                                acc_conv.append(1)
                                            else:
                                                acc_conv.append(0)
                            if len(acc_conv) != len_conv:
                                first_conv = False

                    subject = demo['subject'][i]
                    zv = ZV[i].view(1, -1)
                    age = np.array([demo['age'][i]])
                    label = np.array([demo['label'][i]])
                    timepoint = np.array([demo['timepoint'][i]])
                    x, y = X[i].view(1, -1), Y[i].view(1, -1)

            self.plot_pred(acc, age_diff, labels, 'stable')
            self.plot_pred(acc_conv, age_diff_conv, labels_conv, 'conversion')
            print('stable acc: {}%, conversion acc: {}%'
                  .format(round(sum(acc) / len(acc) * 100, 2), round(sum(acc_conv) / len(acc_conv) * 100, 2)))

            # self.evaluate(ZV_test, demo_all['test'])

            take_time = round(time() - start_time, 2)
            print('Epoch {}/{} take {} seconds'.format(epoch + 1, num_epochs, take_time), '\n')

    @staticmethod
    def plot_pred(acc, age_diff, labels, name):
        print(name)
        acc, age_diff, labels = np.array(acc), np.round(np.array(age_diff), 2), np.array(labels)
        ind = np.argsort(age_diff)
        acc, age_diff, labels = acc[ind], age_diff[ind], labels[ind]

        age_diff[age_diff == 0.25] = 0.5
        age_diff[age_diff == 0.75] = 1.0
        age_diff[age_diff == 1.75] = 1.5
        age_diff[age_diff == 2.75] = 2.5
        age_diff[age_diff == 3.5] = 3.0
        age_diff[age_diff == 3.75] = 4.0
        age_diff[age_diff == 4.5] = 5.0
        age_diff[age_diff == 5.5] = 6.0
        age_diff[age_diff == 6.5] = 7.0
        age_diff[age_diff == 7.5] = 8.0
        age_diff[age_diff == 8.5] = 8.0

        idx = np.where(age_diff == 2.5)[0]
        idx0, idx1 = idx[:len(idx) // 2], idx[len(idx) // 2:]
        age_diff[idx0] = 2.0
        age_diff[idx1] = 3.0

        unique_age = np.unique(age_diff)
        accuracy, num, ad_num, cn_num = [], [], [], []
        for age in unique_age:
            index = np.where(age_diff == age)
            pred = acc[index]
            try:
                accuracy.append(np.sum(pred) / len(pred) * 100)
            except ZeroDivisionError:
                accuracy.append(0.)
            label = labels[index]
            num.append(len(pred))
            ad_num.append(sum(label))
            cn_num.append(len(label) - sum(label))

        print('acc: ', accuracy, '\n')
