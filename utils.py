import torch
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
