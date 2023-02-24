import pandas as pd
import numpy as np
import torch


class Data_preprocess:
    def __init__(self):
        self.p = 0.2
        self.catalog = pd.read_csv('index.csv')
        self.generate_init_data()

    def generate_init_data(self):
        self.catalog = self.catalog.loc[:, self.catalog.columns != 'id']
        tau_list = self.catalog.iloc[:, 0]
        age_list = self.catalog.iloc[:, 2]

        npy_path_list = self.catalog.iloc[:, 3]
        first_age_list = pd.DataFrame(data=[age_list[int(i / 10) * 10] for i in range(10000)], columns=['first_age'])
        subject_list = pd.DataFrame(data=[int(i / 10) for i in range(10000)], columns=['subject'])
        timepoint_list = pd.DataFrame(data=[i % 10 for i in range(10000)], columns=['timepoint'])

        self.catalog = pd.concat([npy_path_list, subject_list, tau_list, age_list, timepoint_list, first_age_list], axis=1)
        self.catalog = self.catalog.rename(columns={'t': 'age', 'tau': 'baseline_age'})

    def generate_train_test(self, fold):
        test_num = int(1000 * self.p)

        test_index = np.arange(test_num, dtype=int) + test_num * fold
        train_index = np.setdiff1d(np.arange(1000, dtype=int), test_index)

        train = self.catalog.loc[self.catalog['subject'].isin(train_index)]
        train_data = train.set_index(pd.Series(range(int((1 - self.p) * 10000))))

        test = self.catalog.loc[self.catalog['subject'].isin(test_index)]
        test_data = test.set_index(pd.Series(range(int(self.p * 10000))))

        return train_data, test_data

    def generate_XY(self, train_data):
        N = int(10000 * (1 - self.p))
        I = int(1000 * (1 - self.p))
    
        delta_age = train_data['age'] - train_data['baseline_age']
        ones = pd.DataFrame(np.ones(shape=[N, 1]))
        X = pd.concat([ones, delta_age, train_data['baseline_age']], axis=1)
    
        zero = pd.DataFrame(np.zeros(shape=[10, 2]))
        for i in range(I):
            y = X.iloc[i * 10:(i + 1) * 10, :2]
            y = y.set_axis([0, 1], axis=1)
            if i == 0:
                zeros = pd.concat([zero for j in range(I - 1)], axis=0)
                Y = pd.concat([y, zeros], axis=0).reset_index(drop=True)
            elif i != I - 1:
                zeros1 = pd.concat([zero for j in range(i)], axis=0)
                zeros2 = pd.concat([zero for j in range(I - 1 - i)], axis=0).reset_index(drop=True)
                yy = pd.concat([zeros1, y, zeros2], axis=0).reset_index(drop=True)
                Y = pd.concat([Y, yy], axis=1)
            else:
                zeros = pd.concat([zero for j in range(I - 1)], axis=0)
                yy = pd.concat([zeros, y], axis=0).reset_index(drop=True)
                Y = pd.concat([Y, yy], axis=1)
    
        X = torch.tensor(X.values)
        Y = torch.tensor(Y.values)
        return X, Y

    # def plot_sample(self):
    #     delta_age = self.catalog['age'] - self.catalog['baseline_age']
    #
    #     from model import AE_starmen
    #     index1, index2 = AE_starmen.generate_sample(tau_list[:1000], delta_age[:1000])
    #
    #     n_col = 10
    #     fig, axes = plt.subplots(2, n_col, figsize=(2 * n_col, 4))
    #     plt.subplots_adjust(wspace=0, hspace=0)
    #     for i in range(n_col):
    #         select = np.random.randint(0, len(index1))
    #         ind1 = index1[select]
    #         ind2 = index2[select]
    #         sub1 = self.catalog.iloc[ind1, :]
    #         sub2 = self.catalog.iloc[ind2, :]
    #         path1 = sub1[0]
    #         path2 = sub2[0]
    #         img_1 = np.load(path1)
    #         img_2 = np.load(path2)
    #         axes[0][i].matshow(255 * img_1)
    #         axes[1][i].matshow(255 * img_2)
    #     for axe in axes:
    #         for ax in axe:
    #             ax.set_xticks([])
    #             ax.set_yticks([])
    #     plt.show()
