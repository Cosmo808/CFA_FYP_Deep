from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch


catalog = pd.read_csv('index.csv')
catalog = catalog.loc[:, catalog.columns != 'id']
tau_list = catalog.iloc[:, 0]
alpha_list = catalog.iloc[:, 1]
age_list = catalog.iloc[:, 2]

npy_path_list = catalog.iloc[:, 3]
first_age_list = pd.DataFrame(data=[age_list[int(i / 10) * 10] for i in range(10000)], columns=['first_age'])
subject_list = pd.DataFrame(data=[int(i / 10) for i in range(10000)], columns=['subject'])
timepoint_list = pd.DataFrame(data=[i % 10 for i in range(10000)], columns=['timepoint'])
catalog = pd.concat([npy_path_list, subject_list, tau_list, age_list, timepoint_list, first_age_list], axis=1)
catalog = catalog.rename(columns={'t': 'age', 'tau': 'baseline_age'})

p = 0.2
index = np.random.permutation(np.arange(1000))
train_index = np.sort(index[int(len(index) * p):])
test_index = np.sort(index[:int(len(index) * p)])

train = catalog.loc[catalog['subject'].isin(train_index)]
train_data = train.set_index(pd.Series(range(int((1 - p) * 10000))))
# torch.save(train_data, 'data/train_starmen')

test = catalog.loc[catalog['subject'].isin(test_index)]
test_data = test.set_index(pd.Series(range(int(p * 10000))))
# torch.save(test_data, 'data/test_starmen')


def hist_norm(data, figure_num, bins=70):
    plt.figure(figure_num)
    plt.hist(data, bins=bins, density=True)
    mu, std = stats.norm.fit(data)
    x_axis = np.arange(mu - 3 * std, mu + 3 * std, 0.001)
    plt.plot(x_axis, 1 * stats.norm.pdf(x_axis, mu, std))
    plt.xlim([min(x_axis), max(x_axis)])
    plt.legend(['mu={}, std={}'.format(np.round(mu, 2), np.round(std, 3))])


def data_distribution():
    hist_norm(tau_list, 0)
    plt.xlabel('Tau')
    plt.title('Distribution of Tau')

    hist_norm(alpha_list, 1)
    plt.xlabel('Alpha')
    plt.title('Distribution of Alpah')

    hist_norm(age_list, 2, bins=80)
    plt.xlabel('Age')
    plt.title('Distribution of Age')

    hist_norm(first_age_list, 3)
    plt.xlabel('Baseline Age')
    plt.title('Distribution of Baseline Age')

    plt.show()


def starmen(rows=4):
    for index, row in catalog.iterrows():
        # tau, alpha, t, path, subject, timepoint
        npy_path = row['path']
        img_np = np.load(npy_path)
        plt.subplot(rows, 10, index + 1)
        plt.imshow(img_np, cmap='gray')
        plt.title(str(int(row['timepoint'] + 1)), fontsize=10)
        plt.xticks([])
        plt.yticks([])
        if index % 10 == 0:
            plt.ylabel('subject {}'.format(row['subject']), rotation=0, fontsize=10, labelpad=25, va='center')
        if index + 1 == rows * 10:
            break
    plt.show()


def plot_sample():
    delta_age = catalog['age'] - catalog['baseline_age']

    from model import AE_starmen
    index1, index2 = AE_starmen.generate_sample(tau_list[:1000], delta_age[:1000])

    n_col = 10
    fig, axes = plt.subplots(2, n_col, figsize=(2 * n_col, 4))
    plt.subplots_adjust(wspace=0, hspace=0)
    for i in range(n_col):
        select = np.random.randint(0, len(index1))
        ind1 = index1[select]
        ind2 = index2[select]
        sub1 = catalog.iloc[ind1, :]
        sub2 = catalog.iloc[ind2, :]
        path1 = sub1[0]
        path2 = sub2[0]
        img_1 = np.load(path1)
        img_2 = np.load(path2)
        axes[0][i].matshow(255 * img_1)
        axes[1][i].matshow(255 * img_2)
    for axe in axes:
        for ax in axe:
            ax.set_xticks([])
            ax.set_yticks([])
    plt.show()


def generate_XY():
    N = int(10000 * (1 - p))
    I = int(1000 * (1 - p))

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
    torch.save(X, 'data/X')
    torch.save(Y, 'data/Y')


if __name__ == '__main__':
    # generate_XY()
    plot_sample()