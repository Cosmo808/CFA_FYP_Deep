from dataset import Dataset_starmen
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils import data
from model import AE_starmen
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
p = 0.2


def generate_init_data():
    catalog = pd.read_csv('index.csv')
    catalog = catalog.loc[:, catalog.columns != 'id']
    tau_list = catalog.iloc[:, 0]
    age_list = catalog.iloc[:, 2]

    npy_path_list = catalog.iloc[:, 3]
    first_age_list = pd.DataFrame(data=[age_list[int(i / 10) * 10] for i in range(10000)], columns=['first_age'])
    subject_list = pd.DataFrame(data=[int(i / 10) for i in range(10000)], columns=['subject'])
    timepoint_list = pd.DataFrame(data=[i % 10 for i in range(10000)], columns=['timepoint'])
    catalog = pd.concat([npy_path_list, subject_list, tau_list, age_list, timepoint_list, first_age_list], axis=1)
    catalog = catalog.rename(columns={'t': 'age', 'tau': 'baseline_age'})
    return catalog


def generate_train_test(catalog, fold):
    test_num = int(1000 * p)

    test_index = np.arange(test_num, dtype=int) + test_num * fold
    train_index = np.setdiff1d(np.arange(1000, dtype=int), test_index)

    train = catalog.loc[catalog['subject'].isin(train_index)]
    train_data = train.set_index(pd.Series(range(int((1 - p) * 10000))))

    test = catalog.loc[catalog['subject'].isin(test_index)]
    test_data = test.set_index(pd.Series(range(int(p * 10000))))

    return train_data, test_data


def generate_XY(train_data):
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
    return X, Y


if __name__ == '__main__':
    # hyperparameter
    epochs = 500
    lr = 1e-3
    batch_size = 128

    logger.info(f"Device is {device}")
    catalog = generate_init_data()

    for fold in range(5):
        logger.info(f"##### Fold {fold + 1}/5 #####\n")

        train_data, test_data = generate_train_test(catalog, fold)
        train_data.requires_grad = False
        test_data.requires_grad = False

        Dataset = Dataset_starmen
        train = Dataset(train_data['path'], train_data['subject'], train_data['baseline_age'], train_data['age'],
                        train_data['timepoint'], train_data['first_age'])
        test = Dataset(test_data['path'], test_data['subject'], test_data['baseline_age'], test_data['age'],
                       test_data['timepoint'], test_data['first_age'])

        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True,
                                                   num_workers=0, drop_last=False, pin_memory=True)

        # training
        autoencoder = AE_starmen()
        X, Y = generate_XY(train_data)
        X, Y = Variable(X).to(device).float(), Variable(Y).to(device).float()
        autoencoder.X, autoencoder.Y = X, Y

        optimizer_fn = optim.Adam
        optimizer = optimizer_fn(autoencoder.parameters(), lr=lr)
        autoencoder.train_(train_loader, test=test, optimizer=optimizer, num_epochs=epochs)
        torch.save(autoencoder, 'model/{}_fold_starmen'.format(fold))