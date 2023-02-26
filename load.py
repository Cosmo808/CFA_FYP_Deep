import torch
from torch.utils import data
import torch.nn.functional as F
from dataset import Dataset_starmen
from data_preprocess import Data_preprocess
from torch.autograd import Variable
import matplotlib
import matplotlib.pyplot as plt
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
fold = 0


if __name__ == '__main__':
    logger.info(f"##### Fold {fold + 1}/5 #####\n")

    # load model
    # autoencoder = torch.load('5-fold/gpu4/{}_fold_starmen'.format(fold), map_location=device)
    autoencoder = torch.load('model/best_starmen', map_location=device)
    autoencoder.eval()

    # load data
    data_generator = Data_preprocess()
    train_data, test_data = data_generator.generate_train_test(fold)
    train_data.requires_grad = False
    test_data.requires_grad = False

    Dataset = Dataset_starmen
    train = Dataset(train_data['path'], train_data['subject'], train_data['baseline_age'], train_data['age'],
                    train_data['timepoint'], train_data['first_age'])
    test = Dataset(test_data['path'], test_data['subject'], test_data['baseline_age'], test_data['age'],
                   test_data['timepoint'], test_data['first_age'])

    # train_loader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True,
    #                                            num_workers=0, drop_last=False, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(train, batch_size=1000, shuffle=False,
                                              num_workers=0, drop_last=False, pin_memory=True)

    for data in test_loader:
        test_image = torch.tensor([[np.load(path)] for path in data[0]], device=device).float()
        break
    test_image = test_image[30:40]
    recon_img, z, zu, zv = autoencoder.forward(test_image)
    global_tra = autoencoder.decoder(zu)
    indiv_hetero = autoencoder.decoder(zv)

    fig, axes = plt.subplots(4, 10, figsize=(20, 8))
    plt.subplots_adjust(wspace=0, hspace=0)

    for i in range(10):
        axes[0][i].matshow(255 * test_image[i][0].cpu().detach().numpy())
    for i in range(10):
        axes[1][i].matshow(255 * recon_img[i][0].cpu().detach().numpy())
    for i in range(10):
        axes[2][i].matshow(255 * global_tra[i][0].cpu().detach().numpy())
    for i in range(10):
        axes[3][i].matshow(255 * indiv_hetero[i][0].cpu().detach().numpy())
    for axe in axes:
        for ax in axe:
            ax.set_xticks([])
            ax.set_yticks([])

    fig, axes = plt.subplots(4, 9, figsize=(18, 8))
    plt.subplots_adjust(wspace=0, hspace=0)

    for i in range(9):
        axes[0][i].matshow((test_image[i + 1][0] - test_image[i][0]).cpu().detach().numpy(), cmap=matplotlib.cm.get_cmap('bwr'),
                                         norm=matplotlib.colors.CenteredNorm())
    for i in range(9):
        axes[1][i].matshow((recon_img[i + 1][0] - recon_img[i][0]).cpu().detach().numpy(), cmap=matplotlib.cm.get_cmap('bwr'),
                                         norm=matplotlib.colors.CenteredNorm())
    for i in range(9):
        axes[2][i].matshow((global_tra[i + 1][0] - global_tra[i][0]).cpu().detach().numpy(), cmap=matplotlib.cm.get_cmap('bwr'),
                                         norm=matplotlib.colors.CenteredNorm())
    for i in range(9):
        axes[3][i].matshow((indiv_hetero[i + 1][0] - indiv_hetero[i][0]).cpu().detach().numpy(), cmap=matplotlib.cm.get_cmap('bwr'),
                                         norm=matplotlib.colors.CenteredNorm())
    for axe in axes:
        for ax in axe:
            ax.set_xticks([])
            ax.set_yticks([])

    # plt.show()

    train_loss, test_loss = [], []
    for fold in range(5):
        autoencoder = torch.load('5-fold/gpu4/{}_fold_starmen'.format(fold), map_location=device)
        train_loss.append(autoencoder.train_loss[-1])
        test_loss.append(autoencoder.test_loss[-1])
    train_mean, test_mean = np.mean(train_loss), np.mean(test_loss)
    train_std, test_std = np.std(train_loss), np.std(test_loss)
    print(train_mean / 64 / 64, train_std / 64 / 64)
    print(test_mean / 64 / 64, test_std / 64 / 64)

