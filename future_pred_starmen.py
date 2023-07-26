import torch
from torch.utils import data
from dataset import Dataset_starmen
from data_preprocess import Data_preprocess_starmen
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_pred_loss(image, missing_num=7):
    autoencoder.eval()
    num_subject = image.size()[0] // 10
    idx0, idx1 = [], []
    for i in range(num_subject):
        idx0 += list(np.arange(i * 10, i * 10 + (10 - missing_num)))
        idx1 += list(np.arange(i * 10 + (10 - missing_num), i * 10 + 10))
    image0, image1 = image[idx0], image[idx1]
    input_0 = Variable(image0).to(autoencoder.device)
    input_1 = Variable(image1).to(autoencoder.device)

    # arange data
    age = pd.DataFrame(data[3].cpu().detach(), columns=['age'])
    baseline_age = pd.DataFrame(data[2].cpu().detach(), columns=['baseline_age'])
    data_xy = pd.concat([age, baseline_age], axis=1)
    # get X and Y
    X, Y = data_generator.generate_XY(data_xy)
    X, Y = Variable(X).to(autoencoder.device).float(), Variable(Y).to(autoencoder.device).float()
    X0, X1 = X[idx0], X[idx1]
    Y0, Y1 = Y[idx0], Y[idx1]
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
    predicted = autoencoder.decoder(z1)

    # plot
    sub = 4
    fig, axes = plt.subplots(2, 10, figsize=(20, 2 * 2))
    plt.subplots_adjust(wspace=0, hspace=0)
    for i in range(10):
        axes[0][i].matshow(255 * image[i + sub * 10][0].cpu().detach().numpy())
    for i in range(10):
        if i < (10 - missing_num):
            axes[1][i].matshow(255 * image[i + sub * 10][0].cpu().detach().numpy())
        else:
            axes[1][i].matshow(255 * predicted[i + sub * missing_num - (10 - missing_num)][0].cpu().detach().numpy())

    for axe in axes:
        for ax in axe:
            ax.set_xticks([])
            ax.set_yticks([])

    plt.savefig('visualization/future_pred.png', bbox_inches='tight')
    plt.close()
    exit()
    return torch.sum((predicted - input_1) ** 2) / input_1.shape[0]


if __name__ == '__main__':

    logger.info(f"Device is {device}")
    data_generator = Data_preprocess_starmen()

    autoencoder = torch.load('model/best_starmen', map_location=device)
    autoencoder.eval()
    autoencoder.Training = False
    autoencoder.device = device

    # load train and test data
    train_data, test_data = data_generator.generate_train_test(0)
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
    batches = 0
    pred_losses = 0
    ZU, ZV = None, None
    with torch.no_grad():
        with torch.no_grad():
            for data in test_loader:
                batches += 1
                image = torch.tensor([[np.load(path)] for path in data[0]], device=autoencoder.device).float()
                input_ = Variable(image).to(autoencoder.device)

                reconstructed, z, zu, zv = autoencoder.forward(input_)

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
                prediction_loss = get_pred_loss(image)
                pred_losses += float(prediction_loss)

    print(pred_losses / batches / 64 / 64)