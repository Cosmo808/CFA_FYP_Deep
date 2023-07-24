import torch
from torch.utils import data
from dataset import Dataset_starmen
from data_preprocess import Data_preprocess_starmen
from torch.autograd import Variable
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

if __name__ == '__main__':
    # load model
    autoencoder = torch.load('model/best_starmen', map_location=device)
    autoencoder.eval()

    # load data
    data_generator = Data_preprocess_starmen()
    dataset = data_generator.generate_all()
    dataset.requires_grad = False

    Dataset = Dataset_starmen
    all_data = Dataset(dataset['path'], dataset['subject'], dataset['baseline_age'], dataset['age'],
                       dataset['timepoint'], dataset['first_age'], dataset['alpha'])

    data_loader = torch.utils.data.DataLoader(all_data, batch_size=512, shuffle=False,
                                              num_workers=0, drop_last=False, pin_memory=True)

    # get Z, ZU, ZV
    with torch.no_grad():
        Z, ZU, ZV = None, None, None
        for data in data_loader:
            image = torch.tensor([[np.load(path)] for path in data[0]], device=device).float()

            # self-reconstruction loss
            input_ = Variable(image).to(device)
            reconstructed, z, zu, zv = autoencoder.forward(input_)
            self_reconstruction_loss = autoencoder.loss(input_, reconstructed)

            # store Z, ZU, ZV
            if Z is None:
                Z, ZU, ZV = z, zu, zv
            else:
                Z = torch.cat((Z, z), 0)
                ZU = torch.cat((ZU, zu), 0)
                ZV = torch.cat((ZV, zv), 0)

    # get psi
    psi = dataset['alpha'] * (dataset['age'] - dataset['baseline_age'])
    psi_array = np.linspace(min(psi), max(psi), num=9)

    index = [np.nonzero(np.abs(np.array(psi) - p) < 0.05)[0][:2] for p in psi_array]
    print(index)
    index = [j for i in index for j in i]
    print(psi_array)
    print(index)

    # individual trajectory
    subject = [i // 10 for i in index]
    subject_img = []
    for s in subject:
        subject_img += list(np.arange(s * 10, (s + 1) * 10))

    path = dataset.iloc[subject_img, 0]
    image = torch.tensor([[np.load(p)] for p in path], device=device).float()

    fig, axes = plt.subplots(len(index), 10, figsize=(20, 2 * len(index)))
    plt.subplots_adjust(wspace=0, hspace=0)
    for i in range(len(index)):
        for j in range(10):
            axes[i][j].matshow(255 * image[10 * i + j][0].cpu().detach().numpy())
    for axe in axes:
        for ax in axe:
            ax.set_xticks([])
            ax.set_yticks([])
    plt.savefig('visualization/latent_space/individual_trajectory.png', bbox_inches='tight')

    # global trajectory
    zu = ZU[index]
    global_tra = autoencoder.decoder(zu)

    fig, axes = plt.subplots(1, len(index), figsize=(2 * len(index), 2))
    plt.subplots_adjust(wspace=0, hspace=0)
    for i in range(len(index)):
        axes[i].matshow(255 * global_tra[i][0].cpu().detach().numpy())
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig('visualization/latent_space/global_trajectory.png', bbox_inches='tight')

    # individual heterogeneity
    index = [np.nonzero(np.abs(np.array(psi) - -3.7439508) < 0.1)[0]]
    index = index[0][:6]
    print(index)

    path = dataset.iloc[index, 0]
    image = torch.tensor([[np.load(p)] for p in path], device=device).float()
    zv = ZV[index]
    indiv_hetero = autoencoder.decoder(zv)

    fig, axes = plt.subplots(len(index), 2, figsize=(4, 2 * len(index)))
    plt.subplots_adjust(wspace=0, hspace=0)
    for i in range(len(index)):
        axes[i][0].matshow(255 * image[i][0].cpu().detach().numpy())
        axes[i][1].matshow(255 * indiv_hetero[i][0].cpu().detach().numpy())
    for axe in axes:
        for ax in axe:
            ax.set_xticks([])
            ax.set_yticks([])
    plt.savefig('visualization/latent_space/indiv_heterogeneity.png', bbox_inches='tight')

    plt.close()
