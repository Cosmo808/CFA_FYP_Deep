import torch
from torch.utils import data
from dataset import Dataset_starmen
from data_preprocess import Data_preprocess
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
    logger.info(f"Device is {device}")
    data_generator = Data_preprocess()
    autoencoder = torch.load('model/best_starmen', map_location=device)
    autoencoder.eval()
    # load train and test data
    train_data, test_data = data_generator.generate_train_test(0)
    train_data.requires_grad = False
    test_data.requires_grad = False

    Dataset = Dataset_starmen
    train = Dataset(train_data['path'], train_data['subject'], train_data['baseline_age'], train_data['age'],
                    train_data['timepoint'], train_data['first_age'])
    test = Dataset(test_data['path'], test_data['subject'], test_data['baseline_age'], test_data['age'],
                   test_data['timepoint'], test_data['first_age'])

    train_loader = torch.utils.data.DataLoader(train, batch_size=256, shuffle=False,
                                               num_workers=0, drop_last=False, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=256, shuffle=False,
                                              num_workers=0, drop_last=False, pin_memory=True)
    for data in test_loader:
        image = torch.tensor([[np.load(path)] for path in data[0]], device=device).float()
        break
    image = image[:50]
    recon_img, z, zu, zv = autoencoder.forward(image)
    global_tra = autoencoder.decoder(zu)
    indiv_hetero = autoencoder.decoder(zv)
    fig, axes = plt.subplots(4 * (image.shape[0] // 10), 10, figsize=(20, 8 * (image.shape[0] // 10)))
    plt.subplots_adjust(wspace=0, hspace=0)
    for i in range(image.shape[0] // 10):
        for j in range(10):
            axes[4 * i][j].matshow(255 * image[10 * i + j][0].cpu().detach().numpy())
            axes[4 * i + 1][j].matshow(255 * recon_img[10 * i + j][0].cpu().detach().numpy())
            axes[4 * i + 2][j].matshow(255 * global_tra[10 * i + j][0].cpu().detach().numpy())
            axes[4 * i + 3][j].matshow(255 * indiv_hetero[10 * i + j][0].cpu().detach().numpy())
    for axe in axes:
        for ax in axe:
            ax.set_xticks([])
            ax.set_yticks([])
    plt.savefig('visualization/a.png', bbox_inches='tight')
    plt.close()
    for i in range(image.shape[0] // 10):
        print(test_data['age'].iloc[i * 10:(i + 1) * 10])