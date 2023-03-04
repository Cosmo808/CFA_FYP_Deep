import torch
from torch.utils import data
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

    train_loader = torch.utils.data.DataLoader(train, batch_size=256, shuffle=False,
                                               num_workers=0, drop_last=False, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=1000, shuffle=False,
                                              num_workers=0, drop_last=False, pin_memory=True)

    for data in test_loader:
        test_image = torch.tensor([[np.load(path)] for path in data[0]], device=device).float()
        break
    index0 = [61, 230, 653]
    i = 653 // 10
    test_image = test_image[i * 10:(i + 1) * 10]
    recon_img, z, zu, zv = autoencoder.forward(test_image)
    global_tra = autoencoder.decoder(zu)
    indiv_hetero = autoencoder.decoder(zv)

    # fig, axes = plt.subplots(3, 10, figsize=(20, 6))
    # plt.subplots_adjust(wspace=0, hspace=0)
    #
    # for i in range(10):
    #     axes[0][i].matshow(255 * test_image[i][0].cpu().detach().numpy())
    # for i in range(10):
    #     axes[1][i].matshow(255 * global_tra[i][0].cpu().detach().numpy())
    # for i in range(10):
    #     axes[2][i].matshow(255 * indiv_hetero[i][0].cpu().detach().numpy())
    # for axe in axes:
    #     for ax in axe:
    #         ax.set_xticks([])
    #         ax.set_yticks([])

    # fig, axes = plt.subplots(3, 10, figsize=(20, 6))
    # plt.subplots_adjust(wspace=0, hspace=0)
    #
    # for i in range(10):
    #     axes[0][i].matshow(255 * test_image[i][0].cpu().detach().numpy())
    # for i in range(10):
    #     axes[1][i].matshow(255 * global_tra[i][0].cpu().detach().numpy())
    # for i in range(10):
    #     axes[2][i].matshow(255 * indiv_hetero[i][0].cpu().detach().numpy())
    # for axe in axes:
    #     for ax in axe:
    #         ax.set_xticks([])
    #         ax.set_yticks([])
    #
    # plt.show()

    for data in test_loader:
        test_image = torch.tensor([[np.load(path)] for path in data[0]], device=device).float()
        baseline_age = data[2]
        delta_age = data[3] - baseline_age

        sample = []
        for index, base_a in enumerate(baseline_age):
            match_ba = [i for i, ba in enumerate(baseline_age) if 1e-5 < np.abs(ba - base_a) <= 0.05]
            if match_ba:
                sample.append([index, match_ba])
        result = []
        for index, match in sample:
            match_age = [i for i in match if 1e-5 < np.abs(delta_age[i] - delta_age[index]) <= 0.05]
            if match_age:
                result.append([index] + match_age)
        break

    sim = []
    for match in result:
        image = test_image[match]
        recon_img, z, zu, zv = autoencoder.forward(image)
        global_tra = autoencoder.decoder(zu)
        indiv_hetero = autoencoder.decoder(zv)
        mean_ = torch.mean(global_tra, dim=0, keepdim=True)
        simi = autoencoder.loss(global_tra, mean_[0])
        sim.append(float(simi) / 64 / 64)
    print(np.mean(sim), np.std(sim))


    # fig, axes = plt.subplots(3, 8, figsize=(16, 6))
    # plt.subplots_adjust(wspace=0, hspace=0)
    # for i in range(4):
    #     axes[0][2 * i].matshow(255 * image0[i][0].cpu().detach().numpy())
    # for i in range(4):
    #     axes[1][2 * i].matshow(255 * global_tra[i][0].cpu().detach().numpy())
    # for i in range(4):
    #     axes[2][2 * i].matshow(255 * indiv_hetero[i][0].cpu().detach().numpy())
    #
    # recon_img, z, zu, zv = autoencoder.forward(image1)
    # global_tra = autoencoder.decoder(zu)
    # indiv_hetero = autoencoder.decoder(zv)
    #
    # for i in range(4):
    #     axes[0][2 * i + 1].matshow(255 * image1[i][0].cpu().detach().numpy())
    # for i in range(4):
    #     axes[1][2 * i + 1].matshow(255 * global_tra[i][0].cpu().detach().numpy())
    # for i in range(4):
    #     axes[2][2 * i + 1].matshow(255 * indiv_hetero[i][0].cpu().detach().numpy())
    #
    # for axe in axes:
    #     for ax in axe:
    #         ax.set_xticks([])
    #         ax.set_yticks([])
    # plt.show()