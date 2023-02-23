import torch
from torch.utils import data
import torch.nn.functional as F
from dataset import Dataset_starmen
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


if __name__ == '__main__':
    ae = torch.load('model/starmen', map_location=device)
    ae.eval()
    ae.to(device)

    # load data
    train_data = torch.load('data/train_starmen')
    test_data = torch.load('data/test_starmen')
    train_data.requires_grad = False
    test_data.requires_grad = False

    Dataset = Dataset_starmen
    train = Dataset(train_data['path'], train_data['subject'], train_data['baseline_age'], train_data['age'],
                    train_data['timepoint'], train_data['first_age'])
    test = Dataset(test_data['path'], test_data['subject'], test_data['baseline_age'], test_data['age'],
                   test_data['timepoint'], test_data['first_age'])

    # train_loader = torch.utils.data.DataLoader(train, batch_size=len(train), shuffle=True,
    #                                            num_workers=0, drop_last=False, pin_memory=True)
    # for data in train_loader:
    #     train_image = torch.tensor([[np.load(path)] for path in data[0]], device=device).float()

    test_loader = torch.utils.data.DataLoader(train, batch_size=256, shuffle=True,
                                               num_workers=0, drop_last=False, pin_memory=True)
    for data in test_loader:
        test_image = torch.tensor([[np.load(path)] for path in data[0]], device=device).float()
        break

    recon_img, z, zu, zv = ae.forward(test_image)
    recon = recon_img[0]
    test = test_image[0] - 1e-6
    print(ae.train_loss[-1])
