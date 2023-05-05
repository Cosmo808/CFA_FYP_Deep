import torch
from torch.utils import data
from dataset import Dataset_starmen
from data_preprocess import Data_preprocess_starmen
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def generate_sample(baseline_age, age):
    sample = []
    for index, base_a in enumerate(baseline_age):
        match_ba = [i for i, ba in enumerate(baseline_age) if 1e-5 < np.abs(ba - base_a) <= 0.05]
        if match_ba:
            sample.append([index, match_ba])
    result = []
    for index, match in sample:
        match_age = [i for i in match if 1e-5 < np.abs(age[i] - age[index]) <= 0.05]
        for ind in match_age:
            result.append([index, ind])
    index0 = [idx[0] for idx in result]
    index1 = [idx[1] for idx in result]
    return index0, index1


if __name__ == '__main__':

    # load data
    data_generator = Data_preprocess_starmen()
    dataset = data_generator.generate_all()
    dataset.requires_grad = False

    Dataset = Dataset_starmen
    all_data = Dataset(dataset['path'], dataset['subject'], dataset['baseline_age'], dataset['age'],
                       dataset['timepoint'], dataset['first_age'], dataset['alpha'])

    data_loader = torch.utils.data.DataLoader(all_data, batch_size=300, shuffle=False,
                                              num_workers=0, drop_last=False, pin_memory=True)

    for data in data_loader:
        image = torch.tensor([[np.load(path)] for path in data[0]], device=device).float()
        baseline_age = data[2]
        delta_age = data[3] - baseline_age
        index0, index1 = generate_sample(baseline_age, delta_age)
        print(index0)
        image0 = image[index0]
        image1 = image[index1]

    fig, axes = plt.subplots(1, 9, figsize=(18, 1))
    plt.subplots_adjust(wspace=0, hspace=0)

    # image = image[10:20]
    for i in range(10):
        if i == 0:
            continue
        grad_img = image0[i] - image1[i]
        axes[i - 1].matshow(grad_img[0].cpu().detach().numpy(), cmap=matplotlib.cm.get_cmap('bwr'),
                            norm=matplotlib.colors.CenteredNorm())
        fig.colorbar(
            matplotlib.cm.ScalarMappable(cmap=matplotlib.cm.get_cmap('bwr'), norm=matplotlib.colors.CenteredNorm()),
            cax=fig.add_axes([0.92, 0.15, 0.01, 0.7]))

    for axe in axes:
        axe.set_xticks([])
        axe.set_yticks([])

    plt.show()