import torch
from torch.utils import data
from dataset import Dataset_starmen
from data_preprocess import Data_preprocess_starmen
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.stats import norm
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

    vis_data = TSNE(n_components=2, perplexity=30.0, n_iter=1000).fit_transform(ZV.cpu().detach().numpy())
    # plot the result

    vis_x = vis_data[:, 0]
    vis_y = vis_data[:, 1]

    fig, ax = plt.subplots(1)
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    plt.scatter(vis_x, vis_y, marker='.', c=norm.cdf(np.array(dataset['age'][:])), cmap=plt.cm.get_cmap("rainbow"))
    plt.axis('off')
    plt.colorbar()
    plt.title('t-SNE of ZV space across age')
    plt.show()