import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils import data
import matplotlib.pyplot as plt
from time import time
from dataset import Dataset_starmen
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

dim_z = 4


class beta_VAE(nn.Module):
    """
    This is the convolutionnal variationnal autoencoder for the 2D starmen dataset.
    """

    def __init__(self):
        super(beta_VAE, self).__init__()
        nn.Module.__init__(self)
        self.beta = 5
        self.gamma = 100
        self.lr = 1e-4                                            # For epochs between MCMC steps
        self.epoch = 0                                            # For tensorboard to keep track of total number of epochs
        self.name = 'beta_VAE'

        self.conv1 = nn.Conv2d(1, 16, 3, stride=2, padding=1)  # 16 x 32 x 32
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)  # 32 x 16 x 16
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)  # 32 x 8 x 8
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.fc10 = nn.Linear(2048, dim_z)
        self.fc11 = nn.Linear(2048, dim_z)

        self.fc3 = nn.Linear(dim_z, 512)
        self.upconv1 = nn.ConvTranspose2d(8, 64, 3, stride=2, padding=1, output_padding=1)  # 32 x 16 x 16
        self.upconv2 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)  # 16 x 32 x 32
        self.upconv3 = nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1)  # 1 x 64 x 64
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(32)

    def encoder(self, image):
        h1 = F.relu(self.bn1(self.conv1(image)))
        h2 = F.relu(self.bn2(self.conv2(h1)))
        h3 = F.relu(self.bn3(self.conv3(h2)))
        mu = torch.tanh(self.fc10(h3.flatten(start_dim=1)))
        logVar = self.fc11(h3.flatten(start_dim=1))
        return mu, logVar

    def decoder(self, encoded):
        h6 = F.relu(self.fc3(encoded)).reshape([encoded.size()[0], 8, 8, 8])
        h7 = F.relu(self.bn4(self.upconv1(h6)))
        h8 = F.relu(self.bn5(self.upconv2(h7)))
        reconstructed = F.relu(self.upconv3(h8))
        return reconstructed
    
    def reparametrize(self, mu, logVar):
        # Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar/2).to(device)
        eps = torch.normal(mean=torch.tensor([0 for i in range(std.shape[1])]).float(), std =1).to(device)
        if self.beta != 0:                   # beta VAE
            return mu + eps*std
        else:                           # regular AE
            return mu
        
    def forward(self, image):
        mu, logVar = self.encoder(image)
        if self.training:
            encoded = self.reparametrize(mu, logVar)
        else:
            encoded = mu
        reconstructed = self.decoder(encoded)
        return mu, logVar, reconstructed

    def loss(self, mu, logVar, reconstructed, input_):
        kl_divergence = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp()) / mu.shape[0]
        recon_error = torch.sum((reconstructed - input_)**2) / input_.shape[0]
        return recon_error, kl_divergence

    def train_(self, data_loader, test, optimizer, num_epochs=20, criterion=None):

        self.to(device)
        if criterion is None:
            criterion = self.loss
        best_loss = 1e10
        es = 0

        for epoch in range(num_epochs):

            start_time = time()
            if es == 100:
                break

            logger.info('Epoch {}/{}'.format(epoch+1, num_epochs ))

            tloss = 0.0
            nb_batches = 0

            for data in data_loader:
                image = torch.tensor([[np.load(path)] for path in data[0]], device=device).float()
                optimizer.zero_grad()

                input_ = Variable(image).to(device)
                mu, logVar, reconstructed = self.forward(input_)
                reconstruction_loss, kl_loss = criterion(mu, logVar, input_, reconstructed)
                loss = reconstruction_loss + self.beta * kl_loss

                loss.backward()
                optimizer.step()
                tloss += float(loss)
                nb_batches += 1

            epoch_loss = tloss/nb_batches
            test_loss, _ = self.evaluate(test)

            if epoch_loss <= best_loss:
                es = 0
                best_loss = epoch_loss
            else:
                es += 1
            end_time = time()
            self.plot_recon(test)
            logger.info(f"Epoch loss (train/test): {epoch_loss:.3e}/{test_loss:.3e} took {end_time-start_time} seconds")

        print('Complete training')
        return

    def evaluate(self, data, longitudinal=None, individual_RER=None, writer=None, train_losses=None):
        """
        This is called on a subset of the dataset and returns the encoded latent variables as well as the evaluation
        loss for this subset.
        """
        self.to(device)
        self.training = False
        self.eval()
        criterion = self.loss
        dataloader = torch.utils.data.DataLoader(data, batch_size=1, num_workers=0, shuffle=False)
        tloss = 0.0
        trecon_loss, tkl_loss, talignment_loss = 0.0, 0.0, 0.0
        nb_batches = 0
        encoded_data = torch.empty([0, dim_z])

        with torch.no_grad():
            for data in dataloader:
                image = torch.tensor([[np.load(path)] for path in data[0]]).float()

                if longitudinal is not None:
                    input_ = Variable(data[0]).to(device)
                    mu, logVar, reconstructed = self.forward(input_)
                    reconstruction_loss, kl_loss = criterion(mu, logVar, input_, reconstructed)
                    alignment_loss = longitudinal(data, mu, reconstructed, individual_RER)
                    loss = reconstruction_loss + self.beta * kl_loss + self.gamma * alignment_loss
                    trecon_loss += reconstruction_loss
                    tkl_loss += kl_loss
                    talignment_loss += alignment_loss
                else:
                    input_ = Variable(image).to(device)
                    mu, logVar, reconstructed = self.forward(input_)
                    reconstruction_loss, kl_loss = criterion(mu, logVar, input_, reconstructed)
                    loss = reconstruction_loss + self.beta * kl_loss

                tloss += float(loss)
                nb_batches += 1
                encoded_data = torch.cat((encoded_data, mu.to('cpu')), 0)

        if writer is not None:
            writer.add_scalars('Loss/recon', {'test' : trecon_loss/nb_batches, 'train' : train_losses[0]} , self.epoch)
            writer.add_scalars('Loss/kl', {'test' : tkl_loss/nb_batches, 'train' : train_losses[1]}, self.epoch)
            writer.add_scalars('Loss/alignment', {'test' : talignment_loss/nb_batches, 'train' : train_losses[2]}, self.epoch)

        loss = tloss/nb_batches
        self.training = True
        return loss, encoded_data

    def plot_recon(self, data, n_subject=3):
        # Plot the reconstruction
        fig, axes = plt.subplots(2 * n_subject, 10, figsize=(20, 4 * n_subject))
        plt.subplots_adjust(wspace=0, hspace=0)
        for j in range(n_subject):
            for i in range(10):
                test_image = torch.tensor(np.load(data[j * 10 + i][0])).resize(1, 1, 64, 64).float()
                test_image = Variable(test_image).to(device)
                _, _, out = self.forward(test_image)
                axes[2 * j][i].matshow(255 * test_image[0][0].cpu().detach().numpy())
                axes[2 * j + 1][i].matshow(255 * out[0][0].cpu().detach().numpy())
        for axe in axes:
            for ax in axe:
                ax.set_xticks([])
                ax.set_yticks([])
        plt.savefig('visualization/beta_VAE_recon.png', bbox_inches='tight')
        plt.close()


def main():
    """
    For debugging purposes only, once the architectures and training routines are efficient,
    this file will not be called as a script anymore.
    """
    # logger.info("DEBUGGING THE network.py FILE")
    logger.info(f"Device is {device}")

    # hyperparameter
    epochs = 200
    lr = 1e-3
    batch_size = 128

    # load data
    train_data = torch.load('data/train_starmen')
    test_data = torch.load('data/test_starmen')

    print(f"Loaded {len(train_data['path']) + len(test_data['path'])} scans")

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
    autoencoder = beta_VAE()
    print(f"Model has a total of {sum(p.numel() for p in autoencoder.parameters())} parameters")

    optimizer_fn = optim.Adam
    optimizer = optimizer_fn(autoencoder.parameters(), lr=lr)
    autoencoder.train_(train_loader, test=test, optimizer=optimizer, num_epochs=epochs)
    torch.save(autoencoder, 'model/beta_VAE')


if __name__ == '__main__':
    main()