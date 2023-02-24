import torch
import torch.optim as optim
from torch.utils import data
import logging
import sys
import os
from dataset import Dataset_starmen
from model import AE_starmen


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
format = logging.Formatter("%(message)s")

ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(format)
logger.addHandler(ch)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    """
        For debugging purposes only, once the architectures and training routines are efficient,
        this file will not be called as a script anymore.
        """

    # logger.info("DEBUGGING THE network.py FILE")
    logger.info(f"Device is {device}")

    # hyperparameter
    epochs = 100
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
    autoencoder = AE_starmen()
    print(f"Model has a total of {sum(p.numel() for p in autoencoder.parameters())} parameters")

    optimizer_fn = optim.Adam
    optimizer = optimizer_fn(autoencoder.parameters(), lr=lr)
    autoencoder.train_(train_loader, test=test, optimizer=optimizer, num_epochs=epochs)
    if not os.path.exists('model'):
        os.mkdir('model')
    torch.save(autoencoder, 'model/starmen')
