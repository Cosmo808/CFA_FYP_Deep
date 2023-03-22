import torch
import torch.optim as optim
from torch.utils import data
from torch.autograd import Variable
import logging
import sys
import os
from dataset import Dataset_starmen
from data_preprocess import Data_preprocess
from model import AE_starmen


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
    """
        For debugging purposes only, once the architectures and training routines are efficient,
        this file will not be called as a script anymore.
        """

    # logger.info("DEBUGGING THE network.py FILE")
    logger.info(f"Device is {device}")
    logger.info(f"##### Fold {fold + 1}/5 #####\n")

    # hyperparameter
    epochs = 200
    lr = 1e-3
    batch_size = 128

    # load data
    data_generator = Data_preprocess()
    train_data, test_data = data_generator.generate_train_test(fold)
    logger.info(f"Loaded {len(train_data['path']) + len(test_data['path'])} scans")

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
    X, Y = data_generator.generate_XY(train_data)
    X, Y = Variable(X).to(device).float(), Variable(Y).to(device).float()
    autoencoder.X, autoencoder.Y = X, Y
    print(f"Model has a total of {sum(p.numel() for p in autoencoder.parameters())} parameters")

    optimizer_fn = optim.Adam
    optimizer = optimizer_fn(autoencoder.parameters(), lr=lr)

    autoencoder = torch.load('model/best_starmen', map_location=device)
    autoencoder.eval()
    autoencoder.train_(train_loader, test=test, optimizer=optimizer, num_epochs=1)
    exit()

    autoencoder.train_(train_loader, test=test, optimizer=optimizer, num_epochs=epochs)
    if not os.path.exists('model'):
        os.mkdir('model')
    torch.save(autoencoder, 'model/{}_starmen'.format(fold))
    logger.info(f"##### Fold {fold + 1}/5 finished #####\n")
    logger.info(f"Model saved in model/{fold}_starmen")