import torch
import torch.optim as optim
from torch.utils import data
from torch.autograd import Variable
import logging
import sys
import os
from dataset import Dataset_starmen
from data_preprocess import Data_preprocess
import argparse
import model


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
format = logging.Formatter("%(message)s")

ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(format)
logger.addHandler(ch)

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--fold', type=int, default=0)
input_para = parser.parse_args()

if __name__ == '__main__':
    logger.info(f"Device is {torch.device(input_para.device)}")
    logger.info(f"##### Fold {input_para.fold + 1}/5 #####\n")

    # make directory
    if not os.path.exists('model'):
        os.mkdir('model')
    if not os.path.exists('visualization'):
        os.mkdir('visualization')

    # hyperparameter
    epochs = 400
    lr = 1e-3
    batch_size = 128

    # load data
    data_generator = Data_preprocess()
    train_data, test_data = data_generator.generate_train_test(input_para.fold)
    logger.info(f"Loaded {len(train_data['path']) + len(test_data['path'])} scans")

    train_data.requires_grad = False
    test_data.requires_grad = False
    Dataset = Dataset_starmen
    train = Dataset(train_data['path'], train_data['subject'], train_data['baseline_age'], train_data['age'],
                    train_data['timepoint'], train_data['first_age'], train_data['alpha'])
    test = Dataset(test_data['path'], test_data['subject'], test_data['baseline_age'], test_data['age'],
                   test_data['timepoint'], test_data['first_age'], test_data['alpha'])

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False,
                                               num_workers=0, drop_last=False, pin_memory=True)

    # training
    autoencoder = model.Riem_VAE()
    autoencoder.device = torch.device(input_para.device)
    if hasattr(autoencoder, 'X'):
        X, Y = data_generator.generate_XY(train_data)
        X, Y = Variable(X).to(torch.device(input_para.device)).float(), Variable(Y).to(torch.device(input_para.device)).float()
        autoencoder.X, autoencoder.Y = X, Y
    if hasattr(autoencoder, 'batch_size'):
        autoencoder.batch_size = batch_size
    if hasattr(autoencoder, 'fold'):
        autoencoder.fold = input_para.fold
    print(f"Model has a total of {sum(p.numel() for p in autoencoder.parameters())} parameters")

    optimizer_fn = optim.Adam
    optimizer = optimizer_fn(autoencoder.parameters(), lr=lr)
    autoencoder.train_(train_loader, test=test, optimizer=optimizer, num_epochs=epochs)
    torch.save(autoencoder, 'model/{}_fold_{}'.format(input_para.fold, autoencoder.name))
    logger.info(f"##### Fold {input_para.fold + 1}/5 finished #####\n")
    logger.info("Model saved in model/{}_fold_{}".format(input_para.fold, autoencoder.name))