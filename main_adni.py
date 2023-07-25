import torch
import torch.optim as optim
from torch.utils import data
from torch.autograd import Variable
import logging
import sys
import os
from dataset import Dataset_adni
from data_preprocess import Data_preprocess_ADNI
import argparse
import model_adni
from utils import adni_utils, RNN_classifier

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
format = logging.Formatter("%(message)s")

ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(format)
logger.addHandler(ch)

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--epochs', type=int, default=40)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--bs', type=int, default=128)
parser.add_argument('--number', type=int, default=10242)
parser.add_argument('--label', type=int, default=-1)
parser.add_argument('--lor', type=int, default=0, help='left (0) or right (1)')
input_para = parser.parse_args()

# hyperparameter
device = torch.device(f"cuda:{input_para.cuda}")
fold = input_para.fold
epochs = input_para.epochs
lr = input_para.lr
batch_size = input_para.bs
number = input_para.number
label = input_para.label
left_right = input_para.lor


if __name__ == '__main__':
    logger.info(f"Device is {device}")
    logger.info(f"##### Fold {fold + 1}/2 #####\n")

    # make directory
    if not os.path.exists('model'):
        os.mkdir('model')
    if not os.path.exists('visualization'):
        os.mkdir('visualization')

    # load data
    data_generator = Data_preprocess_ADNI(number=number, label=label)
    demo_train, demo_test = data_generator.generate_demo_train_test(fold)
    thick_train, thick_test, input_dim = data_generator.generate_thick_train_test(fold)
    logger.info(f"Loaded {len(demo_train['age']) + len(demo_test['age'])} scans")

    Dataset = Dataset_adni
    train = Dataset(thick_train['left'], thick_train['right'], demo_train['age'], demo_train['baseline_age'],
                    demo_train['label'], demo_train['subject'], demo_train['timepoint'])
    test = Dataset(thick_test['left'], thick_test['right'], demo_test['age'], demo_test['baseline_age'],
                   demo_test['label'], demo_test['subject'], demo_test['timepoint'])

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False,
                                               num_workers=0, drop_last=False)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False,
                                              num_workers=0, drop_last=False)
    print('Generating data loader finished...')

    # training
    autoencoder = model_adni.AE_adni(input_dim, left_right)
    autoencoder.device = device
    if hasattr(autoencoder, 'X'):
        X, Y = data_generator.generate_XY(demo_train)
        # X0, Y0 = data_generator.generate_XY(demo_train)
        # X1, Y1 = data_generator.generate_XY(demo_test)
        # X = torch.cat((X0, X1), dim=0)
        # Y = torch.cat(
        #     (torch.cat((Y0, torch.zeros(size=[Y0.size()[0], Y1.size()[1]])), 1), torch.cat((torch.zeros(size=[Y1.size()[0], Y0.size()[1]]), Y1), 1)), dim=0
        # )
        X, Y = Variable(X).to(device).float(), Variable(Y).to(device).float()
        autoencoder.X, autoencoder.Y = X, Y
    if hasattr(autoencoder, 'batch_size'):
        autoencoder.batch_size = batch_size
    print(f"Model has a total of {sum(p.numel() for p in autoencoder.parameters())} parameters")

    print('Start training...')
    optimizer_fn = optim.Adam
    optimizer = optimizer_fn(autoencoder.parameters(), lr=lr)
    ZU, ZV = autoencoder.train_(train_loader, test_loader, optimizer=optimizer, num_epochs=epochs)

    left_right = 'left' if left_right == 0 else 'right'
    label = 'CN' if label == 0 else 'MCI' if label == 1 else 'AD' if label == 2 else 'all'
    torch.save(autoencoder, 'model/{}_fold_{}_{}_{}'.format(fold, label, left_right, autoencoder.name))
    logger.info(f'##### Fold {fold + 1}/2 finished #####\n')
    logger.info('Model saved in model/{}_fold_{}_{}_{}'.format(fold, label, left_right, autoencoder.name))

    adni_utils = adni_utils()
    # spearman corr
    # age_list, index = adni_utils.generate_age_index(demo_train['age'])
    # thick = thick_train['left']
    # thick = torch.tensor(thick[index], device=device).float()
    # _, _, ZU, _, _, _ = autoencoder.forward(thick)
    # adni_utils.global_pca_save(ZU, age_list, autoencoder.name)

    # t-SNE analysis
    label = demo_train['label']
    label[label == 2] = 1
    legend = ['CN', 'MCI', 'AD']
    adni_utils.t_SNE(ZU['train'], label, legend, 'ZU')
    adni_utils.t_SNE(ZV['train'], label, legend, 'ZV')
    print('t-SNE finished...')

    # rnn classification
    rnn = RNN_classifier(12, ZV['train'].size()[1])
    print('Start training classifier...')
    optimizer_fn = optim.Adam
    optimizer_rnn = optimizer_fn(rnn.parameters(), lr=lr)
    demo_all = {'train': demo_train, 'test': demo_test}
    rnn.train_(ZV['train'].to(device).float(), ZV['test'].to(device).float(), demo_all, optimizer=optimizer_rnn, num_epochs=30)