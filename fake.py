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


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
format = logging.Formatter("%(message)s")

ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(format)
logger.addHandler(ch)

device = torch.device(f"cuda:0")
ratio = 1.0
fold = 0


if __name__ == '__main__':
    logger.info(f"Device is {device}")

    # load data
    data_generator = Data_preprocess_ADNI(ratio=ratio)
    data_generator.device = device
    demo_train, demo_test = data_generator.generate_demo_train_test(fold)
    thick_train, thick_test, input_dim = data_generator.generate_thick_train_test(fold)
    logger.info(f"Loaded {len(demo_train['age']) + len(demo_test['age'])} scans")

    thick_train['left'].requires_grad, thick_test['right'].requires_grad = False, False
    thick_train['left'].requires_grad, thick_test['right'].requires_grad = False, False

    Dataset = Dataset_adni
    train = Dataset(thick_train['left'], thick_train['right'], demo_train['age'], demo_train['baseline_age'],
                    demo_train['label'], demo_train['subject'], demo_train['timepoint'])
    test = Dataset(thick_test['left'], thick_test['right'], demo_test['age'], demo_test['baseline_age'],
                   demo_test['label'], demo_test['subject'], demo_test['timepoint'])

    print(train[0])