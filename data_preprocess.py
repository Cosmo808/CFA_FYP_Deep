import pandas as pd
import numpy as np
import torch
import csv
import h5py


class Data_preprocess_starmen:
    def __init__(self):
        self.p = 0.2
        self.catalog = pd.read_csv('index.csv')
        self.generate_init_data()

    def generate_init_data(self):
        self.catalog = self.catalog.loc[:, self.catalog.columns != 'id']
        tau_list = self.catalog.iloc[:, 0]
        alpha_list = self.catalog.iloc[:, 1]
        age_list = self.catalog.iloc[:, 2]

        npy_path_list = self.catalog.iloc[:, 3]
        first_age_list = pd.DataFrame(data=[age_list[int(i / 10) * 10] for i in range(10000)], columns=['first_age'])
        subject_list = pd.DataFrame(data=[int(i / 10) for i in range(10000)], columns=['subject'])
        timepoint_list = pd.DataFrame(data=[i % 10 for i in range(10000)], columns=['timepoint'])

        self.catalog = pd.concat([npy_path_list, subject_list, tau_list, age_list, timepoint_list, first_age_list, alpha_list], axis=1)
        self.catalog = self.catalog.rename(columns={'t': 'age', 'tau': 'baseline_age'})

    def generate_train_test(self, fold):
        test_num = int(1000 * self.p)

        test_index = np.arange(test_num, dtype=int) + test_num * fold
        train_index = np.setdiff1d(np.arange(1000, dtype=int), test_index)

        train = self.catalog.loc[self.catalog['subject'].isin(train_index)]
        train_data = train.set_index(pd.Series(range(int((1 - self.p) * 10000))))

        test = self.catalog.loc[self.catalog['subject'].isin(test_index)]
        test_data = test.set_index(pd.Series(range(int(self.p * 10000))))

        return train_data, test_data

    def generate_all(self):
        return self.catalog

    def generate_XY(self, train_data):
        N = len(train_data.index)
        I = N // 10
    
        delta_age = train_data['age'] - train_data['baseline_age']
        ones = pd.DataFrame(np.ones(shape=[N, 1]))
        X = pd.concat([ones, delta_age, train_data['baseline_age']], axis=1)
    
        zero = pd.DataFrame(np.zeros(shape=[10, 2]))
        for i in range(I):
            y = X.iloc[i * 10:(i + 1) * 10, :2]
            y = y.set_axis([0, 1], axis=1)
            if i == 0:
                zeros = pd.concat([zero for j in range(I - 1)], axis=0)
                Y = pd.concat([y, zeros], axis=0).reset_index(drop=True)
            elif i != I - 1:
                zeros1 = pd.concat([zero for j in range(i)], axis=0)
                zeros2 = pd.concat([zero for j in range(I - 1 - i)], axis=0).reset_index(drop=True)
                yy = pd.concat([zeros1, y, zeros2], axis=0).reset_index(drop=True)
                Y = pd.concat([Y, yy], axis=1)
            else:
                zeros = pd.concat([zero for j in range(I - 1)], axis=0)
                yy = pd.concat([zeros, y], axis=0).reset_index(drop=True)
                Y = pd.concat([Y, yy], axis=1)
    
        X = torch.tensor(X.values)
        Y = torch.tensor(Y.values)
        return X, Y


class Data_preprocess_ADNI:
    def __init__(self, number=0.25, label=-1):
        self.number = number
        self.label = label

        # demographic
        self.demo_train = h5py.File('./ADNI/adni_all_surf_info_regular_longitudinal_random_train.mat')
        self.demo_test = h5py.File('./ADNI/adni_all_surf_info_regular_longitudinal_random_test.mat')

        # self.demo_train = h5py.File('/projects/students/chaoqiang/VGCNNRNN/DataPrepare/DataOutput/'
        #                             'adni_all_surf_info_regular_longitudinal_random_train.mat')
        # self.demo_test = h5py.File('/projects/students/chaoqiang/VGCNNRNN/DataPrepare/DataOutput/'
        #                            'adni_all_surf_info_regular_longitudinal_random_test.mat')
        print('Reading demographical data finished...')

        # thickness
        self.thickness_train = h5py.File('/projects/students/chaoqiang/VGCNNRNN/DataPrepare/DataOutput/'
                                         'adni_all_surf_thickness_regular_longitudinal_random_train.mat')
        self.thickness_test = h5py.File('/projects/students/chaoqiang/VGCNNRNN/DataPrepare/DataOutput/'
                                        'adni_all_surf_thickness_regular_longitudinal_random_test.mat')
        print('Reading thickness data finished...')

        # sort index
        self.label_idx_train, self.label_idx_test = None, None
        self.idx_train, self.idx_test = None, None

    def generate_demo_train_test(self, fold):
        # load data
        age_train = torch.tensor(self.demo_train['Age']).float().squeeze()
        age_test = torch.tensor(self.demo_test['Age']).float().squeeze()
        label_train = torch.tensor(self.demo_train['Label']).float().squeeze()
        label_test = torch.tensor(self.demo_test['Label']).float().squeeze()
        timepoint_train = torch.tensor(self.demo_train['Wave']).float().squeeze()
        timepoint_test = torch.tensor(self.demo_test['Wave']).float().squeeze()
        with open('ADNI/subject_train.csv', 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            subject_train = torch.tensor([[int(cell[:3] + cell[6:]) for cell in row] for row in csvreader]).squeeze()
        with open('ADNI/subject_test.csv', 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            subject_test = torch.tensor([[int(cell[:3] + cell[6:]) for cell in row] for row in csvreader]).squeeze()

        # select label
        if self.label != -1:
            if self.label == 0:
                self.label_idx_train = torch.nonzero(label_train == 0)
                self.label_idx_test = torch.nonzero(label_test == 0)
            if self.label == 1:
                self.label_idx_train = torch.cat((torch.nonzero(label_train == 1), torch.nonzero(label_train == 2)), dim=0)
                self.label_idx_test = torch.cat((torch.nonzero(label_test == 1), torch.nonzero(label_test == 2)), dim=0)
            if self.label == 2:
                self.label_idx_train = torch.nonzero(label_train == 3)
                self.label_idx_test = torch.nonzero(label_test == 3)
            age_train, age_test = age_train[self.label_idx_train].squeeze(), age_test[self.label_idx_test].squeeze()
            label_train, label_test = label_train[self.label_idx_train].squeeze(), label_test[self.label_idx_test].squeeze()
            timepoint_train, timepoint_test = timepoint_train[self.label_idx_train].squeeze(), timepoint_test[self.label_idx_test].squeeze()
            subject_train, subject_test = subject_train[self.label_idx_train].squeeze(), subject_test[self.label_idx_test].squeeze()

        # get sort data index
        idx_train, idx_test = np.lexsort((timepoint_train.numpy(), subject_train.numpy())), np.lexsort((timepoint_test.numpy(), subject_test.numpy()))
        self.idx_train, self.idx_test = idx_train, idx_test

        # sort data
        age_train, age_test = age_train[idx_train], age_test[idx_test]
        label_train, label_test = label_train[idx_train], label_test[idx_test]
        timepoint_train, timepoint_test = timepoint_train[idx_train], timepoint_test[idx_test]
        subject_train, subject_test = subject_train[idx_train], subject_test[idx_test]

        baseline_age_train, baseline_age_test = [], []
        s_old = None
        for age, subject in zip(age_train, subject_train):
            if s_old is None:
                baseline_age_train.append(age)
                s_old = subject
            else:
                if subject == s_old:
                    baseline_age_train.append(baseline_age_train[-1])
                else:
                    baseline_age_train.append(age)
                    s_old = subject
        for age, subject in zip(age_test, subject_test):
            if s_old is None:
                baseline_age_test.append(age)
                s_old = subject
            else:
                if subject == s_old:
                    baseline_age_test.append(baseline_age_test[-1])
                else:
                    baseline_age_test.append(age)
                    s_old = subject
        baseline_age_train = torch.tensor(baseline_age_train).float()
        baseline_age_test = torch.tensor(baseline_age_test).float()

        demo_train = {'age': age_train, 'baseline_age': baseline_age_train, 'label': label_train,
                      'subject': subject_train, 'timepoint': timepoint_train}
        demo_test = {'age': age_test, 'baseline_age': baseline_age_test, 'label': label_test,
                     'subject': subject_test, 'timepoint': timepoint_test}

        print('Generating demographical data finished...')
        if fold == 0:
            return demo_train, demo_test
        else:
            return demo_test, demo_train

    def generate_thick_train_test(self, fold):
        if self.idx_train is None:
            _, _ = self.generate_demo_train_test(fold)

        left_thick_train = self.thickness_train['lthick_regular'][:, :self.number]
        right_thick_train = self.thickness_train['rthick_regular'][:, :self.number]
        left_thick_test = self.thickness_test['lthick_regular'][:, :self.number]
        right_thick_test = self.thickness_test['rthick_regular'][:, :self.number]

        if self.label != -1:
            label_idx_train = self.label_idx_train.view(1, -1).squeeze().numpy()
            label_idx_test = self.label_idx_test.view(1, -1).squeeze().numpy()
            left_thick_train, left_thick_test = left_thick_train[label_idx_train], left_thick_test[label_idx_test]
            right_thick_train, right_thick_test = right_thick_train[label_idx_train], right_thick_test[label_idx_test]

        print('Start sorting index...')
        left_thick_train, right_thick_train = left_thick_train[self.idx_train], right_thick_train[self.idx_train]
        left_thick_test, right_thick_test = left_thick_test[self.idx_test], right_thick_test[self.idx_test]
        thick_train = {'left': left_thick_train, 'right': right_thick_train}
        thick_test = {'left': left_thick_test, 'right': right_thick_test}

        print('Generating thickness data finished...')
        if fold == 0:
            return thick_train, thick_test, self.number
        else:
            return thick_test, thick_train, self.number

    def generate_XY(self, data):
        N = data['age'].size()[0]
        I = len(torch.unique(data['subject']))

        delta_age = (data['age'] - data['baseline_age']).view(N, -1)
        ones = torch.ones(size=delta_age.size())
        X = torch.cat((ones, delta_age, data['baseline_age'].view(N, -1)), dim=1)

        Y, old_s, cnt_zero = None, None, 0
        for i in range(N):
            if old_s is None:
                old_s = data['subject'][i]
            elif old_s != data['subject'][i]:
                old_s = data['subject'][i]
                cnt_zero += 1

            zeros0 = torch.zeros(size=[1, 2 * cnt_zero])
            zeros1 = torch.zeros(size=[1, 2 * (I - 1 - cnt_zero)])
            yy = X[i, :2].view(1, 2)
            yy = torch.cat((zeros0, yy, zeros1), dim=1)

            if Y is None:
                Y = yy
            else:
                Y = torch.cat((Y, yy), dim=0)

        print('Generating X and Y finished...')
        return X, Y

    def generate_orig_data(self):
        # load data
        age_train = torch.tensor(self.demo_train['Age']).float().squeeze()
        age_test = torch.tensor(self.demo_test['Age']).float().squeeze()
        label_train = torch.tensor(self.demo_train['Label']).float().squeeze()
        label_test = torch.tensor(self.demo_test['Label']).float().squeeze()
        timepoint_train = torch.tensor(self.demo_train['Wave']).float().squeeze()
        timepoint_test = torch.tensor(self.demo_test['Wave']).float().squeeze()
        with open('ADNI/subject_train.csv', 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            subject_train = torch.tensor([[int(cell[:3] + cell[6:]) for cell in row] for row in csvreader]).squeeze()
        with open('ADNI/subject_test.csv', 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            subject_test = torch.tensor([[int(cell[:3] + cell[6:]) for cell in row] for row in csvreader]).squeeze()

        demo_train = {'age': age_train, 'label': label_train,
                      'subject': subject_train, 'timepoint': timepoint_train}
        demo_test = {'age': age_test, 'label': label_test,
                     'subject': subject_test, 'timepoint': timepoint_test}
        print("Generating original demographical data finished...")

        left_thick_train = self.thickness_train['lthick_regular']
        right_thick_train = self.thickness_train['rthick_regular']
        left_thick_test = self.thickness_test['lthick_regular']
        right_thick_test = self.thickness_test['rthick_regular']

        thick_train = {'left': left_thick_train, 'right': right_thick_train}
        thick_test = {'left': left_thick_test, 'right': right_thick_test}

        print("Generating original thickness data finished...")

        return demo_train, demo_test, thick_train, thick_test