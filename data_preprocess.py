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
    def __init__(self, ratio=0.25):
        self.device = torch.device('cuda:0')
        self.ratio = ratio

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
        self.idx1_train, self.idx1_test = None, None
        self.idx2_train, self.idx2_test = None, None

    def generate_demo_train_test(self, fold):
        # load data
        age_train = torch.tensor(self.demo_train['Age'], device=self.device).float().squeeze()
        age_test = torch.tensor(self.demo_test['Age'], device=self.device).float().squeeze()
        label_train = torch.tensor(self.demo_train['Label'], device=self.device).float().squeeze()
        label_test = torch.tensor(self.demo_test['Label'], device=self.device).float().squeeze()
        timepoint_train = torch.tensor(self.demo_train['Wave'], device=self.device).float().squeeze()
        timepoint_test = torch.tensor(self.demo_test['Wave'], device=self.device).float().squeeze()
        with open('ADNI/subject_train.csv', 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            subject_train = torch.tensor([[int(cell[:3] + cell[6:]) for cell in row] for row in csvreader], device=self.device).squeeze()
        with open('ADNI/subject_test.csv', 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            subject_test = torch.tensor([[int(cell[:3] + cell[6:]) for cell in row] for row in csvreader], device=self.device).squeeze()

        # get sort data index
        idx1_train, idx1_test = timepoint_train.sort()[1], timepoint_test.sort()[1]
        sorted_train, sorted_test = subject_train[idx1_train], subject_test[idx1_test]
        idx2_train, idx2_test = sorted_train.sort()[1], sorted_test.sort()[1]
        self.idx1_train, self.idx1_test = idx1_train, idx1_test
        self.idx2_train, self.idx2_test = idx2_train, idx2_test
        # sort data
        age_train, age_test = age_train[idx1_train], age_test[idx1_test]
        label_train, label_test = label_train[idx1_train], label_test[idx1_test]
        timepoint_train, timepoint_test = timepoint_train[idx1_train], timepoint_test[idx1_test]
        subject_train, subject_test = subject_train[idx1_train], subject_test[idx1_test]

        age_train, age_test = age_train[idx2_train], age_test[idx2_test]
        label_train, label_test = label_train[idx2_train], label_test[idx2_test]
        timepoint_train, timepoint_test = timepoint_train[idx2_train], timepoint_test[idx2_test]
        subject_train, subject_test = subject_train[idx2_train], subject_test[idx2_test]

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
        baseline_age_train = torch.tensor(baseline_age_train, device=self.device).float()
        baseline_age_test = torch.tensor(baseline_age_test, device=self.device).float()

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
        if self.idx1_train is None:
            _, _ = self.generate_demo_train_test(fold)

        num = int(self.thickness_train['lthick_regular'].shape[1] * self.ratio)
        left_thick_train = torch.tensor(self.thickness_train['lthick_regular'][:, :num], device=self.device).float()
        right_thick_train = torch.tensor(self.thickness_train['rthick_regular'][:, :num], device=self.device).float()
        left_thick_test = torch.tensor(self.thickness_test['lthick_regular'][:, :num], device=self.device).float()
        right_thick_test = torch.tensor(self.thickness_test['rthick_regular'][:, :num], device=self.device).float()

        left_thick_train, right_thick_train = left_thick_train[self.idx1_train], right_thick_train[self.idx1_train]
        left_thick_test, right_thick_test = left_thick_test[self.idx2_test], right_thick_test[self.idx2_test]
        thick_train = {'left': left_thick_train, 'right': right_thick_train}
        thick_test = {'left': left_thick_test, 'right': right_thick_test}

        print('Generating thickness data finished...')
        if fold == 0:
            return thick_train, thick_test, num
        else:
            return thick_test, thick_train, num

    def generate_XY(self, data):
        N = data['age'].size()[0]
        I = len(torch.unique(data['subject']))

        delta_age = (data['age'] - data['baseline_age']).view(N, -1)
        ones = torch.ones(size=delta_age.size(), device=self.device)
        X = torch.cat((ones, delta_age, data['baseline_age'].view(N, -1)), dim=1)

        Y, old_s, cnt_zero = None, None, 0
        for i in range(N):
            if old_s is None:
                old_s = data['subject'][i]
            elif old_s != data['subject'][i]:
                old_s = data['subject'][i]
                cnt_zero += 1

            zeros0 = torch.zeros(size=[1, 2 * cnt_zero], device=self.device)
            zeros1 = torch.zeros(size=[1, 2 * (I - 1 - cnt_zero)], device=self.device)
            yy = X[i, :2].view(1, 2)
            yy = torch.cat((zeros0, yy, zeros1), dim=1)

            if Y is None:
                Y = yy
            else:
                Y = torch.cat((Y, yy), dim=0)

        print('Generating X and Y finished...')
        return X, Y
