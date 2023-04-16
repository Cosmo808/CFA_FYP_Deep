from torch.utils import data


class Dataset_starmen(data.Dataset):
    def __init__(self, image_path, subject, baseline_age, age, timepoint, first_age, alpha):
        self.image_path = image_path
        self.subject = subject
        self.baseline_age = baseline_age
        self.age = age
        self.timepoint = timepoint
        self.first_age = first_age
        self.alpha = alpha

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        """Generates one sample of data"""
        # Select sample
        x = self.image_path[index]
        y = self.subject[index]
        z = self.baseline_age[index]
        u = self.age[index]
        v = self.timepoint[index]
        w = self.first_age[index]
        a = self.alpha[index]
        return x, y, z, u, v, w, a


class Dataset_adni(data.Dataset):
    def __init__(self, left_thickness, right_thickness, age, baseline_age, label, subject, timepoint):
        self.lthick = left_thickness
        self.rthick = right_thickness
        self.age = age
        self.baseline_age = baseline_age
        self.label = label
        self.subject = subject
        self.timepoint = timepoint

    def __len__(self):
        return len(self.age)

    def __getitem__(self, index):
        a = self.lthick[index]
        b = self.rthick[index]
        c = self.age[index]
        d = self.baseline_age[index]
        e = self.label[index]
        f = self.subject[index]
        g = self.timepoint[index]
        return a, b, c, d, e, f, g