from torch.utils import data


class Dataset_starmen(data.Dataset):
    def __init__(self, image_path, subject, baseline_age, age, timepoint, first_age):
        self.image_path = image_path
        self.subject = subject
        self.baseline_age = baseline_age
        self.age = age
        self.timepoint = timepoint
        self.first_age = first_age

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
        return x, y, z, u, v, w