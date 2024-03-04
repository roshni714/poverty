import numpy as np


def standardize(z):
    z_mean = z.mean(axis=0)
    z_std = z.std(axis=0)

    if isinstance(z_std, np.ndarray):
        z_std[np.where(z_std == 0.0)[0]] = 1e-5
    elif z_std == 0.0:
        z_std = 1e-5

    data = (z - z_mean) / z_std
    return data, z_mean, z_std


class Dataset:
    def __init__(self, X, y, r=None):
        self.X = X
        self.y = y

        if r is None:
            self.r = np.ones(self.y.shape) / len(self.y)
        else:
            self.r = r / r.sum()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i, :], self.y[i], self.r[i]
