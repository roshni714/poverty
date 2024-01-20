import numpy as np


def standardize(z):
    z_mean = z.mean(axis=0)
    z_std = z.std(axis=0)
    data = (z - z_mean) / z_std
    return data, z_mean, z_std


class Dataset:
    def __init__(self, X, y, r=None):
        self.X = X
        self.y = y
        if r is None:
            self.r = np.ones(y.shape) / len(y)
        else:
            self.r = r

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i, :], self.y[i], self.r[i]


def split_data(X, y, p, d, r=None):
    if r is None:
        r = np.ones(y.shape[0]) / len(y)

    rng = np.random.RandomState(123456)
    permutation = rng.permutation(X.shape[0])
    index_train = permutation[: int(p * X.shape[0])]
    index_test = permutation[int(p * X.shape[0]) :]
    X_train = X[index_train, :d]
    X_test = X[index_test, :d]
    y_train = y[index_train]
    y_test = y[index_test]
    r_train = r[index_train]
    r_test = r[index_test]
    return Dataset(X_train, y_train, r_train), Dataset(X_test, y_test, r_test)
