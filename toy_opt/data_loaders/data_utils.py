import numpy as np


def standardize(z):
    z_mean = z.mean(axis=0)
    z_std = z.std(axis=0)
    data = (z - z_mean) / z_std
    return data, z_mean, z_std


class Dataset:
    def __init__(self, X, y, d, r=None):
        self.X = X[:, :d]
        self.y = y
        self.d = d
        self.full_X = X
        if r is None:
            self.r = np.ones(y.shape) / len(y)
        else:
            self.r = r / r.sum()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i, : self.d], self.y[i], self.r[i]


def split_data(X, y, d, p, r=None):
    rng = np.random.RandomState(123456)
    permutation = rng.permutation(X.shape[0])
    index_train = permutation[: int(p * X.shape[0])]
    index_test = permutation[int(p * X.shape[0]) :]
    X_train = X[index_train]
    X_test = X[index_test]
    y_train = y[index_train]
    y_test = y[index_test]
    if r is not None:
        r_train = r[index_train]
        r_test = r[index_test]
    else:
        r_train = None
        r_test = None

    return Dataset(X_train, y_train, d=d, r=r_train), Dataset(
        X_test, y_test, d=d, r=r_test
    )
