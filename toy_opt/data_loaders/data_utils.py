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


def split_data(X, y, p, r=None):
    # Truncate and drop
    #    idx = np.where(np.logical_and(y >= outcome_range[0], y <= outcome_range[1]))
    #    y = y[idx]
    #    X = X[idx, :].squeeze()
    #    if r is not None:
    #        r = r[idx]

    #    y = np.clip(y, a_min=outcome_range[0], a_max=outcome_range[1])

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

    return Dataset(X_train, y_train, r=r_train), Dataset(X_test, y_test, r=r_test)
