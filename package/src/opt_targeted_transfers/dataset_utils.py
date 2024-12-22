import numpy as np


def standardize(z):
    """
    Standardize a numpy array.

    :param z: The input array to be standardized.
    :type z: numpy.ndarray
    :return: A tuple containing the standardized array, mean array, and standard deviation array.
    :rtype: tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray)
    """
    z_mean = z.mean(axis=0)
    z_std = z.std(axis=0)

    if isinstance(z_std, np.ndarray):
        z_std[np.where(z_std == 0.0)[0]] = 1.0
    elif z_std == 0.0:
        z_std = 1.0

    data = (z - z_mean) / z_std
    return data, z_mean, z_std


class Dataset:
    def __init__(self, X=None, y=None, r=None):
        """
        Initialize a Dataset object.

        :param X: The input features.
        :type X: numpy.ndarray
        :param y: The target values. Defaults to None.
        :type y: numpy.ndarray or None
        :param r: The weights. Defaults to uniform weights if r is None.
        :type r: numpy.ndarray or None
        """
        self.X = X
        self.y = y

        if r is None:
            self.r = np.ones(len(X)) / len(self.X)
        else:
            self.r = r / r.sum()

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        if self.X is not None:
            return self.X.shape[0]
        elif self.y is not None:
            return self.y.shape[0]

    def __getitem__(self, i):
        """
        Get the i-th sample from the dataset.

        :param i: The index of the sample.
        :type i: int
        :return: The i-th sample as a tuple (X_i, y_i, r_i) if y is not None, else (X_i, r_i).
        :rtype: tuple
        """
        if self.X is not None and self.y is not None:
            return self.X[i, :], self.y[i], self.r[i]
        elif self.X is not None:
            return self.y[i], self.r[i]
        elif self.y is not None:
            return self.X[i, :], self.r[i]
