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
        z_std[np.where(z_std == 0.0)[0]] = 1e-5
    elif z_std == 0.0:
        z_std = 1e-5

    data = (z - z_mean) / z_std
    return data, z_mean, z_std


class Dataset:
    def __init__(self, X, y=None, r=None, normalize_weight_sum=True):
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
            r = np.ones(len(X))

        if normalize_weight_sum:
            self.r = r / r.sum()

        else:
            self.r = r

            
    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return self.X.shape[0]

    def __getitem__(self, i):
        """
        Get the i-th sample from the dataset.

        :param i: The index of the sample.
        :type i: int
        :return: The i-th sample as a tuple (X_i, y_i, r_i) if y is not None, else (X_i, r_i).
        :rtype: tuple
        """
        if self.y is not None:
            return self.X[i, :], self.y[i], self.r[i]
        else:
            return self.X[i, :], self.r[i]
