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
    def __init__(self, df, outcome, covs=None, weight=None):
        """
        Initialize a Dataset object.

        :param df: The input data.
        :type df: pandas.DataFrame
        :param outcome: The name of the target column.
        :type outcome: str
        :param covs: The names of the input features.
        :type covs: list
        :param weight: The name of the weight column.
        :type weight: str
        """
        self.df = df
        self.outcome = outcome
        self.weight = weight
        self.covs = covs

    def get_data(self, normalize_weight=True):
        """
        Get the input features, target values, and weights.

        :return: The input features, target values, and weights.
        :rtype: tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray)
        """
        if self.covs is None:
            X = self.df[self.covs].values
        else:
            X = self.df[self.covs].values.reshape(len(self.df), len(self.covs))
        y = self.df[self.outcome].values
        if self.weight is None:
            r = np.ones(y.shape)
        else:
            r = self.df[self.weight].values

        if normalize_weight:
            r = r / r.sum()

        return X, y, r
