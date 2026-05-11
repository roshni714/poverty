import numpy as np
import pandas as pd


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


def split(dataset, frac=0.7, seed=0):
    """
    Split a dataset into two parts.

    :param dataset: The input dataset.
    :type dataset: Dataset
    :param frac: The fraction of the dataset to be used for training.
    :type frac: float
    """
    n = len(dataset)
    np.random.seed(seed)
    idx = np.random.permutation(n)
    train_idx = idx[: int(frac * n)]
    test_idx = idx[int(frac * n) :]

    weight = dataset.weight
    train_dataset = Dataset(
        dataset.df.iloc[train_idx].copy(deep=True).reset_index(drop=True),
        outcome=dataset.outcome,
        covs=dataset.covs,
        weight=dataset.weight,
    )
    train_dataset.df[weight] = train_dataset.df[weight] / train_dataset.df[weight].sum()
    test_dataset = Dataset(
        dataset.df.iloc[test_idx].copy(deep=True).reset_index(drop=True),
        outcome=dataset.outcome,
        covs=dataset.covs,
        weight=dataset.weight,
    )
    test_dataset.df[weight] = test_dataset.df[weight] / test_dataset.df[weight].sum()

    return train_dataset, test_dataset

def bootstrap_subsample(dataset, frac=0.7, seed=0):
    n = int(len(dataset) * frac)
    np.random.seed(seed)
    indices = np.random.choice(n, size=n, replace=True)
    weight = dataset.weight
    subsampled_dataset = Dataset(
        dataset.df.iloc[indices].copy(deep=True).reset_index(drop=True),
        outcome=dataset.outcome,
        covs=dataset.covs,
        weight=dataset.weight,
    )
    subsampled_dataset.df[weight] = subsampled_dataset.df[weight] / subsampled_dataset.df[weight].sum()
    return subsampled_dataset


class Dataset:
    def __init__(self, df, outcome=None, covs=None, weight=None):
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
        self.covs = sorted(covs)

    def get_data(self, normalize_weight=True):
        """
        Get the input features, target values, and weights.

        :return: The input features, target values, and weights.
        :rtype: tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray)
        """
        if self.covs is None:
            X = self.df[self.covs].values
        else:
            self.covs = sorted(self.covs)
            selected_columns = [
                col
                for col in self.df.columns
                if any(col.startswith(var) for var in self.covs)
            ]
            X = self.df[sorted(selected_columns)].values
        if self.weight is None:
            r = np.ones(y.shape)
        else:
            r = self.df[self.weight].values

        if normalize_weight:
            r = r / r.sum()

        if self.outcome is None:
            return X, r
        else:
            y = self.df[self.outcome].values
            return X, y, r

    def __len__(self):
        return len(self.df)
