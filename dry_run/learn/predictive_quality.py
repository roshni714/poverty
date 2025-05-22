from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from learn.data_loader import load_datasets
import numpy as np


def m(x, w):
    """Weighted Mean"""
    return np.sum(x * w) / np.sum(w)


def cov(x, y, w):
    """Weighted Covariance"""
    return np.sum(w * (x - m(x, w)) * (y - m(y, w))) / np.sum(w)


def corr(x, y, w):
    """Weighted Correlation"""
    return cov(x, y, w) / np.sqrt(cov(x, x, w) * cov(y, y, w))


def get_out_of_sample_r2(country, upper_val=10.0):
    train_dataset, validation_dataset, test_covariate_dataset, test_dataset = (
        load_datasets(
            f"data/{country}/train.parquet",
            f"data/{country}/test.parquet",
            f"data/{country}/summary.parquet",
            True,
            "consumption_per_capita_per_day",
            "headcount_adjusted_hh_wgt",
        )
    )

    X_train, y_train, r_train = train_dataset.get_data(normalize_weight=False)
    X_val, y_val, r_val = validation_dataset.get_data(normalize_weight=False)
    X_combined = np.vstack((X_train, X_val))
    y_combined = np.hstack((y_train, y_val))
    idx = y_combined <= upper_val
    r_combined = np.hstack((r_train, r_val))
    r_combined /= r_combined[idx].sum()

    # Fit a linear regression model
    model = LinearRegression(fit_intercept=True)
    model.fit(X_combined[idx], y_combined[idx], sample_weight=r_combined[idx])
    # Calculate R^2 on the test set
    X_test, y_test, r_test = test_dataset.get_data(normalize_weight=False)
    idx_test = y_test <= upper_val
    r_test /= r_test[idx_test].sum()
    y_pred = model.predict(X_test[idx_test])
    r = corr(y_test[idx_test], y_pred, r_test[idx_test])
    r2 = r**2

    # print(country, r2)

    return r2
