import numpy as np
import xgboost as xg
from cond_dist import ConditionalDistribution
from data_loaders.data_utils import standardize


def fit_mle(X, z, r):
    X, X_mean, X_std = standardize(X)
    z, z_mean, z_std = standardize(z)

    bst = xg.XGBRegressor(objective="reg:squarederror", max_depth=3, n_estimators=10)

    f_hat = bst.fit(X, z, sample_weight=r)
    r_sq = (z - f_hat.predict(X)) ** 2

    bst2 = xg.XGBRegressor(objective="reg:squarederror", max_depth=3, n_estimators=10)
    g_hat = bst2.fit(X, r_sq, sample_weight=r)

    return f_hat, g_hat, X_mean, X_std, z_mean, z_std


def get_cond_density_estimator(train_dataset):
    X = train_dataset.X
    y = train_dataset.y
    r = train_dataset.r

    min_y = np.min(y) - np.min(y) / 10
    z = np.log(y - min_y)
    cond_mean, cond_var, X_mean, X_std, z_mean, z_std = fit_mle(X, z, r)

    def helper(X_test):
        # accepts unstandardized inputs
        X_test = (X_test - X_mean) / X_std
        z_hat = cond_mean.predict(X_test) * z_std + z_mean
        gamma = np.exp(z_hat)
        sigma_sq_hat = np.maximum(cond_var.predict(X_test), 0.01) * (z_std**2)
        sigma = np.sqrt(sigma_sq_hat)

        estimated_test_densities = []
        for i in range(len(gamma)):
            estimated_test_densities.append(
                ConditionalDistribution(loc=min_y, shape=sigma[i], scale=gamma[i])
            )
        return np.array(estimated_test_densities)

    return helper
