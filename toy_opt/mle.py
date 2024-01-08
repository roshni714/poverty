import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from cond_dist import ConditionalDistribution
from sim_data_gen import generate_homoscedastic_data
from utils import standardize


def fit_mle(X, z):
    X, X_mean, X_std = standardize(X)
    z, z_mean, z_std = standardize(z)
    f_hat = RandomForestRegressor(max_depth=2 * X.shape[1]).fit(X, z)
    r_sq = (z - f_hat.predict(X)) ** 2
    g_hat = LinearRegression().fit(X, r_sq)

    return f_hat, g_hat, X_mean, X_std, z_mean, z_std


def get_estimated_cond_densities(X, y):
    z = np.log(y)
    cond_mean, cond_var, X_mean, X_std, z_mean, z_std = fit_mle(X, z)

    X = (X - X_mean) / X_std
    z_hat = cond_mean.predict(X) * z_std + z_mean
    gamma = np.exp(z_hat)
    sigma_sq_hat = np.maximum(cond_var.predict(X), 0.01) * (z_std**2)
    sigma = np.sqrt(sigma_sq_hat)

    estimated_densities = []
    for i in range(len(gamma)):
        estimated_densities.append(
            ConditionalDistribution(loc=0.0, shape=sigma[i], scale=gamma[i])
        )
    return estimated_densities
