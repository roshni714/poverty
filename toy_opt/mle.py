import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from cond_dist import ConditionalDistribution
from sim_data_gen import generate_homoscedastic_data


def standardize(z):
    z_mean = z.mean(axis=0)
    z_std = z.std(axis=0)
    data = (z - z_mean) / z_std
    return data, z_mean, z_std


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
    sigma_sq_hat = cond_var.predict(X) * (z_std**2)
    sigma = np.sqrt(sigma_sq_hat)

    estimated_densities = []
    for i in range(len(gamma)):
        estimated_densities.append(
            ConditionalDistribution(loc=0.0, shape=sigma[i], scale=gamma[i])
        )
    return estimated_densities


if __name__ == "__main__":
    X, y, true_cond_densities = generate_homoscedastic_data(1000, 2)

    estimated_cond_densities = get_estimated_cond_densities(X, y)

    for i in range(100):
        print("loc", true_cond_densities[i].scale, estimated_cond_densities[i].scale)
        print("shape", true_cond_densities[i].shape, estimated_cond_densities[i].shape)
