import numpy as np
from opt_targeted_transfers.dataset_utils import standardize
from sklearn.linear_model import Ridge


def get_pmt_linear_regressor(train_dataset, validation_dataset):
    # revise this to use cross validation in future?
    X_train, y_train, r_train = train_dataset.get_data()
    X_val, y_val, r_val = validation_dataset.get_data()

    X = np.concatenate([X_train, X_val], axis=0)
    y = np.concatenate([y_train, y_val], axis=0)
    r = np.concatenate([r_train, r_val], axis=0)

    X, X_mean, X_std = standardize(X)
    y, y_mean, y_std = standardize(y)

    model = Ridge(fit_intercept=True, alpha=0.1)
    model.fit(X, y, sample_weight=r)

    def estimator(X_test):
        X_test = (X_test - X_mean) / X_std
        predicted_consumption = (
            model.predict(X_test).reshape(X_test.shape[0], 1) * y_std + y_mean
        )

        return predicted_consumption.flatten()

    return estimator
