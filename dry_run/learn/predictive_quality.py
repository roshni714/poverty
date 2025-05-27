from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from learn.data_loader import load_datasets
import numpy as np


def get_out_of_sample_rmse(country, upper_val=10.0):
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
    rmse = np.sqrt(np.mean((y_test[idx_test] - y_pred) ** 2))

    return rmse
