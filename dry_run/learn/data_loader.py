import pandas as pd
import numpy as np

from opt_targeted_transfers import Dataset, split

AUX_DATA_CSV = "/data/eop/compiled_country_data/auxiliary_data_20251014.csv"

def load_datasets(trainpath, testpath, summarypath, outcome, weight):
    """
    Load datasets.

    Args:
        trainpath (str): Path to the training data file.
        testpath (str): Path to the test data file.
        outcome (str): Outcome variable.
        weight (str): Weight variable.

    Returns:
        train_dataset (Dataset): Training dataset.
        test_dataset (Dataset): Test dataset.
    """
    data1 = _load_data(trainpath)
    data2 = _load_data(testpath)
    summary = pd.read_parquet(summarypath)
    all_data = pd.concat([data1, data2], ignore_index=True)
    all_data = convert_to_onehot(all_data, summary)

    train_data = _load_data(trainpath)
    test_data = _load_data(testpath)
    covs = list(train_data.columns)
    covs.remove(outcome)
    covs.remove(weight)

    train_data = convert_to_onehot(train_data, summary)
    test_data = convert_to_onehot(test_data, summary)

    train_missing_columns = set(all_data.columns) - set(train_data.columns)
    res = [train_data]
    for col in train_missing_columns:
        res.append(pd.DataFrame({col: np.zeros(len(train_data))}))
    final_train_data = pd.concat(res, axis=1)

    test_missing_columns = set(all_data.columns) - set(test_data.columns)
    res = [test_data]
    for col in test_missing_columns:
        res.append(pd.DataFrame({col: np.zeros(len(test_data))}))
    final_test_data = pd.concat(res, axis=1)

    train_dataset = Dataset(final_train_data, outcome=outcome, covs=covs, weight=weight)
    test_dataset = Dataset(final_test_data, outcome=outcome, covs=covs, weight=weight)
    test_covariate_dataset = Dataset(
        final_test_data, outcome=None, covs=covs, weight=weight
    )

    train_dataset, validation_dataset = split(train_dataset)
    return train_dataset, validation_dataset, test_covariate_dataset, test_dataset


def convert_to_onehot(df, summary):
    """
    Convert categorical columns to one-hot encoding.

    :param df: The input data.
    :type df: pandas.DataFrame
    :return new_df: The input data with one-hot encoding.
    :rtype: pandas.DataFrame
    """
    categorical_columns = summary[summary["type"] == "categorical"][
        "covariate"
    ].tolist()

<<<<<<< Updated upstream
    one_hot = pd.get_dummies(df[categorical_columns]).astype(np.float32)
    df.drop(columns=categorical_columns, inplace=True)
    new_df = pd.concat([df, one_hot], axis=1)
    return new_df
=======
    categorical_columns = [col for col in categorical_columns if col in df.columns]
    if len(categorical_columns) > 0:

        one_hot = pd.get_dummies(df[categorical_columns]).astype(np.float32)
        df.drop(columns=categorical_columns, inplace=True)
        new_df = pd.concat([df, one_hot], axis=1)
        return new_df
    else:
        return df
>>>>>>> Stashed changes


def _load_data(path):
    """
    Load data.

    Args:
        path (str): Path to the data file.

    Returns:
        data_for_wgan (pd.DataFrame): Data for WGAN training.
        data_wrapper (wgan.DataWrapper): DataWrapper object for WGAN training.
    """
    data = pd.read_parquet(path)

    # some of this preprocessing code should eventually be deprecated because
    # it should be handled by prior data preprocessing code

    # compute outcome conversion factor
    # a = 340.2 / 430.05  # Malawi CPI in 2017 USD / Malawi CPI in 2019 USD
    # b = 241.98  # Malawi Kwacha to USD exchange rate in 2017
    # adulteq = data["adulteq"]
    # # can alternatively implement this as data["num_adults"] + alpha * data["num_children"]
    # # where alpha is in (0, 1).
    # conversion_factor = (a / b) * (1 / 365) * (1 / adulteq)
    # data["consumption_per_capita_per_day"] = data["rexpagg"] * conversion_factor
    # data["consumption_per_capita_per_day"] = np.clip(
    #     data["consumption_per_capita_per_day"], 0, truncation_upper_value
    # )

    # we include hh_wgt and consumption_per_capita_per_day so that
    # we can synthetically generate samples from the joint distribution (X, Y, R)
    # durable_verifiable_covariates = list(
    #     pd.read_csv("data/durable_verifiable_covariates.csv")["Covariates"]
    # )

    # data = data[
    #     durable_verifiable_covariates + ["consumption_per_capita_per_day", "hh_wgt"]
    # ]
    return data.reset_index(drop=True)
