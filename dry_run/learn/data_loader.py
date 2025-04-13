import pandas as pd
import numpy as np

from opt_targeted_transfers import Dataset, split


def get_training_data_for_geo_extrapolation(data, summary, seed=1537498):
    """
    Preprocess the training data for geo-extrapolation.

    Args:
        data (pd.DataFrame): The input data.
        summary (pd.DataFrame): The summary data.

    Returns:
        pd.DataFrame: The preprocessed data without geographic identifiers only for a subset of geographic regions.
    """

    geo_cols = summary[summary["geographic_indicator"] == True][
        "variable_name"
    ].tolist()

    geo_col_unique_counts = {col: data[col].nunique() for col in geo_cols}
    finest_geo_col = max(geo_col_unique_counts, key=geo_col_unique_counts.get)
    n_unique = geo_col_unique_counts[finest_geo_col]
    n_subset = int(0.75 * n_unique)
    rng = np.random.default_rng(seed)
    train_geo_col = rng.choice(data[finest_geo_col].unique(), n_subset, replace=False)
    data = data[data[finest_geo_col].isin(train_geo_col)].reset_index(drop=True)
    data.drop(columns=geo_cols, inplace=True)
    return data


def get_testing_data_for_geo_extrapolation(data, summary):
    """
    Preprocess the testing data for geo-extrapolation.

    Args:
        data (pd.DataFrame): The input data.
        summary (pd.DataFrame): The summary data.

    Returns:
        pd.DataFrame: The preprocessed data without geographic identifiers
    """

    geo_cols = summary[summary["geographic_indicator"] == True][
        "variable_name"
    ].tolist()
    data.drop(columns=geo_cols, inplace=True)
    return data


def load_datasets(trainpath, testpath, summarypath, geo_extrapolation, outcome, weight):
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

    if geo_extrapolation:
        data1 = get_training_data_for_geo_extrapolation(data1, summary)
        data2 = get_testing_data_for_geo_extrapolation(data2, summary)

    all_data = pd.concat([data1, data2], ignore_index=True)
    all_data = convert_to_onehot(all_data, summary)

    train_data = _load_data(trainpath)
    test_data = _load_data(testpath)
    if geo_extrapolation:
        train_data = get_training_data_for_geo_extrapolation(train_data, summary)
        test_data = get_testing_data_for_geo_extrapolation(test_data, summary)
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

    train_dataset = Dataset(
        final_train_data.astype("float32"), outcome=outcome, covs=covs, weight=weight
    )
    test_dataset = Dataset(
        final_test_data.astype("float32"), outcome=outcome, covs=covs, weight=weight
    )
    test_covariate_dataset = Dataset(
        final_test_data.astype("float32"), outcome=None, covs=covs, weight=weight
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
    if "type" in summary.columns:
        data_type = "type"
    elif "data_type" in summary.columns:
        data_type = "data_type"
    if "covariate" in summary.columns:
        covariate = "covariate"
    elif "variable_name" in summary.columns:
        covariate = "variable_name"

    categorical_columns = summary[summary[data_type] == "categorical"][
        covariate
    ].tolist()

    categorical_columns = [col for col in categorical_columns if col in df.columns]

    one_hot = pd.get_dummies(df[categorical_columns]).astype(np.float32)
    df.drop(columns=categorical_columns, inplace=True)
    new_df = pd.concat([df, one_hot], axis=1)
    return new_df


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

    if "hhid" in data.columns:
        data = data.drop(columns=["hhid"])
    if "case_id" in data.columns:
        data = data.drop(columns=["case_id"])
    if "hh_id" in data.columns:
        data = data.drop(columns=["hh_id"])

    return data.reset_index(drop=True)
