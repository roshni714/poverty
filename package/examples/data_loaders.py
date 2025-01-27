import pandas as pd
import numpy as np

PATH_TO_TRAIN_DATA = (
    "~/zfs/gsb/intermediate-yens/rsahoo/poverty/dry_run/data/train.parquet"
)
PATH_TO_TEST_DATA = (
    "~/zfs/gsb/intermediate-yens/rsahoo/poverty/dry_run/data/test.parquet"
)
PATH_TO_SUMMARY = (
    "~/zfs/gsb/intermediate-yens/rsahoo/poverty/dry_run/data/summary_2019.parquet"
)


def load_data(path):
    """
    Load data and add missing columns to dataset.

    :param path: The path to the data file.
    :type path: str
    :return: The data with missing columns added.
    :rtype: pandas.DataFrame
    """
    summary = pd.read_parquet(PATH_TO_SUMMARY)
    data = _load_data(path)
    data = convert_to_onehot(data, summary)
    data1 = _load_data(PATH_TO_TRAIN_DATA)
    data2 = _load_data(PATH_TO_TEST_DATA)
    all_data = pd.concat([data1, data2], ignore_index=True)
    all_data = convert_to_onehot(all_data, summary)
    missing_columns = set(all_data.columns) - set(data.columns)
    res = [data]
    for col in missing_columns:
        res.append(pd.DataFrame({col: np.zeros(len(data))}))
    final = pd.concat(res, axis=1)
    return final


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

    # some of this preprocessing code should eventually be deprecated because
    # it should be handled by prior data preprocessing code

    # compute outcome conversion factor
    # a = 340.2 / 430.05  # Malawi CPI in 2017 USD / Malawi CPI in 2019 USD
    # b = 241.98  # Malawi Kwacha to USD exchange rate in 2017
    # adulteq = data["adulteq"]
    # can alternatively implement this as data["num_adults"] + alpha * data["num_children"]
    # where alpha is in (0, 1).
    # conversion_factor = (a / b) * (1 / 365) * (1 / adulteq)
    # data["consumption_per_capita_per_day"] = data["rexpagg"] * conversion_factor
    # data["consumption_per_capita_per_day"] = np.clip(
    #     data["consumption_per_capita_per_day"], 0, truncation_upper_value
    # )

    # we include hh_wgt and consumption_per_capita_per_day so that
    # we can synthetically generate samples from the joint distribution (X, Y, R)
    # durable_verifiable_covariates = list(
    #    pd.read_csv(PATH_TO_DURABLE_VERIFIABLE)["Covariates"]
    # )

    # data = data[
    #    durable_verifiable_covariates + ["consumption_per_capita_per_day", "hh_wgt"]
    # ]
    return data.reset_index(drop=True)
