import pandas as pd


def get_data_dimension(country):
    """
    Get the number of samples and features for a given country's dataset.
    Args:
        country (str): Country code.
    Returns:
        n (int): Total number of samples (train + test).
        d (int): Number of features.
    """
    train_data = pd.read_parquet("data/{}/train.parquet".format(country))
    n_train = len(train_data)
    test_data = pd.read_parquet("data/{}/test.parquet".format(country))
    n_test = len(test_data)
    n = n_train + n_test
    d = len(train_data.columns)
    # remove hh_wgt, household_size_adjusted_hh_wgt, and consumption_per_capita_per_day
    d -= 3
    return n, d


def convert_nominal_2023_to_nominal_survey_year(amt, country, metadata):
    """
    Convert an amount in 2023 nominal terms to nominal terms for the survey year of the given country.
    Args:
        amt (float): Amount in 2023 nominal terms.
        country (str): Country code.
    Returns:
        float: Amount in nominal terms for the survey year.
    """
    df = metadata.preprocess_country_aux_data()
    second_df = metadata.preprocess_secondary_aux_data()
    survey_year = int(df[df["country_code"] == country]["survey_year"].values[0])
    inflation_adjustment = second_df[
        second_df["indicator"]
        == "conversion_factor_nominal_USD_{}_to_2023".format(survey_year)
    ]["value"].values[0]
    amt = amt * (inflation_adjustment)
    return amt


def get_country_name(code, metadata):
    """
    Get the country name given its country code.
    Args:
        code (str): Country code.
    Returns:
        str: Country name.
    """
    wpc_aux_data = metadata.preprocess_wpc_data([code])
    name = wpc_aux_data["country_name"].values[0]
    return name


def make_string_country_list(l, metadata):
    """
    Convert a list of country codes into a human-readable string of country names.
    Args:
        l (list): List of country codes.
    Returns:
        str: Human-readable string of country names.
    """
    if len(l) == 0:
        return ""
    elif len(l) == 1:
        return get_country_name(l[0], metadata)
    else:
        return (
            ", ".join([get_country_name(c, metadata) for c in l[:-1]])
            + ", and "
            + get_country_name(l[-1], metadata)
        )
