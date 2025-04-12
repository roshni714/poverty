import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

METHODS = {
    "oracle_gap": {
        "csv": "oracle_gap",
        "name": "Oracle Poverty Gap Targeting",
        "color": "green",
        "linestyle": "-",
    },
    "continuous_gap": {
        "csv": "output_gt_continuous_gap",
        "name": "Poverty Gap Targeting (Continuous-Valued)",
        "color": "blue",
        "linestyle": "-",
    },
    "binary_gap": {
        "csv": "output_gt_binary_gap",
        "name": "Poverty Gap Targeting (Binary-Valued)",
        "color": "blue",
        "linestyle": "--",
    },
    "continuous_rate": {
        "csv": "output_gt_continuous_rate",
        "name": "Poverty Rate Targeting (Continuous-Valued)",
        "color": "orange",
        "linestyle": "-",
    },
    "binary_rate": {
        "csv": "output_gt_binary_rate",
        "name": "Poverty Rate Targeting (Binary-Valued)",
        "color": "orange",
        "linestyle": "--",
    },
    "pmt": {"csv": "pmt", "name": "PMT", "color": "red", "linestyle": "-"},
}


def _load_data(country, method):
    """
    Load the data for a specific country and method.

    :param country: The name of the country.
    :type country: str
    :param method: The method to load data for.
    :type method: str
    :return: A DataFrame containing the data for the specified country and method.
    :rtype: pandas.DataFrame
    """
    df = pd.read_csv("results/{}/{}.csv".format(country, METHODS[method]["csv"]))
    return df


def get_initial_poverty_gaps_and_rates(countries):
    max_pre_transfer_poverty_gap = 0.0
    max_pre_transfer_poverty_rate = 0.0
    initial = {}
    for country in countries:
        df = _load_data(country, "oracle_gap")
        max_pre_transfer_poverty_gap = max(
            max_pre_transfer_poverty_gap, df["post_transfer_poverty_gap"].max()
        )
        max_pre_transfer_poverty_rate = max(
            max_pre_transfer_poverty_rate, df["post_transfer_poverty_rate"].max()
        )
        initial[country] = {
            "gap": df["post_transfer_poverty_gap"].max(),
            "rate": df["post_transfer_poverty_rate"].max(),
        }
    return max_pre_transfer_poverty_gap, max_pre_transfer_poverty_rate, initial


def get_conversion_factors(countries):
    df = pd.read_csv("conversion_factors.csv")
    conversion_factors = {}
    for country in countries:
        country_df = df[df["country"] == country]
        factor = (
            country_df["total_population_survey_year"].values[0]
            * 365
            * 1.31
            * (
                country_df["PPP_exchange_rate_2017"].values[0]
                / country_df["market_exchange_rate_2017"].values[0]
            )
            / 1000000000
        )
        conversion_factors[country] = factor
    return conversion_factors


def get_aggregate_interpolators(countries, method):

    dic = METHODS[method].copy()

    # Load conversion factors for all countries
    # TODO

    max_pre_transfer_poverty_gap, max_pre_transfer_poverty_rate, initial = (
        get_initial_poverty_gaps_and_rates(countries)
    )
    conversion_factors = get_conversion_factors(countries)

    country_interpolators = {}
    for country in countries:
        df = _load_data(country, method)
        # TODO compute country conversion factor
        country_conversion_factor = conversion_factors[country]

        gaps = list(df["post_transfer_poverty_gap"] * country_conversion_factor)
        # gaps.append(initial[country]["gap"] * country_conversion_factor)
        cost_gaps = list(df["policy_cost_per_capita"] * country_conversion_factor)

        if country == "ethiopia" and method == "pmt":
            print(gaps)
            print(cost_gaps)

        country_gap_interpolator = interp1d(
            gaps,
            cost_gaps,
            kind="linear",
            bounds_error=False,
            fill_value=(2.15 * country_conversion_factor, 0.0),
        )

        country_interpolators[country] = {}
        country_interpolators[country]["gap_interpolator"] = country_gap_interpolator

        rates = list(df["post_transfer_poverty_rate"] * 100)
        # rates.append(initial[country]["rate"] * country_conversion_factor)
        cost_rates = list(df["policy_cost_per_capita"] * country_conversion_factor)
        # cost_rates.append(0.0)
        country_rate_interpolator = interp1d(
            rates,
            cost_rates,
            kind="linear",
            bounds_error=False,
            fill_value=(2.15 * country_conversion_factor, 0.0),
        )
        country_interpolators[country]["rate_interpolator"] = country_rate_interpolator

    # Compute aggregate rate interpolator
    gaps = np.linspace(
        0.0, max_pre_transfer_poverty_gap * country_conversion_factor, 50
    )
    costs = []
    for country in countries:
        country_gap_interpolator = country_interpolators[country]["gap_interpolator"]
        costs.append(np.clip(country_gap_interpolator(gaps), a_min=0.0, a_max=None))
    costs = np.array(costs)
    aggregate_gap_costs = np.sum(costs, axis=0)
    aggregate_gap_interpolator = interp1d(gaps, aggregate_gap_costs, kind="linear")

    # Compute aggregate rate interpolator
    rates = np.linspace(0.0, max_pre_transfer_poverty_rate * 100, 50)
    rate_costs = []
    for country in countries:
        country_rate_interpolator = country_interpolators[country]["rate_interpolator"]
        rate_costs.append(
            np.clip(country_rate_interpolator(rates), a_min=0.0, a_max=None)
        )
    rate_costs = np.array(rate_costs)
    aggregate_rate_costs = np.sum(rate_costs, axis=0)
    aggregate_rate_interpolator = interp1d(rates, aggregate_rate_costs, kind="linear")

    return {
        "gap": {"interpolator": aggregate_gap_interpolator, "range": (0, gaps[-1])},
        "rate": {"interpolator": aggregate_rate_interpolator, "range": (0, rates[-1])},
    }
