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
    "ubi": {"csv": "ubi", "name": "UBI", "color": "purple", "linestyle": "-"},
}


def _load_data(country, method, geo_extrapolation):
    """
    Load the data for a specific country and method.

    :param country: The name of the country.
    :type country: str
    :param method: The method to load data for.
    :type method: str
    :return: A DataFrame containing the data for the specified country and method.
    :rtype: pandas.DataFrame
    """
    if geo_extrapolation:
        subfolder = "geo_extrapolation"
    else:
        subfolder = "geo_interpolation"
    if method not in ["oracle_gap", "pmt", "ubi"]:
        name = METHODS[method]["csv"].split("output_")[1]
        df = pd.read_csv(
            "results/{}/{}/{}.csv".format(
                country, subfolder, f"output_{subfolder}_{name}"
            )
        )
    else:
        df = pd.read_csv(
            "results/{}/{}/{}.csv".format(country, subfolder, METHODS[method]["csv"])
        )
    return df


def get_initial_poverty_gaps_and_rates(countries):
    initial = {}
    conversion_factors = get_conversion_factors(countries)
    for country in countries:
        df = _load_data(country, "oracle_gap", geo_extrapolation=False)
        initial[country] = {
            "gap": df["post_transfer_poverty_gap"].max() * conversion_factors[country],
            "rate": df["post_transfer_poverty_rate"].max() * 100,
        }
    return initial


def get_conversion_factors(countries):
    df = pd.read_csv("currency_conversion.csv")
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


def get_country_interpolators(countries, method, geo_extrapolation):
    country_interpolators = {}
    conversion_factors = get_conversion_factors(countries)

    for country in countries:
        df = _load_data(country, method, geo_extrapolation)
        country_conversion_factor = conversion_factors[country]

        gaps = list(df["post_transfer_poverty_gap"] * country_conversion_factor)
        # gaps.append(initial[country]["gap"] * country_conversion_factor)
        cost_gaps = list(df["policy_cost_per_capita"] * country_conversion_factor)

        country_gap_interpolator = interp1d(
            gaps, cost_gaps, kind="linear", fill_value="extrapolate"
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
            fill_value="extrapolate",
        )
        country_interpolators[country]["rate_interpolator"] = country_rate_interpolator
    return country_interpolators


def get_aggregate_interpolators_wc_poverty_measure(
    countries, method, geo_extrapolation
):
    dic = METHODS[method].copy()
    initial = get_initial_poverty_gaps_and_rates(countries)
    max_initial_poverty_gap = max([initial[country]["gap"] for country in countries])

    max_initial_poverty_rate = max([initial[country]["rate"] for country in countries])

    print("max gap", max_initial_poverty_gap)
    print("max rate", max_initial_poverty_rate)

    country_interpolators = get_country_interpolators(
        countries, method, geo_extrapolation
    )

    # Compute aggregate rate interpolator
    gaps = np.linspace(0.0, max_initial_poverty_gap, 50)
    costs = []
    for country in countries:
        country_gap_interpolator = country_interpolators[country]["gap_interpolator"]
        costs.append(np.clip(country_gap_interpolator(gaps), a_min=0.0, a_max=None))
    costs = np.array(costs)
    aggregate_gap_costs = np.sum(costs, axis=0)
    aggregate_gap_interpolator = interp1d(gaps, aggregate_gap_costs, kind="linear")

    # Compute aggregate rate interpolator
    rates = np.linspace(0.0, max_initial_poverty_rate, 50)
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


def get_country_weights(countries):
    country_weights = {}
    conversion_factors = pd.read_csv("currency_conversion.csv")
    conversion_factors["weight"] = (
        conversion_factors["total_population_survey_year"]
        / conversion_factors["total_population_survey_year"].sum()
    )
    return (
        conversion_factors[["country", "weight", "total_population_survey_year"]]
        .set_index("country")
        .to_dict()
    )


def get_aggregate_interpolators_population_weighted_poverty_measure(
    countries, method, geo_extrapolation
):
    dic = METHODS[method].copy()
    initial = get_initial_poverty_gaps_and_rates(countries)
    country_weights = get_country_weights(countries)

    pop_weighted_initial_poverty_gap = np.array(
        [initial[country]["gap"] for country in countries]
    ).sum()

    pop_weighted_initial_poverty_rate = sum(
        np.array([initial[country]["rate"] for country in countries])
        * np.array([country_weights["weight"][country] for country in countries])
    )

    max_initial_poverty_gap = max([initial[country]["gap"] for country in countries])

    max_initial_poverty_rate = max([initial[country]["rate"] for country in countries])

    print("initial gap", pop_weighted_initial_poverty_gap)
    print("initial rate", pop_weighted_initial_poverty_rate)
    print("max gap", max_initial_poverty_gap)
    print("max rate", max_initial_poverty_rate)

    country_interpolators = get_country_interpolators(
        countries, method, geo_extrapolation
    )

    gaps = np.linspace(0.0, max_initial_poverty_gap, 50)
    rates = np.linspace(0.0, max_initial_poverty_rate, 50)

    country_interpolators_wc_measure_to_actual_measure = {}
    for country in countries:
        actual_gaps = np.clip(gaps, a_min=None, a_max=initial[country]["gap"])
        actual_rates = np.clip(rates, a_min=None, a_max=initial[country]["rate"])
        country_interpolators_wc_measure_to_actual_measure[country] = {
            "gap": interp1d(gaps, actual_gaps, kind="linear"),
            "rate": interp1d(rates, actual_rates, kind="linear"),
        }

    agg_gaps = 0.0
    agg_rates = 0.0
    for country in countries:
        agg_gaps += country_interpolators_wc_measure_to_actual_measure[country]["gap"](
            gaps
        )
        agg_rates += (
            country_interpolators_wc_measure_to_actual_measure[country]["rate"](rates)
            * country_weights["weight"][country]
        )
    assert max(agg_gaps) == pop_weighted_initial_poverty_gap
    assert max(agg_rates) == pop_weighted_initial_poverty_rate

    aggregate_interpolator_wc_gap_to_actual_gap = interp1d(
        gaps, agg_gaps, kind="linear"
    )
    aggregate_interpolator_wc_rate_to_actual_rate = interp1d(
        rates, agg_rates, kind="linear"
    )

    # Compute aggregate rate interpolator

    costs = []
    for country in countries:
        country_gap_interpolator = country_interpolators[country]["gap_interpolator"]
        costs.append(np.clip(country_gap_interpolator(gaps), a_min=0.0, a_max=None))
    costs = np.array(costs)
    aggregate_gap_costs = np.sum(costs, axis=0)
    xs_gap = aggregate_interpolator_wc_gap_to_actual_gap(gaps)
    print("x range gap", xs_gap[0], xs_gap[-1])
    aggregate_gap_interpolator = interp1d(xs_gap, aggregate_gap_costs, kind="linear")

    # Compute aggregate rate interpolator

    rate_costs = []
    for country in countries:
        country_rate_interpolator = country_interpolators[country]["rate_interpolator"]
        rate_costs.append(
            np.clip(country_rate_interpolator(rates), a_min=0.0, a_max=None)
        )
    rate_costs = np.array(rate_costs)
    aggregate_rate_costs = np.sum(rate_costs, axis=0)
    xs_rate = aggregate_interpolator_wc_rate_to_actual_rate(rates)
    aggregate_rate_interpolator = interp1d(xs_rate, aggregate_rate_costs, kind="linear")

    return {
        "gap": {
            "interpolator": aggregate_gap_interpolator,
            "range": (0.0, pop_weighted_initial_poverty_gap),
        },
        "rate": {
            "interpolator": aggregate_rate_interpolator,
            "range": (0.0, pop_weighted_initial_poverty_rate),
        },
    }
