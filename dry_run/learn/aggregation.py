import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression

METHODS = {
    "oracle_gap": {
        "csv": "oracle_gap",
        "name": "Oracle Gap Targeting",
        "color": "green",
        "linestyle": "-",
    },
    "continuous_gap": {
        "csv": "output_gt_continuous_gap",
        "name": "Gap Targeting (Continuous-Valued)",
        "color": "blue",
        "linestyle": "-",
    },
    "binary_gap": {
        "csv": "output_gt_binary_gap",
        "name": "Gap Targeting (Binary-Valued)",
        "color": "blue",
        "linestyle": "--",
    },
    "continuous_rate": {
        "csv": "output_gt_continuous_rate",
        "name": "Rate Targeting (Continuous-Valued)",
        "color": "orange",
        "linestyle": "-",
    },
    "binary_rate": {
        "csv": "output_gt_binary_rate",
        "name": "Rate Targeting (Binary-Valued)",
        "color": "orange",
        "linestyle": "--",
    },
    "pmt": {"csv": "pmt", "name": "PMT (Linear)", "color": "red", "linestyle": "-"},
    "modern_pmt": {
        "csv": "output_gt_modern_pmt",
        "name": "PMT (NN)",
        "color": "red",
        "linestyle": ":",
    },
    "ubi": {
        "csv": "ubi",
        "name": "UBI (Variable)",
        "color": "purple",
        "linestyle": "-",
    },
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
    df = pd.read_csv(
        "learn/results/{}/{}/{}.csv".format(country, subfolder, METHODS[method]["csv"])
    )
    return df


def get_initial_poverty_gaps_and_rates(countries):
    initial = {}
    for country in countries:
        df = _load_data(country, "oracle_gap", geo_extrapolation=True)
        initial[country] = {
            "gap": (df["post_transfer_poverty_gap"].max() / 2.15) * 100,
            "rate": df["post_transfer_poverty_rate"].max() * 100,
        }
    return initial


def get_min_poverty_gaps_and_rates(countries, method="oracle_gap"):
    min = {}
    for country in countries:
        df = _load_data(country, method, geo_extrapolation=True)
        min[country] = {
            "gap": (df["post_transfer_poverty_gap"].min() / 2.15) * 100,
            "rate": df["post_transfer_poverty_rate"].min() * 100,
        }
    return min


def prune_results(xs, ys, val=0.0):
    mask = np.abs(xs - val) < 1e-3
    if mask.sum() <= 1:
        return xs, ys
    xs_nonzero = list(np.array(xs)[~mask])
    ys_nonzero = list(np.array(ys)[~mask])

    ys_zero = ys[mask]
    xs = xs_nonzero + [val]
    ys = ys_nonzero + [min(ys_zero)]
    return np.array(xs), np.array(ys)


def get_conversion_factors(countries):
    df = pd.read_csv("learn/eop_conversion_factor.csv")
    conversion_factors = {}
    for country in countries:
        country_df = df[df["country"] == country]
        factor = (
            country_df["total_population_survey_year"].values[0]
            * 365
            * 1.23
            * (
                country_df["PPP_conversion_factor_2017"].values[0]
                / country_df["market_exchange_rate_2017"].values[0]
            )
            / 1000000000
        )
        conversion_factors[country] = factor
    return conversion_factors


def get_country_interpolators(countries, method, geo_extrapolation):
    country_interpolators = {}
    conversion_factors = get_conversion_factors(countries)

    initial = get_initial_poverty_gaps_and_rates(countries)

    for country in countries:
        df = _load_data(country, method, geo_extrapolation)
        country_conversion_factor = conversion_factors[country]

        gaps = list(df["post_transfer_poverty_gap"] * 100 / 2.15)
        cost_gaps = list(df["policy_cost_per_capita"] * country_conversion_factor)

        gaps.append(initial[country]["gap"])
        cost_gaps.append(0.0)

        gaps_pruned, cost_gaps_pruned = prune_results(
            np.array(gaps), np.array(cost_gaps)
        )

        # print("gaps pruned", gaps_pruned)
        country_gap_to_cost_interpolator = interp1d(
            gaps_pruned, cost_gaps_pruned, kind="linear"
        )

        country_interpolators[country] = {}
        country_interpolators[country][
            "gap_to_cost_interpolator"
        ] = country_gap_to_cost_interpolator

        country_interpolators[country]["gap_to_cost_interpolator_domain"] = (
            min(gaps_pruned),
            max(gaps_pruned),
        )
        rates = list(df["post_transfer_poverty_rate"] * 100)
        rates.append(initial[country]["rate"])
        rates1 = rates.copy()
        rates1.append(0.0)
        gaps1 = gaps.copy()
        gaps1.append(0.0)
        gaps_pruned, rates_pruned = prune_results(np.array(gaps1), np.array(rates1))
        country_gap_to_rate_interpolator = interp1d(
            gaps_pruned,
            rates_pruned,
            kind="linear",
        )
        country_interpolators[country][
            "gap_to_rate_interpolator"
        ] = country_gap_to_rate_interpolator

        country_interpolators[country]["gap_to_rate_interpolator_domain"] = (
            min(gaps_pruned),
            max(gaps_pruned),
        )

        rates_pruned, costs_pruned = prune_results(np.array(rates), np.array(cost_gaps))
        rate_to_cost_interpolator = interp1d(
            rates_pruned,
            costs_pruned,
            kind="linear",
        )
        country_interpolators[country][
            "rate_to_cost_interpolator"
        ] = rate_to_cost_interpolator
        country_interpolators[country]["rate_to_cost_interpolator_domain"] = (
            min(rates_pruned),
            max(rates_pruned),
        )

    return country_interpolators


def get_country_interpolators_fraction(countries, method, geo_extrapolation):
    country_interpolators = {}
    conversion_factors = get_conversion_factors(countries)

    initial = get_initial_poverty_gaps_and_rates(countries)

    for country in countries:
        df = _load_data(country, method, geo_extrapolation)
        country_conversion_factor = conversion_factors[country]

        gap_fractions = [0.0] + list(
            (initial[country]["gap"] - (df["post_transfer_poverty_gap"] * 100 / 2.15))
            * 100
            / initial[country]["gap"]
        )

        cost_gaps = [0.0] + list(
            df["policy_cost_per_capita"] * country_conversion_factor
        )

        pruned_gap_fractions, pruned_cost_gaps = prune_results(
            np.array(gap_fractions), np.array(cost_gaps), val=100.0
        )

        country_gap_interpolator = interp1d(
            pruned_gap_fractions,
            pruned_cost_gaps,
            kind="linear",
        )

        country_interpolators[country] = {}
        country_interpolators[country]["gap_interpolator"] = country_gap_interpolator

        rate_fractions = [0.0] + list(
            (
                (initial[country]["rate"] - (df["post_transfer_poverty_rate"] * 100))
                / initial[country]["rate"]
            )
            * 100
        )
        # rates.append(initial[country]["rate"] * country_conversion_factor)
        cost_rates = [0.0] + list(
            df["policy_cost_per_capita"] * country_conversion_factor
        )

        pruned_rate_fractions, pruned_cost_rates = prune_results(
            np.array(rate_fractions), np.array(cost_rates), val=100.0
        )
        # cost_rates.append(0.0)
        country_rate_interpolator = interp1d(
            pruned_rate_fractions,
            pruned_cost_rates,
            kind="linear",
        )
        country_interpolators[country]["rate_interpolator"] = country_rate_interpolator
    return country_interpolators


def get_aggregate_interpolators_fraction(countries, method, geo_extrapolation):
    country_interpolators = get_country_interpolators_fraction(
        countries, method, geo_extrapolation
    )

    # Compute aggregate rate interpolator
    fracs = np.linspace(0, 100, 200)
    costs = []
    for country in countries:
        country_gap_interpolator = country_interpolators[country]["gap_interpolator"]
        costs.append(np.clip(country_gap_interpolator(fracs), a_min=0.0, a_max=None))
    costs = np.array(costs)
    aggregate_gap_costs = np.sum(costs, axis=0)
    aggregate_gap_interpolator = interp1d(fracs, aggregate_gap_costs, kind="linear")

    # Compute aggregate rate interpolator
    rate_costs = []
    for country in countries:
        country_rate_interpolator = country_interpolators[country]["rate_interpolator"]
        rate_costs.append(
            np.clip(country_rate_interpolator(fracs), a_min=0.0, a_max=None)
        )
    rate_costs = np.array(rate_costs)
    aggregate_rate_costs = np.sum(rate_costs, axis=0)
    aggregate_rate_interpolator = interp1d(fracs, aggregate_rate_costs, kind="linear")

    return {
        "gap": {
            "interpolator": aggregate_gap_interpolator,
            "range": (fracs[0], fracs[-1]),
        },
        "rate": {
            "interpolator": aggregate_rate_interpolator,
            "range": (fracs[0], fracs[-1]),
        },
    }


def get_country_weights(countries):
    conversion_factors = pd.read_csv("learn/eop_conversion_factor.csv")
    conversion_factors = conversion_factors[
        conversion_factors["country"].isin(countries)
    ]
    conversion_factors["weight"] = (
        conversion_factors["total_population_survey_year"]
        / conversion_factors["total_population_survey_year"].sum()
    )
    return (
        conversion_factors[["country", "weight", "total_population_survey_year"]]
        .set_index("country")
        .to_dict()
    )


def get_wc_gap_to_actual_gap_country(initial, country, wc_gaps):
    actual_gaps = np.clip(wc_gaps, a_min=None, a_max=initial[country]["gap"])
    return actual_gaps


def get_wc_rate_to_actual_rate_country(initial, country, wc_rates):
    actual_rates = np.clip(wc_rates, a_min=None, a_max=initial[country]["rate"])
    return actual_rates


def get_wc_gap_to_actual_rate_country(initial, country_interpolators, country, wc_gaps):
    actual_gaps = get_wc_gap_to_actual_gap_country(initial, country, wc_gaps)
    actual_rates = country_interpolators[country]["gap_to_rate_interpolator"](
        actual_gaps
    )
    return actual_rates


def get_aggregate_conversion_factor(countries):
    conversion_factors = get_conversion_factors(countries)
    aggregate_conversion_factor = (
        np.array([conversion_factors[country] for country in countries])
    ).sum()
    return aggregate_conversion_factor


def get_aggregate_ubi_cost(countries):
    aggregate_conversion_factor = get_aggregate_conversion_factor(countries)
    return 2.15 * aggregate_conversion_factor


def get_initial_aggregate_gap_and_rate(countries):
    initial = get_initial_poverty_gaps_and_rates(countries)
    country_weights = get_country_weights(countries)
    weights = np.array([country_weights["weight"][country] for country in countries])
    pop_weighted_initial_poverty_gap = (
        np.array([initial[country]["gap"] for country in countries]) * weights
    ).sum()
    pop_weighted_initial_poverty_rate = (
        np.array([initial[country]["rate"] for country in countries]) * weights
    ).sum()
    return pop_weighted_initial_poverty_gap, pop_weighted_initial_poverty_rate


def get_aggregate_interpolators_population_weighted_poverty_measure_global_gap(
    countries, method, geo_extrapolation
):
    initial = get_initial_poverty_gaps_and_rates(countries)
    country_weights = get_country_weights(countries)
    weights = np.array([country_weights["weight"][country] for country in countries])
    pop_weighted_initial_poverty_gap, pop_weighted_initial_poverty_rate = (
        get_initial_aggregate_gap_and_rate(countries)
    )

    min = get_min_poverty_gaps_and_rates(countries)
    min_poverty_gap = max([min[country]["gap"] for country in countries])
    min_poverty_rate = max([min[country]["rate"] for country in countries])

    max_initial_poverty_gap = max([initial[country]["gap"] for country in countries])
    country_interpolators = get_country_interpolators(
        countries, method, geo_extrapolation
    )

    gaps = np.linspace(min_poverty_gap, max_initial_poverty_gap, 200)
    agg_gaps = 0.0
    agg_rates = 0.0
    for i, country in enumerate(countries):
        agg_gaps += (
            get_wc_gap_to_actual_gap_country(initial, country, wc_gaps=gaps)
            * weights[i]
        )
        agg_rates += (
            get_wc_gap_to_actual_rate_country(
                initial,
                country_interpolators=country_interpolators,
                country=country,
                wc_gaps=gaps,
            )
            * weights[i]
        )
    # print(max(agg_gaps), pop_weighted_initial_poverty_gap)
    np.testing.assert_almost_equal(
        max(agg_gaps), pop_weighted_initial_poverty_gap, decimal=2
    )
    # assert max(agg_rates) == pop_weighted_initial_poverty_rate

    # print(gaps)
    # print(agg_gaps)
    # print(agg_rates)
    aggregate_interpolator_wc_gap_to_actual_gap = interp1d(
        gaps, agg_gaps, kind="linear"
    )

    aggregate_interpolator_wc_gap_to_actual_rate = interp1d(
        gaps, agg_rates, kind="linear"
    )

    # print("wc gaps", gaps)
    # print("agg_gaps", agg_gaps)
    # print("agg_rates", agg_rates)

    # Compute wc gap to cost interpolator
    country_costs = []
    for country in countries:
        country_actual_gaps = np.clip(gaps, a_min=None, a_max=initial[country]["gap"])
        country_costs.append(
            np.clip(
                country_interpolators[country]["gap_to_cost_interpolator"](
                    country_actual_gaps
                ),
                a_min=0.0,
                a_max=None,
            )
        )
        # print(country_actual_gaps)
        # print(country_costs)
    costs = np.array(country_costs)
    aggregate_gap_costs = np.sum(costs, axis=0)

    aggregate_interpolator_actual_gap_to_cost = interp1d(
        aggregate_interpolator_wc_gap_to_actual_gap(gaps),
        aggregate_gap_costs,
        kind="linear",
    )
    aggregate_interpolator_actual_rate_to_cost = interp1d(
        aggregate_interpolator_wc_gap_to_actual_rate(gaps),
        aggregate_gap_costs,
        kind="linear",
    )

    return {
        "gap": {
            "interpolator": aggregate_interpolator_actual_gap_to_cost,
            "range": (min_poverty_gap, pop_weighted_initial_poverty_gap),
        },
        "rate": {
            "interpolator": aggregate_interpolator_actual_rate_to_cost,
            "range": (min_poverty_rate, pop_weighted_initial_poverty_rate),
        },
    }


def get_aggregate_interpolators_population_weighted_poverty_measure(
    countries, method, geo_extrapolation
):
    initial = get_initial_poverty_gaps_and_rates(countries)
    country_weights = get_country_weights(countries)
    weights = np.array([country_weights["weight"][country] for country in countries])
    pop_weighted_initial_poverty_gap, pop_weighted_initial_poverty_rate = (
        get_initial_aggregate_gap_and_rate(countries)
    )

    min_stuff = get_min_poverty_gaps_and_rates(countries, method)
    min_poverty_gap = max([min_stuff[country]["gap"] for country in countries])
    min_poverty_rate = max([min_stuff[country]["rate"] for country in countries])

    max_initial_poverty_gap = max([initial[country]["gap"] for country in countries])
    max_initial_poverty_rate = max([initial[country]["rate"] for country in countries])
    country_interpolators = get_country_interpolators(
        countries, method, geo_extrapolation
    )

    gaps = np.linspace(min_poverty_gap, max_initial_poverty_gap, 200)
    rates = np.linspace(min_poverty_rate, max_initial_poverty_rate, 200)
    agg_gaps = 0.0
    agg_rates = 0.0
    for i, country in enumerate(countries):
        agg_gaps += (
            get_wc_gap_to_actual_gap_country(initial, country, wc_gaps=gaps)
            * weights[i]
        )
        agg_rates += (
            get_wc_rate_to_actual_rate_country(initial, country=country, wc_rates=rates)
            * weights[i]
        )
    np.testing.assert_almost_equal(
        max(agg_rates), pop_weighted_initial_poverty_rate, decimal=2
    )
    # assert max(agg_rates) == pop_weighted_initial_poverty_rate

    aggregate_interpolator_wc_gap_to_actual_gap = interp1d(
        gaps, agg_gaps, kind="linear"
    )

    aggregate_interpolator_wc_rate_to_actual_rate = interp1d(
        rates, agg_rates, kind="linear"
    )

    # Compute wc gap to cost interpolator
    country_costs_gaps = []
    country_costs_rates = []
    for country in countries:
        country_actual_gaps = get_wc_gap_to_actual_gap_country(
            initial, country, wc_gaps=gaps
        )

        country_actual_rates = get_wc_rate_to_actual_rate_country(
            initial, country=country, wc_rates=rates
        )
        domain_gap = country_interpolators[country]["gap_to_cost_interpolator_domain"]
        assert min(country_actual_gaps) >= domain_gap[0]
        assert max(country_actual_gaps) <= domain_gap[1]

        domain_rate = country_interpolators[country]["rate_to_cost_interpolator_domain"]
        assert min(country_actual_rates) >= domain_rate[0]
        assert max(country_actual_rates) <= domain_rate[1]

        country_costs_gaps.append(
            np.clip(
                country_interpolators[country]["gap_to_cost_interpolator"](
                    country_actual_gaps
                ),
                a_min=0.0,
                a_max=None,
            )
        )
        country_costs_rates.append(
            np.clip(
                country_interpolators[country]["rate_to_cost_interpolator"](
                    country_actual_rates
                ),
                a_min=0.0,
                a_max=None,
            )
        )
    costs_gap = np.array(country_costs_gaps)
    costs_rates = np.array(country_costs_rates)
    aggregate_gap_costs = np.sum(costs_gap, axis=0)
    aggregate_rate_costs = np.sum(costs_rates, axis=0)

    x1s = aggregate_interpolator_wc_gap_to_actual_gap(gaps)
    aggregate_interpolator_actual_gap_to_cost = interp1d(
        x1s,
        aggregate_gap_costs,
        kind="linear",
    )
    x2s = aggregate_interpolator_wc_rate_to_actual_rate(rates)
    aggregate_interpolator_actual_rate_to_cost = interp1d(
        x2s,
        aggregate_rate_costs,
        kind="linear",
    )

    return {
        "gap": {
            "interpolator": aggregate_interpolator_actual_gap_to_cost,
            "range": (min(x1s), max(x1s)),
        },
        "rate": {
            "interpolator": aggregate_interpolator_actual_rate_to_cost,
            "range": (min(x2s), max(x2s)),
        },
    }
