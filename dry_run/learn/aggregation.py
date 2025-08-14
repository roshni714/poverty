import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from constants import C_BAR

METHODS = {
    "oracle_gap": {
        "csv": "oracle_gap",
        "name": "Oracle Gap Targeting",
        "color": "green",
        "linestyle": "-",
    },
    "continuous_gap": {
        "csv": "output_gt_continuous_gap",
        "name": "Gap Targeting (Continuous)",
        "color": "blue",
        "linestyle": "-",
    },
    "binary_gap": {
        "csv": "output_gt_binary_gap",
        "name": "Gap Targeting (Binary)",
        "color": "blue",
        "linestyle": "--",
    },
    "continuous_rate": {
        "csv": "output_gt_continuous_rate",
        "name": "Rate Targeting (Continuous)",
        "color": "orange",
        "linestyle": "-",
    },
    "binary_rate": {
        "csv": "output_gt_binary_rate",
        "name": "Rate Targeting (Binary)",
        "color": "orange",
        "linestyle": "--",
    },
    "pmt": {
        "csv": "output_gt_pmt",
        "name": "PMT (Lasso)",
        "color": "red",
        "linestyle": "-",
    },
    "modern_pmt": {
        "csv": "output_gt_modern_pmt",
        "name": "PMT (NN)",
        "color": "red",
        "linestyle": "--",
    },
    "ubi": {
        "csv": "ubi",
        "name": "UBI by Country",
        "color": "purple",
        "linestyle": "-",
    },
}

AUX_DATA_CSV = "learn/aux_data_07302025.csv"

# def prune_results(xs, ys, val=0.0):
#     mask = np.abs(xs - val) < 1e-3
#     if mask.sum() <= 1:
#         return xs, ys
#     xs_nonzero = list(np.array(xs)[~mask])
#     ys_nonzero = list(np.array(ys)[~mask])

#     ys_zero = ys[mask]
#     xs = xs_nonzero + [val]
#     ys = ys_nonzero + [min(ys_zero)]
#     return np.array(xs), np.array(ys)


class CountryMethodPovertyResults:
    def __init__(self, country, method, geo_extrapolation):
        self.country = country
        self.method = method
        self.geo_extrapolation = geo_extrapolation
        self.conversion_factor = self._get_conversion_factor()
        self._get_min_poverty_gap_and_rate()
        self._get_initial_poverty_gap_and_rate()
        self._get_result_interpolators()

    def _load_data(self, method):
        """
        Load the data for a specific country and method.

        :param country: The name of the country.
        :type country: str
        :param method: The method to load data for.
        :type method: str
        :return: A DataFrame containing the data for the specified country and method.
        :rtype: pandas.DataFrame
        """
        if self.geo_extrapolation:
            subfolder = "geo_extrapolation"
        else:
            subfolder = "geo_interpolation"
        df = pd.read_csv(
            "learn/results/{}/{}/{}.csv".format(
                self.country, subfolder, METHODS[method]["csv"]
            )
        )
        return df

    def _get_conversion_factor(self):
        df = pd.read_csv(AUX_DATA_CSV)
        country_df = df[df["country"] == self.country]
        factor = (
            country_df["total_population_survey_year"].values[0]
            * 365
            * 1.14  # Conversion factor from 2021 to 2023
            * (
                country_df["PPP_conversion_factor_2021"].values[0]
                / country_df["market_exchange_rate_2021"].values[0]
            )
            / 1000000000
        )
        return factor

    def _get_initial_poverty_gap_and_rate(self):
        df = self._load_data(self.method)
        self.initial_gap = (df["post_transfer_poverty_gap"].max() / C_BAR) * 100
        self.initial_rate = df["post_transfer_poverty_rate"].max() * 100

    def _get_min_poverty_gap_and_rate(self):
        df = self._load_data(self.method)
        self.min_gap = (df["post_transfer_poverty_gap"].min() / C_BAR) * 100
        self.min_rate = df["post_transfer_poverty_rate"].min() * 100

    def _get_result_interpolators(self):
        df = self._load_data(self.method)
        country_conversion_factor = self.conversion_factor
        gaps = list(df["post_transfer_poverty_gap"] * 100 / C_BAR)
        cost_gaps = list(df["policy_cost_per_capita"] * country_conversion_factor)
        gaps.append(self.initial_gap)
        cost_gaps.append(0.0)

        # gaps_pruned, cost_gaps_pruned = prune_results(
        #    np.array(gaps), np.array(cost_gaps)
        # )

        # print("gaps pruned", gaps_pruned)
        country_gap_to_cost_interpolator = interp1d(gaps, cost_gaps, kind="linear")

        self.gap_to_cost_interpolator = country_gap_to_cost_interpolator
        self.gap_to_cost_interpolator_domain = (
            min(gaps),
            max(gaps),
        )
        rates = list(df["post_transfer_poverty_rate"] * 100)
        rates.append(self.initial_rate)
        rates1 = rates.copy()
        rates1.append(0.0)
        gaps1 = gaps.copy()
        gaps1.append(0.0)
        # (np.array(gaps1), np.array(rates1))
        country_gap_to_rate_interpolator = interp1d(
            rates,
            gaps,
            kind="linear",
        )
        self.rate_to_gap_interpolator = country_gap_to_rate_interpolator

        self.rate_to_gap_interpolator_domain = (
            min(gaps),
            max(gaps),
        )

        # rates_pruned, costs_pruned = prune_results(np.array(rates), np.array(cost_gaps))
        rate_to_cost_interpolator = interp1d(
            rates,
            cost_gaps,
            kind="linear",
        )
        self.rate_to_cost_interpolator = rate_to_cost_interpolator
        self.rate_to_cost_interpolator_domain = (
            min(rates),
            max(rates),
        )


class AggregatePovertyResults:

    def __init__(self, countries, method, geo_extrapolation):
        self.countries = countries
        self.method = method
        self.geo_extrapolation = geo_extrapolation
        self.country_results = {
            country: CountryMethodPovertyResults(country, method, geo_extrapolation)
            for country in countries
        }
        self._get_country_weights_and_pop()
        self._get_aggregate_interpolators()

    def _get_country_weights_and_pop(self):
        conversion_factors = pd.read_csv(AUX_DATA_CSV)
        conversion_factors = conversion_factors[
            conversion_factors["country"].isin(self.countries)
        ]
        conversion_factors["weight"] = (
            conversion_factors["total_population_survey_year"]
            / conversion_factors["total_population_survey_year"].sum()
        )
        self.country_weights = (
            conversion_factors[["country", "weight", "total_population_survey_year"]]
            .set_index("country")
            .to_dict()
        )

    def get_aggregate_ubi_cost(self):
        aggregate_conversion_factor = sum(
            [
                self.country_results[country].conversion_factor
                for country in self.countries
            ]
        )
        return C_BAR * aggregate_conversion_factor

    def get_initial_aggregate_gap_and_rate(self):
        initial_gaps = [
            self.country_results[country].initial_gap for country in self.countries
        ]
        initial_rates = [
            self.country_results[country].initial_rate for country in self.countries
        ]
        weights = np.array(
            [self.country_weights["weight"][country] for country in self.countries]
        )
        pop_weighted_initial_poverty_gap = (np.array(initial_gaps) * weights).sum()
        pop_weighted_initial_poverty_rate = (np.array(initial_rates) * weights).sum()
        return pop_weighted_initial_poverty_gap, pop_weighted_initial_poverty_rate

    def get_min_aggregate_gap_and_rate(self):
        min_gaps = [self.country_results[country].min_gap for country in self.countries]
        min_rates = [
            self.country_results[country].min_rate for country in self.countries
        ]
        weights = np.array(
            [self.country_weights["weight"][country] for country in self.countries]
        )
        pop_weighted_min_poverty_gap = (np.array(min_gaps) * weights).sum()
        pop_weighted_min_poverty_rate = (np.array(min_rates) * weights).sum()
        return pop_weighted_min_poverty_gap, pop_weighted_min_poverty_rate

    def get_max_min_gap_and_rate(self):
        max_min_poverty_gap = max(
            [self.country_results[country].min_gap for country in self.countries]
        )
        max_min_poverty_rate = max(
            [self.country_results[country].min_rate for country in self.countries]
        )
        return max_min_poverty_gap, max_min_poverty_rate

    def _get_aggregate_interpolators(self):
        _, max_min_rate = self.get_max_min_gap_and_rate()
        max_max_rate = max(
            [self.country_results[country].initial_rate for country in self.countries]
        )
        weights = np.array(
            [self.country_weights["weight"][country] for country in self.countries]
        )

        wc_rates = np.linspace(max_min_rate, max_max_rate, 50)
        agg_gaps = 0.0
        agg_rates = 0.0
        for i, country in enumerate(self.countries):
            country_result = self.country_results[country]
            country_actual_rates = np.clip(
                wc_rates, a_min=None, a_max=country_result.initial_rate
            )
            weighted_country_actual_rates = country_actual_rates * weights[i]
            weighted_country_actual_gaps = (
                country_result.rate_to_gap_interpolator(country_actual_rates)
                * weights[i]
            )
            agg_rates += weighted_country_actual_rates
            agg_gaps += weighted_country_actual_gaps

        aggregate_interpolator_wc_rate_to_actual_gap = interp1d(
            wc_rates, agg_gaps, kind="linear"
        )
        aggregate_interpolator_wc_rate_to_actual_rate = interp1d(
            wc_rates, agg_rates, kind="linear"
        )

        agg_costs = 0.0
        for i, country in enumerate(self.countries):
            country_result = self.country_results[country]
            country_actual_rates = np.clip(
                wc_rates, a_min=None, a_max=country_result.initial_rate
            )
            country_actual_costs = country_result.rate_to_cost_interpolator(
                country_actual_rates
            )
            agg_costs += country_actual_costs

        aggregate_interpolator_wc_rate_to_costs = interp1d(
            wc_rates, agg_costs, kind="linear"
        )

        actual_rates = aggregate_interpolator_wc_rate_to_actual_rate(wc_rates)
        actual_gaps = aggregate_interpolator_wc_rate_to_actual_gap(wc_rates)
        actual_costs = aggregate_interpolator_wc_rate_to_costs(wc_rates)

        self.aggregate_interpolator_rate_to_cost = interp1d(
            actual_rates, actual_costs, kind="linear"
        )

        self.aggregate_interpolator_gap_to_cost = interp1d(
            actual_gaps, actual_costs, kind="linear"
        )
        self.aggregate_interpolator_rate_domain = (min(actual_rates), max(actual_rates))
        self.aggregate_interpolator_gap_domain = (min(actual_gaps), max(actual_gaps))
