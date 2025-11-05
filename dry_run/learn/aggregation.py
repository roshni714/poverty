import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from learn.formatting import METHODS
from learn.aux_data_prep import preprocess_wpc_data, preprocess_country_aux_data
import unicodedata


class CountryMethodPovertyResults:
    def __init__(self, country, method, geo_extrapolation, povertyline, year):
        self.country = country
        self.method = method
        self.geo_extrapolation = geo_extrapolation
        self.year = year
        self.povertyline = povertyline
        self.conversion_factor = self._get_conversion_factor()
        self._get_min_poverty_gap_index_and_rate()
        self._get_initial_poverty_gap_index_and_rate()
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
            "learn/results/{}/{}/year={}/{}.csv".format(
                self.country, subfolder, self.year, METHODS[method]["csv"]
            )
        )
        return df

    def _get_conversion_factor(self):
        df = preprocess_country_aux_data()

        # second_df = pd.read_csv(SECONDARY_AUX_DATA_CSV)
        # nominal_conversion_factor = second_df["conversion_factor_nominal_USD_{}_to_2023".format(self.year)].values[0]
        # print("WARNING: FIX HARDCODED NOMINAL CONVERSION FACTOR")
        if self.year == 2021:
            nominal_conversion_factor = 1.14
        elif self.year == 2017:
            nominal_conversion_factor = 1.23

        country_df = df[df["country_code"] == self.country]
        if country_df.shape[0] == 0:
            raise ValueError(
                "Country {} not found in auxiliary data.".format(self.country)
            )

        factor = (
            country_df["total_population_survey_year"].values[0]
            * 365
            * nominal_conversion_factor
            * (
                country_df["PPP_conversion_factor_{}".format(self.year)].values[0]
                / country_df["market_exchange_rate_{}".format(self.year)].values[0]
            )
        ) / 1000000000
        return factor

    def _get_initial_poverty_gap_index_and_rate(self):
        df = self._load_data("oracle_gap")
        self.initial_gap_index = (
            df["initial_poverty_gap"].max() / self.povertyline
        ) * 100
        self.initial_rate = df["initial_poverty_rate"].max() * 100

    def get_poverty_gap(self):
        return (
            self.conversion_factor * (self.initial_gap_index / 100) * self.povertyline
        )

    def get_ubi_cost(self):
        return self.conversion_factor * self.povertyline

    def _get_min_poverty_gap_index_and_rate(self):
        df = self._load_data(self.method)
        self.min_gap_index = (
            df["post_transfer_poverty_gap"].min() / self.povertyline
        ) * 100
        self.min_rate = df["post_transfer_poverty_rate"].min() * 100

    def _get_result_interpolators(self):
        df = self._load_data(self.method)
        country_conversion_factor = self.conversion_factor
        gaps = list(df["post_transfer_poverty_gap"] * 100 / self.povertyline)
        cost_gaps = list(df["policy_cost_per_capita"] * country_conversion_factor)
        gaps.append(self.initial_gap_index)
        cost_gaps.append(0.0)

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
            rates1,
            gaps1,
            kind="linear",
        )
        self.rate_to_gap_interpolator = country_gap_to_rate_interpolator

        self.rate_to_gap_interpolator_domain = (
            min(rates1),
            max(rates1),
        )

        # rates_pruned, costs_pruned = prune_results(np.array(rates), np.array(cost_gaps))

        # gaps.append(0.)
        # cost_gaps.append(self.povertyline * country_conversion_factor)
        # rates.append(0.)
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

    def __init__(self, countries, method, geo_extrapolation, year, povertyline):
        self.countries = countries
        self.method = method
        self.geo_extrapolation = geo_extrapolation
        self.year = year
        self.povertyline = povertyline
        self.country_results = {
            country: CountryMethodPovertyResults(
                country, method, geo_extrapolation, povertyline, year
            )
            for country in countries
        }
        self._get_country_weights_and_pop()
        self._get_aggregate_interpolators()

    def _get_country_weights_and_pop(self):
        df = preprocess_country_aux_data()
        df = df[df["country_code"].isin(self.countries)]
        df["weight"] = (
            df["total_population_survey_year"]
            / df["total_population_survey_year"].sum()
        )
        self.country_weights = (
            df[["country_code", "weight", "total_population_survey_year"]]
            .set_index("country_code")
            .to_dict()
        )

    def get_aggregate_ubi_cost(self):
        aggregate_conversion_factor = sum(
            [
                self.country_results[country].conversion_factor
                for country in self.countries
            ]
        )
        return self.povertyline * aggregate_conversion_factor

    def get_aggregate_poverty_gap(self):
        gaps = np.array(
            [
                self.country_results[country].get_poverty_gap()
                for country in self.countries
            ]
        )

        return np.sum(gaps)

    def get_initial_aggregate_gap_index_and_rate(self):
        initial_gaps = [
            self.country_results[country].initial_gap_index
            for country in self.countries
        ]
        initial_rates = [
            self.country_results[country].initial_rate for country in self.countries
        ]
        weights = np.array(
            [self.country_weights["weight"][country] for country in self.countries]
        )
        pop_weighted_initial_poverty_gap_index = (
            np.array(initial_gaps) * weights
        ).sum()
        pop_weighted_initial_poverty_rate = (np.array(initial_rates) * weights).sum()
        return pop_weighted_initial_poverty_gap_index, pop_weighted_initial_poverty_rate

    def get_min_aggregate_gap_index_and_rate(self):
        min_gaps = [
            self.country_results[country].min_gap_index for country in self.countries
        ]
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
            [self.country_results[country].min_gap_index for country in self.countries]
        )

        rates = [self.country_results[country].min_rate for country in self.countries]
        # if self.method == "continuous_gap":
        #    import pdb; pdb.set_trace()

        max_min_poverty_rate = max(rates)

        return max_min_poverty_gap, max_min_poverty_rate

    def _get_aggregate_interpolators(self):
        _, max_min_rate = self.get_max_min_gap_and_rate()
        # if self.method == "continuous_gap":
        #    import pdb; pdb.set_trace()

        max_max_rate = max(
            [self.country_results[country].initial_rate for country in self.countries]
        )
        weights = np.array(
            [self.country_weights["weight"][country] for country in self.countries]
        )

        wc_rates = np.linspace(max_min_rate, max_max_rate, 400)
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
        self.aggregate_interpolator_rate_to_gap = interp1d(
            actual_rates, actual_gaps, kind="linear"
        )
        self.aggregate_interpolator_rate_domain = (min(actual_rates), max(actual_rates))
        self.aggregate_interpolator_gap_domain = (min(actual_gaps), max(actual_gaps))
