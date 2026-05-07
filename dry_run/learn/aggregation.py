import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from learn.formatting import METHODS
import os


class AveragedCountryMethodPovertyResults:
    """
    Class to report poverty results averaged across multiple runs for a specific country and method.

    :param country: The name of the country.
    :type country: str
    :param method: The method to report results for.
    :type method: str
    :param geo_extrapolation: Whether to use geo-extrapolation or geo-interpolation.
    :type geo_extrapolation: bool
    :param povertyline: The international poverty line in PPP (e.g. 2.15, 3.0)
    :type povertyline: float
    :param year: The year of the international poverty line (2017 or 2021).
    """

    def __init__(self, country, method, metadata, train_frac=None):
        self.country = country
        self.method = method
        self.geo_extrapolation = metadata.geo_extrapolation
        self.restricted_feature_set = metadata.restricted_feature_set
        self.year = metadata.year
        self.povertyline = metadata.povertyline
        self.train_frac = train_frac
        self.metadata = metadata
        self.n_seeds = 10

        self.country_seed_results = [
            CountryMethodPovertyResults(
                country,
                method,
                metadata,
                train_frac=train_frac,
                seed=seed,
            )
            for seed in range(self.n_seeds)
        ]
        self._get_initial()
        self._get_min_poverty_gap_index_and_rate()
        self._get_averaged_result_interpolators()
        self._get_std_dev_result_interpolators()

    def get_ubi_cost(self):
        """
        Get the cost of a universal basic income at the poverty line in billion USD (nominal 2023).
        :return: UBI cost.
        :rtype: float
        """
        return self.country_seed_results[0].get_ubi_cost()

    def _get_initial(self):
        """
        Set initial poverty gap index and rate by averaging across seeds.
        """

        self.initial_gap_index = self.country_seed_results[0].initial_gap_index
        self.initial_rate = self.country_seed_results[0].initial_rate
        self.initial_welfare = self.country_seed_results[0].initial_welfare

    def _get_min_poverty_gap_index_and_rate(self):
        """
        Get minimum poverty gap index and rate achieved by method.
        Note this method is used so that we do not extrapolate beyond bounds of the results.
        """
        self.min_gap_index = max(
            self.country_seed_results[seed].min_gap_index
            for seed in range(self.n_seeds)
        )
        self.min_rate = max(
            self.country_seed_results[seed].min_rate for seed in range(self.n_seeds)
        )

    def _get_averaged_result_interpolators(self):
        """
        Set averaged result interpolators for poverty gap index to cost, poverty rate to cost, and poverty rate to poverty gap index by averaging across seeds.
        """
        gap_to_cost_interpolators = [
            result.gap_to_cost_interpolator for result in self.country_seed_results
        ]
        rate_to_cost_interpolators = [
            result.rate_to_cost_interpolator for result in self.country_seed_results
        ]
        rate_to_gap_interpolators = [
            result.rate_to_gap_interpolator for result in self.country_seed_results
        ]
        welfare_to_cost_interpolators = [
            result.welfare_to_cost_interpolator for result in self.country_seed_results
        ]
        rate_to_welfare_interpolators = [
            result.rate_to_welfare_interpolator for result in self.country_seed_results
        ]

        def averaged_gap_to_cost_interpolator(gap):
            return np.mean(
                [interpolator(gap) for interpolator in gap_to_cost_interpolators],
                axis=0,
            )

        def averaged_rate_to_cost_interpolator(rate):
            return np.mean(
                [interpolator(rate) for interpolator in rate_to_cost_interpolators],
                axis=0,
            )

        def averaged_rate_to_gap_interpolator(rate):
            return np.mean(
                [interpolator(rate) for interpolator in rate_to_gap_interpolators],
                axis=0,
            )

        def averaged_welfare_to_cost_interpolator(welfare):
            return np.mean(
                [
                    interpolator(welfare)
                    for interpolator in welfare_to_cost_interpolators
                ],
                axis=0,
            )

        def averaged_rate_to_welfare_interpolator(rate):
            return np.mean(
                [interpolator(rate) for interpolator in rate_to_welfare_interpolators],
                axis=0,
            )

        self.gap_to_cost_interpolator = averaged_gap_to_cost_interpolator
        self.rate_to_cost_interpolator = averaged_rate_to_cost_interpolator
        self.rate_to_gap_interpolator = averaged_rate_to_gap_interpolator
        self.welfare_to_cost_interpolator = averaged_welfare_to_cost_interpolator
        self.rate_to_welfare_interpolator = averaged_rate_to_welfare_interpolator

    def _get_std_dev_result_interpolators(self):
        """
        Set standard deviation result interpolators for poverty gap index to cost, poverty rate to cost, and poverty rate to poverty gap index by computing standard deviation across seeds.
        """
        gap_to_cost_interpolators = [
            result.gap_to_cost_interpolator for result in self.country_seed_results
        ]
        rate_to_cost_interpolators = [
            result.rate_to_cost_interpolator for result in self.country_seed_results
        ]
        rate_to_gap_interpolators = [
            result.rate_to_gap_interpolator for result in self.country_seed_results
        ]
        welfare_to_cost_interpolators = [
            result.welfare_to_cost_interpolator for result in self.country_seed_results
        ]
        rate_to_welfare_interpolators = [
            result.rate_to_welfare_interpolator for result in self.country_seed_results
        ]

        def std_error_gap_to_cost_interpolator(gap):
            return np.std(
                [interpolator(gap) for interpolator in gap_to_cost_interpolators],
                axis=0,
            )

        def std_error_rate_to_cost_interpolator(rate):
            return np.std(
                [interpolator(rate) for interpolator in rate_to_cost_interpolators],
                axis=0,
            )

        def std_error_rate_to_gap_interpolator(rate):
            return np.std(
                [interpolator(rate) for interpolator in rate_to_gap_interpolators],
                axis=0,
            )

        def std_error_welfare_to_cost_interpolator(welfare):
            return np.std(
                [
                    interpolator(welfare)
                    for interpolator in welfare_to_cost_interpolators
                ],
                axis=0,
            )

        def std_error_rate_to_welfare_interpolator(rate):
            return np.std(
                [interpolator(rate) for interpolator in rate_to_welfare_interpolators],
                axis=0,
            )

        self.std_dev_gap_to_cost_interpolator = std_error_gap_to_cost_interpolator
        self.std_dev_rate_to_cost_interpolator = std_error_rate_to_cost_interpolator
        self.std_dev_rate_to_gap_interpolator = std_error_rate_to_gap_interpolator
        self.std_dev_welfare_to_cost_interpolator = (
            std_error_welfare_to_cost_interpolator
        )
        self.std_dev_rate_to_welfare_interpolator = (
            std_error_rate_to_welfare_interpolator
        )


class CountryMethodPovertyResults:
    """
    Class to report poverty results for a specific country and method.

    :param country: The name of the country.
    :type country: str
    :param method: The method to report results for.
    :type method: str
    :param geo_extrapolation: Whether to use geo-extrapolation or geo-interpolation.
    :type geo_extrapolation: bool
    :param povertyline: The international poverty line in PPP (e.g. 2.15, 3.0)
    :type povertyline: float
    :param year: The year of the international poverty line (2017 or 2021).
    """

    def __init__(self, country, method, metadata, train_frac=None, seed=None):
        self.country = country
        self.method = method
        self.geo_extrapolation = metadata.geo_extrapolation
        self.restricted_feature_set = metadata.restricted_feature_set
        self.year = metadata.year
        self.povertyline = metadata.povertyline
        self.train_frac = train_frac
        self.metadata = metadata
        self.seed = seed
        self.conversion_factor = self._get_conversion_factor()
        self._get_min_poverty_gap_index_and_rate()
        self._get_initial()
        self._get_result_interpolators()

    def _load_data(self):
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

        if self.train_frac is None:

            if self.method == "oracle_gap":
                df = pd.read_csv(
                    "learn/results/{}/{}/year={}/{}.csv".format(
                        self.country, subfolder, self.year, METHODS[self.method]["csv"]
                    )
                )
            elif self.method != "oracle_gap" and self.restricted_feature_set:
                df = pd.read_csv(
                    "learn/results/{}/{}/year={}_d=20/{}.csv".format(
                        self.country, subfolder, self.year, METHODS[self.method]["csv"]
                    )
                )
            else:
                df = pd.read_csv(
                    "learn/results/{}/{}/year={}/{}.csv".format(
                        self.country, subfolder, self.year, METHODS[self.method]["csv"]
                    )
                )
            return df

        else:
            df = pd.read_csv(
                "learn/results/{}/{}/year={}_sample_size/{}_nprop={}_seed={}.csv".format(
                    self.country,
                    subfolder,
                    self.year,
                    METHODS[self.method]["csv"],
                    self.train_frac,
                    self.seed,
                )
            )
            return df

    def _get_conversion_factor(self):
        """
        Conversion factor from per capita per day in PPP in year corresponding to self.year to total annual cost in billion USD (nominal 2023).

        :return: Conversion factor.
        :rtype: float
        """
        df = self.metadata.preprocess_country_aux_data()

        # nominal conversion factor from year (of international poverty line) to 2023
        secondary_df = self.metadata.preprocess_secondary_aux_data()
        inflation_adjustment = (
            1
            / secondary_df[
                secondary_df["indicator"]
                == "conversion_factor_nominal_USD_2023_to_{}".format(self.year)
            ]["value"]
            .values[0]
            .item()
        )

        country_code = self.country[:3]

        country_df = df[df["country_code"] == country_code]
        if country_df.shape[0] == 0:
            raise ValueError(
                "Country {} not found in auxiliary data.".format(country_code)
            )

        factor = (
            country_df["total_population_survey_year"].values[0]  # population
            * 365  # days per year
            * inflation_adjustment  # from year to 2023 nominal
            * (
                country_df["PPP_conversion_factor_{}".format(self.year)].values[0]
                / country_df["market_exchange_rate_{}".format(self.year)].values[0]
            )  # from PPP to nominal USD in year
        ) / 1000000000  # to billion USD
        return factor

    def _get_initial(self):
        """
        Set initial poverty gap index and rate.
        """
        df = self._load_data()
        self.initial_gap_index = (
            df["initial_poverty_gap"].values[0] / self.povertyline
        ) * 100
        self.initial_rate = df["initial_poverty_rate"].values[0] * 100

        if "initial_welfare" not in df.columns:
            print(
                "WARNING: initial welfare not found in data for country {}, method {}. Setting initial welfare to 0.".format(
                    self.country, self.method
                )
            )
            self.initial_welfare = np.log(self.conversion_factor)
        else:
            self.initial_welfare = df["initial_welfare"].values[0] + np.log(
                self.conversion_factor
            )

    def get_poverty_gap(self):
        """
        Get the total poverty gap in billion USD (nominal 2023).
        :return: Total poverty gap.
        :rtype: float
        """
        return (
            self.conversion_factor * (self.initial_gap_index / 100) * self.povertyline
        )

    def get_ubi_cost(self):
        """
        Get the cost of a universal basic income at the poverty line in billion USD (nominal 2023).
        :return: UBI cost.
        :rtype: float
        """
        return self.conversion_factor * self.povertyline

    def _get_min_poverty_gap_index_and_rate(self):
        """
        Get minimum poverty gap index and rate achieved by method.
        Note this method is used so that we do not extrapolate beyond bounds of the results.
        """
        df = self._load_data()
        self.min_gap_index = (
            df["post_transfer_poverty_gap"].min() / self.povertyline
        ) * 100
        self.min_rate = df["post_transfer_poverty_rate"].min() * 100

    def _get_result_interpolators(self):
        """
        Set results interpolators for poverty gap index to cost, poverty rate to cost, and poverty rate to poverty gap index.
        """
        df = self._load_data()
        country_conversion_factor = self.conversion_factor

        # Build interpolator from poverty gap index to cost
        gaps = list(df["post_transfer_poverty_gap"] * 100 / self.povertyline)
        costs = list(df["policy_cost_per_capita"] * country_conversion_factor)
        gaps.append(self.initial_gap_index)
        costs.append(0.0)
        country_gap_to_cost_interpolator = interp1d(gaps, costs, kind="linear")
        self.gap_to_cost_interpolator = country_gap_to_cost_interpolator
        self.gap_to_cost_interpolator_domain = (
            min(gaps),
            max(gaps),
        )

        # Build interpolator from poverty gap index to poverty rate
        rates = list(df["post_transfer_poverty_rate"] * 100)
        rates.append(self.initial_rate)
        rates1 = rates.copy()
        rates1.append(0.0)
        gaps1 = gaps.copy()
        gaps1.append(0.0)
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

        # Build interpolator from poverty rate to cost
        rate_to_cost_interpolator = interp1d(
            rates,
            costs,
            kind="linear",
        )
        self.rate_to_cost_interpolator = rate_to_cost_interpolator
        self.rate_to_cost_interpolator_domain = (
            min(rates),
            max(rates),
        )

        # Build interpolator from cost to welfare
        if "post_transfer_welfare" not in df.columns:
            df["post_transfer_welfare"] = 0.0

        welfares = list(
            df["post_transfer_welfare"] + np.log(country_conversion_factor)
        )  # Replace "welfare_metric" with the actual column name
        welfares.append(self.initial_welfare)
        welfare_to_cost_interpolator = interp1d(welfares, costs, kind="linear")
        self.welfare_to_cost_interpolator_domain = (min(welfares), max(welfares))
        self.welfare_to_cost_interpolator = welfare_to_cost_interpolator

        rate_to_welfare_interpolator = interp1d(rates, welfares, kind="linear")
        self.rate_to_welfare_interpolator = rate_to_welfare_interpolator
        self.rate_to_welfare_interpolator_domain = (
            min(rates),
            max(rates),
        )

    def _load_transfer_data(self):
        """
        Load the transfer data for a specific country and method.
        :return: A DataFrame containing the transfer data for the specified country and method.
        :rtype: pandas.DataFrame
        """
        if self.geo_extrapolation:
            subfolder = "geo_extrapolation"
        else:
            subfolder = "geo_interpolation"

        if self.method == "oracle_gap":
            directory_path = "learn/results/{}/{}/year={}".format(
                self.country, subfolder, self.year
            )
            prefix = "oracle_gap_budget="
        elif self.method != "oracle_gap" and self.restricted_feature_set:
            directory_path = "learn/results/{}/{}/year={}_d=20".format(
                self.country, subfolder, self.year
            )
            prefix = "output_gt_{}_budget=".format(self.method)
        else:
            directory_path = "learn/results/{}/{}/year={}".format(
                self.country, subfolder, self.year
            )
            prefix = "output_gt_{}_budget=".format(self.method)

        files_with_prefix = []
        for filename in os.listdir(directory_path):
            if filename.startswith(prefix) and os.path.isfile(
                os.path.join(directory_path, filename)
            ):
                files_with_prefix.append(filename)

        transfer_dfs = []
        budgets = []

        for filename in files_with_prefix:
            if filename.startswith(prefix):
                transfer_df = pd.read_csv(os.path.join(directory_path, filename))
                budget = filename.split("_budget=")[1].split(".csv")[0]
                budgets.append(float(budget))
                test_data = pd.read_parquet(
                    "data/{}/test.parquet".format(self.country)
                ).reset_index(drop=True)
                transfer_df = transfer_df.merge(
                    test_data["headcount_adjusted_hh_wgt"],
                    left_index=True,
                    right_index=True,
                )
                transfer_dfs.append(transfer_df)

        return budgets, transfer_dfs

    def get_number_of_people_targeted(self):
        """
        Get the number of people targeted by the method.
        :return: Number of people targeted.
        :rtype: float
        """
        transfer_dfs = self._load_transfer_data()

        pov_rates = []
        num_targeted = []
        df = self.metadata.preprocess_country_aux_data()
        total_pop = df[df["country_code"] == self.country][
            "total_population_survey_year"
        ].values[0]
        for transfer_df in transfer_dfs:
            transfer_df["headcount_adjusted_hh_wgt"] /= transfer_df[
                "headcount_adjusted_hh_wgt"
            ].sum()
            sum_weights_positive_transfer = transfer_df[transfer_df["ev_transfer"] > 0][
                "headcount_adjusted_hh_wgt"
            ].sum()
            post_transfer_pov_rate = np.sum(
                (
                    transfer_df["consumption"] + transfer_df["ev_transfer"]
                    < self.povertyline
                )
                * transfer_df["headcount_adjusted_hh_wgt"]
            )
            pov_rates.append(post_transfer_pov_rate)
            num_targeted.append(sum_weights_positive_transfer * total_pop)

        interpolator = interp1d(np.array(pov_rates) * 100, num_targeted, kind="linear")
        return interpolator


class AggregatePovertyResults:
    """
    Class to report aggregate poverty results across multiple countries for a specific method.
    :param countries: List of country codes.
    :type countries: list
    :param method: The method to report results for.
    :type method: str
    :param geo_extrapolation: Whether to use geo-extrapolation or geo-interpolation.
    :type geo_extrapolation: bool
    :param year: The year of the international poverty line (2017 or 2021).
    :type year: int
    :param povertyline: The international poverty line in PPP (e.g. 2.15, 3.0)
    :type povertyline: float
    """

    def __init__(self, countries, method, metadata, train_frac=None):
        self.countries = countries
        self.method = method
        self.geo_extrapolation = metadata.geo_extrapolation
        self.year = metadata.year
        self.povertyline = metadata.povertyline
        self.metadata = metadata
        self.train_frac = train_frac
        if self.train_frac is not None:
            self.country_results = {
                country: AveragedCountryMethodPovertyResults(
                    country,
                    method,
                    metadata,
                    train_frac=train_frac,
                )
                for country in countries
            }
        else:
            self.country_results = {
                country: CountryMethodPovertyResults(
                    country,
                    method,
                    metadata,
                    train_frac=train_frac,
                )
                for country in countries
            }
        self._get_country_weights_and_pop()
        self._get_aggregate_interpolators()

    def _get_country_weights_and_pop(self):
        """
        Set population weights for each country based on survey year population.
        """
        df = self.metadata.preprocess_country_aux_data()
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
        """
        Get the aggregate cost of a universal basic income at the poverty line in billion USD (nominal 2023).
        :return: Aggregate UBI cost.
        :rtype: float
        """

        return sum(
            [self.country_results[country].get_ubi_cost() for country in self.countries]
        )

    def get_aggregate_poverty_gap(self):
        """
        Get the aggregate poverty gap in billion USD (nominal 2023).
        :return: Aggregate poverty gap.
        :rtype: float
        """
        gaps = np.array(
            [
                self.country_results[country].get_poverty_gap()
                for country in self.countries
            ]
        )

        return np.sum(gaps)

    def get_initial(self):
        """
        Get initial population-weighted poverty gap index and rate.
        :return: Tuple of (initial aggregate poverty gap index, initial aggregate poverty rate).
        :rtype: tuple
        """
        initial_gaps = [
            self.country_results[country].initial_gap_index
            for country in self.countries
        ]
        initial_rates = [
            self.country_results[country].initial_rate for country in self.countries
        ]

        initial_welfare = [
            self.country_results[country].initial_welfare for country in self.countries
        ]
        weights = np.array(
            [self.country_weights["weight"][country] for country in self.countries]
        )
        pop_weighted_initial_poverty_gap_index = (
            np.array(initial_gaps) * weights
        ).sum()
        pop_weighted_initial_poverty_rate = (np.array(initial_rates) * weights).sum()

        pop_weighted_initial_welfare = (np.array(initial_welfare) * weights).sum()

        return (
            pop_weighted_initial_poverty_gap_index,
            pop_weighted_initial_poverty_rate,
            pop_weighted_initial_welfare,
        )

    def get_min_aggregate_gap_index_and_rate(self):
        """
        Get minimum population-weighted poverty gap index and rate achieved by method.
        :return: Tuple of (minimum aggregate poverty gap index, minimum aggregate poverty rate).
        :rtype: tuple
        """
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
        """
        Get maximum of minimum poverty gap index and rate across countries.
        :return: Tuple of (maximum of minimum poverty gap index, maximum of minimum poverty rate).
        :rtype: tuple
        """
        max_min_poverty_gap = max(
            [self.country_results[country].min_gap_index for country in self.countries]
        )

        rates = [self.country_results[country].min_rate for country in self.countries]
        max_min_poverty_rate = max(rates)

        return max_min_poverty_gap, max_min_poverty_rate

    def _get_aggregate_interpolators(self):
        """
        Set aggregate interpolators for poverty gap index to cost, poverty rate to cost, and poverty rate to poverty gap index.
        """
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

        # Let wc_rate be the national poverty rate target for all countries.
        # Build an interpolator from wc_rate to actual population-weighted poverty gap index and rate.

        agg_gaps = 0.0
        agg_rates = 0.0
        agg_welfare = 0.0
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
            weighted_country_actual_welfare = (
                country_result.rate_to_welfare_interpolator(country_actual_rates)
                * weights[i]
            )
            agg_welfare += weighted_country_actual_welfare

            agg_rates += weighted_country_actual_rates
            agg_gaps += weighted_country_actual_gaps

        aggregate_interpolator_wc_rate_to_actual_gap = interp1d(
            wc_rates, agg_gaps, kind="linear"
        )
        aggregate_interpolator_wc_rate_to_actual_rate = interp1d(
            wc_rates, agg_rates, kind="linear"
        )
        aggregate_interpolator_wc_rate_to_actual_welfare = interp1d(
            wc_rates, agg_welfare, kind="linear"
        )

        # Build an interpolator from wc_rate to aggregate cost by
        # (1) mapping wc_rate to actual poverty rate for each country,
        # (2) mapping actual poverty rate to cost for each country,
        # (3) summing costs across countries.
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
        actual_welfares = aggregate_interpolator_wc_rate_to_actual_welfare(wc_rates)

        self.aggregate_interpolator_rate_to_cost = interp1d(
            actual_rates, actual_costs, kind="linear"
        )

        self.aggregate_interpolator_gap_to_cost = interp1d(
            actual_gaps, actual_costs, kind="linear"
        )
        self.aggregate_interpolator_rate_to_gap = interp1d(
            actual_rates, actual_gaps, kind="linear"
        )

        self.aggregate_interpolator_rate_to_welfare = interp1d(
            actual_rates, actual_welfares, kind="linear"
        )

        self.aggregate_interpolator_welfare_to_cost = interp1d(
            actual_welfares, actual_costs, kind="linear"
        )
        self.aggregate_interpolator_rate_domain = (min(actual_rates), max(actual_rates))
        self.aggregate_interpolator_gap_domain = (min(actual_gaps), max(actual_gaps))
        self.aggregate_interpolator_welfare_domain = (
            min(actual_welfares),
            max(actual_welfares),
        )
