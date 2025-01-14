from opt_targeted_transfers.dataset_utils import Dataset
from opt_targeted_transfers.prediction import get_prediction_function
from opt_targeted_transfers.density_estimation import get_cond_density_estimator
from opt_targeted_transfers.knapsack import compute_alpha_opt_policies
from opt_targeted_transfers.oracle import (
    run_oracle_poverty_rate,
    run_oracle_poverty_gap_floor_scheme,
    run_oracle_poverty_gap_lift_to_line_scheme,
)

from opt_targeted_transfers.quantile_regression import get_quantile_regressor
from opt_targeted_transfers.evaluate import (
    post_transfer_metrics,
    expected_value_transfers,
    policy_cost,
)
from opt_targeted_transfers.conditional_improvement import (
    get_conditional_improvement_regressor,
)
from opt_targeted_transfers.reporting import write_result

import dill as pickle
import numpy as np
from bisect import bisect_left

import torch
import numpy as np
import tqdm
import copy

from opt_targeted_transfers.dataset_utils import standardize


class TargetedTransfers:
    """
    Base class for TargetedTransfers
    """

    def __init__(
        self,
        c_bar=2.15,
        budget=None,
    ):
        self.c_bar = c_bar
        self.budget = budget
        self.name = None

    def fit(self, train_dataset, validation_dataset):
        pass

    def run_opt(self, test_covariate_dataset):
        pass

    def set_budget(self, budget):
        self.budget = budget
        self.assignments = None

    def evaluate(self, test_dataset):
        """
        Evaluate optimal policy.

        :param test_dataset: The test dataset.
        :type test_dataset: Dataset
        :return: A dictionary of evaluation results.
        :rtype: dict
        """
        if self.assignments is None:
            raise ValueError("Must run run_opt before evaluate")

        result = post_transfer_metrics(test_dataset, self.assignments, self.c_bar)
        d = len(test_dataset.covs)
        result.update(
            {
                "budget": self.budget,
                "policy_type": self.name,
                "d": d, 
            }
        )
        return result

    def evaluate_equity(self, test_dataset, path=None):
        """
        Evaluate equity of optimal policy.

        :param test_dataset: The test dataset.
        :type test_dataset: Dataset
        :return: A dictionary of evaluation results.
        :rtype: dict
        """
        if self.assigments is None:
            raise ValueError("Must run run_opt before evaluate_equity")

        d = len(test_dataset.covs)
        all_transfers_ev = expected_value_transfers(test_dataset, self.assignments)

        _, y_test, _ = test_dataset.get_data()

        for i in range(len(all_transfers_ev)):
            write_result(
                path, {"consumption": y_test[i], "ev_transfer": all_transfers_ev[i]}
            )

    def compute_auc(self, test_dataset, metrics, budgets, test_covariate_dataset=None):
        """
        Compute the AUC for a list of budgets.

        :param test_covariate_dataset: The dataset that include the covariates and weights from the test set.
        :type test_covariate_dataset: Dataset
        :param test_dataset: The test dataset.
        :type test_dataset: Dataset
        :param metrics: The metrics to evaluate the policy.
        :type metrics: list
        :param budgets: The list of budgets in increasing order.
        :type budgets: list
        :return: A dictionary of AUC values.
        :rtype: dict
        """
        res = {metric: {"auc": 0.0, "results": []} for metric in metrics}
        for budget in budgets:
            self.set_budget(budget)
            if "oracle" in self.name:
                self.run_opt(test_dataset)
            else:
                self.run_opt(test_covariate_dataset=test_covariate_dataset)
            evaluate_res = self.evaluate(test_dataset)
            for metric in metrics:
                res[metric]["results"].append(evaluate_res[metric])

        for metric in metrics:
            res[metric]["auc"] = np.trapz(y=res[metric]["results"], x=budgets)

        return res


class RateTargetedTransfers(TargetedTransfers):
    """
    Computes the optimal rate targeting transfer policy.
    """

    def __init__(self, c_bar=2.15, budget=None):
        """
        Initialize a new instance of the UnconditionalTargetedTransfers class.
        :param c_bar: The minimum threshold value (poverty line). Defaults to 2.15.
        :type c_bar: float
        :param tolerance: The tolerance. Defaults to None.
        :type tolerance: float or None
        """
        super().__init__(c_bar=c_bar, budget=budget)
        self.name = "continuous_rate"
        self.density_estimator = None

    def fit(
        self,
        train_dataset,
        validation_dataset,
        n_bins=100,
        n_knots=4,
        degree=4,
        truncation_upper_value=10,
        n_epochs=300,
    ):
        """
        Fitting the conditional density.

        :param train_dataset: The dataset used for training the regressors
        :type train_dataset: Dataset
        :param validation_dataset: The dataset used for validation.
        :type validation_dataset: Dataset
        :param n_bins: The number of bins to use for the outcome space. Defaults to 100.
        :type n_bins: int
        :param n_knots: The number of knots to use for the spline basis functions. Defaults to 4.
        :type n_knots: int
        :param degree: The degree of the spline basis functions. Defaults to 3.
        :type degree: int
        :param truncation_upper_value: The upper value to use for truncating the outcome variables. Defaults to 10.
        :type truncation_upper_value: float
        :param n_epochs: The number of epochs to train the model. Defaults to 300.
        :type n_epochs: int
        """
        density_estimator = get_cond_density_estimator(
            train_dataset,
            validation_dataset,
            n_bins=n_bins,
            n_knots=n_knots,
            degree=degree,
            truncation_upper_value=truncation_upper_value,
            n_epochs=n_epochs,
        )

        pickle.dump(
            density_estimator,
            open("{}_cond_density_estimator.pickle".format(self.name), "wb"),
        )

        self.density_estimator = density_estimator

    def run_opt(
        self,
        test_covariate_dataset,
        min_alpha=None,
        max_alpha=None,
        n_alpha=200,
        path=None,
    ):
        """
        Run the optimization algorithm.

        :param test_covariate_dataset: The dataset that include the covariates and weights from the test set.
        :type test_covariate_dataset: Dataset
        :param min_alpha: The minimum value of alpha for optimization. Defaults to None.
        :type min_alpha: float or None
        :param max_alpha: The maximum value of alpha for optimization. Defaults to None.
        :type max_alpha: float or None
        :param n_alpha: The number of alpha values to consider. Defaults to 200.
        :type n_alpha: int
        :param path: The path to save the optimization results. Defaults to None.
        :type path: str or None
        """
        if self.density_estimator is None:
            assert False, "Need to first set density estimator"

        (
            all_opt_assignments,
            poverty_rates,
            alphas,
        ) = compute_alpha_opt_policies(
            test_covariate_dataset,
            self.density_estimator,
            budget=self.budget,
            c_bar=self.c_bar,
            min_alpha=min_alpha,
            max_alpha=max_alpha,
            n_alpha=n_alpha,
            path=path,
        )

        idx = np.argmin(poverty_rates)
        self.assignments = all_opt_assignments[idx]
        return self.assignments

    def compute_auc(
        self,
        test_covariate_dataset,
        test_dataset,
        metrics,
        budgets,
        min_alpha=None,
        max_alpha=None,
        n_alpha=200,
        path=None,
    ):
        """
        Compute the AUC for a list of budgets.

        :param test_covariate_dataset: The dataset that include the covariates and weights from the test set.
        :type test_covariate_dataset: Dataset
        :param test_dataset: The test dataset.
        :type test_dataset: Dataset
        :param metrics: The metrics to evaluate the policy.
        :type metrics: list
        :param budgets: The list of budgets in increasing order.
        :type budgets: list
        :return: A dictionary of AUC values.
        :rtype: dict
        """
        res = {metric: {"auc": 0.0, "results": []} for metric in metrics}
        for budget in budgets:
            self.set_budget(budget)
            if path is not None:
                full_path = f"budget={budget}_" + path
            else:
                full_path = None
            self.run_opt(
                test_covariate_dataset,
                min_alpha=min_alpha,
                max_alpha=max_alpha,
                n_alpha=n_alpha,
                path=full_path,
            )
            evaluate_res = self.evaluate(test_dataset)
            for metric in metrics:
                res[metric]["results"].append(evaluate_res[metric])

        for metric in metrics:
            res[metric]["auc"] = np.trapz(y=res[metric]["results"], x=budgets)

        return res


class BinaryTargetedTransfers(TargetedTransfers):
    def __init__(self, c_bar=2.15, budget=None, n_regressors=20):

        super().__init__(c_bar=c_bar, budget=budget)

        self.t_to_household_estimator_map = None
        self.n_regressors = n_regressors
        self.candidate_t_values = np.linspace(0.01, self.c_bar, self.n_regressors)

    def fit(
        self,
        train_dataset,
    ):
        pass

    def optimize_transfers_for_budget_grid(self, test_covariate_dataset, budgets):
        """
        Computes transfers for each budget in the list of budgets. Enables calling
        run_opt for each budget in the list.
        """

        if self.t_to_household_estimator_map is None:
            raise ValueError("Need to run fit before a policy can be computed")

        X_test, r_test = test_covariate_dataset.get_data()

        # for each t, order the households in the test set
        t_to_ordered_households_map = (
            dict()
        )  # rank households from largest to smallest benefit
        t_to_estimated_benefits_map = dict()

        for t, household_benefit_estimator in self.t_to_household_estimator_map.items():
            estimated_benefits = household_benefit_estimator(X_test)
            t_to_estimated_benefits_map[t] = estimated_benefits

            sorting_indices = np.argsort(estimated_benefits)[::-1]
            t_to_ordered_households_map[t] = sorting_indices

        # For each budget, select optimal transfer value on the test set
        self.budget_to_households_map = dict()
        self.budget_to_t_map = dict()

        for budget in budgets:

            highest_estimated_benefits = -1
            best_household_list = []
            best_t = None

            for t in self.candidate_t_values:
                ordered_households = t_to_ordered_households_map[t]
                estimated_benefits = t_to_estimated_benefits_map[t]

                indices_to_receive_transfers = self._get_indices_to_receive_transfers(
                    test_covariate_dataset, ordered_households, t, budget
                )
                total_estimated_benefits = (
                    estimated_benefits[indices_to_receive_transfers]
                    * r_test[indices_to_receive_transfers]
                ).sum()

                # sanity check
                assert total_estimated_benefits >= 0

                if total_estimated_benefits > highest_estimated_benefits:
                    highest_estimated_benefits = total_estimated_benefits
                    best_household_list = indices_to_receive_transfers
                    best_t = t

            assert best_household_list is not None
            assert best_t is not None

            self.budget_to_households_map[budget] = best_household_list
            self.budget_to_t_map[budget] = best_t

    def _get_indices_to_receive_transfers(
        self, test_covariate_dataset, household_idx_ranked_by_benefit, t, budget
    ):

        _, r_test = test_covariate_dataset.get_data()

        pop_weight_receive_transfers = budget / t
        weights_ranked_by_benefit = r_test[household_idx_ranked_by_benefit]
        cumsum_weights = np.cumsum(weights_ranked_by_benefit)
        indicator_receive_transfers = cumsum_weights <= pop_weight_receive_transfers
        idx_receive_transfers = household_idx_ranked_by_benefit[
            indicator_receive_transfers
        ]

        assert r_test[idx_receive_transfers].sum() * t <= budget
        return idx_receive_transfers

    def run_opt(self, test_covariate_dataset):

        if self.budget_to_households_map is None:
            raise ValueError(
                "Must run optimize_transfers_for_budget_grid before run_opt for "
                "binary targeting"
            )

        assert self.budget is not None

        if self.budget not in self.budget_to_households_map.keys():
            raise ValueError(
                f"budget {self.budget} was not included in list provided to "
                "optimize_transfers_for_budget_grid."
            )

        indices_to_receive_transfers = self.budget_to_households_map[self.budget]
        best_t = self.budget_to_t_map[self.budget]

        assignments = {i: [(0.0, 1.0)] for i in range(len(test_covariate_dataset))}
        for i in indices_to_receive_transfers:
            assignments[i] = [(best_t, 1.0)]
        self.assignments = assignments
        return assignments


class BinaryGapTargetedTransfers(BinaryTargetedTransfers):
    def __init__(self, c_bar=2.15, budget=None, n_regressors=20):

        super().__init__(
            c_bar=c_bar, budget=budget, n_regressors=n_regressors
        )

        self.name = "binary_gap"

    def fit(
        self,
        train_dataset,
        validation_dataset,
        n_layers=1,
        n_hidden_units=256,
        lr=5e-3,
        n_epochs=300,
        seed=123456,
    ):

        # For each transfer size t, fit benefit estimator using training data
        self.t_to_household_estimator_map = dict()
        for transfer_size in self.candidate_t_values:
            self.t_to_household_estimator_map[transfer_size] = (
                get_conditional_improvement_regressor(
                    loss_type=self.name,
                    train_dataset=train_dataset,
                    validation_dataset=validation_dataset,
                    t=transfer_size,
                    c_bar=self.c_bar,
                    n_layers=n_layers,
                    n_hidden_units=n_hidden_units,
                    lr=lr,
                    n_epochs=n_epochs,
                    seed=seed,
                )
            )


class BinaryRateTargetedTransfers(BinaryTargetedTransfers):
    def __init__(self, c_bar=2.15, budget=None, n_regressors=20):

        super().__init__(
            c_bar=c_bar, budget=budget, n_regressors=n_regressors
        )

        self.name = "binary_rate"

    def fit(
        self,
        train_dataset,
        validation_dataset,
        n_layers=1,
        n_hidden_units=256,
        lr=5e-3,
        n_epochs=300,
        seed=123456,
    ):

        # For each transfer size t, fit benefit estimator using training data
        self.t_to_household_estimator_map = dict()
        for transfer_size in self.candidate_t_values:
            self.t_to_household_estimator_map[transfer_size] = (
                get_conditional_improvement_regressor(
                    loss_type=self.name,
                    train_dataset=train_dataset,
                    validation_dataset=validation_dataset,
                    t=transfer_size,
                    c_bar=self.c_bar,
                    n_layers=n_layers,
                    n_hidden_units=n_hidden_units,
                    lr=lr,
                    n_epochs=n_epochs,
                    seed=seed,
                )
            )


class GapTargetedTransfers(TargetedTransfers):
    """
    Poverty-gap targeting.
    """

    def __init__(self, c_bar=2.15, budget=None, n_regressors=20):
        """
        :param c_bar: The minimum threshold value (poverty line). Defaults to 2.15.
        :type c_bar: float
        """

        super().__init__(c_bar=c_bar, budget=budget)
        self.n_regressors = n_regressors
        self.name = "continuous_gap"
        self.quantile_regressors = None

    def fit(
        self,
        train_dataset,
        validation_dataset,
        n_layers=1,
        n_hidden_units=256,
        lr=5e-3,
        n_epochs=300,
        seed=123456,
    ):
        """
        Fitting the quantile regression.

        :param train_dataset: The dataset used for training the regressors
        :type train_dataset: Dataset
        :param n_regressors: The number of quantile regressors to fit. Defaults to 20.
        :type n_regressors: int
        :param n_layers: The number of hidden layers in the neural network. Defaults to 1.
        :type n_layers: int
        :param n_hidden_units: The number of hidden units in each hidden layer. Defaults to 256.
        :type n_hidden_units: int
        :param lr: The learning rate for training the neural network. Defaults to 5e-3.
        :type lr: float
        :param n_epochs: The number of epochs for training the neural network. Defaults to 300.
        :type n_epochs: int
        :param seed: The random seed for reproducibility. Defaults to 123456.
        :type seed: int
        """

        quantiles = np.linspace(0.05, 0.95, self.n_regressors)
        self.quantile_regressors = dict()

        for quantile in quantiles:
            print("Fitting quantile regressor for quantile {}".format(quantile))
            quantile_regressor = get_quantile_regressor(
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                quantile=quantile,
                n_layers=n_layers,
                n_hidden_units=n_hidden_units,
                lr=lr,
                n_epochs=n_epochs,
                seed=seed,
            )
            self.quantile_regressors[quantile] = quantile_regressor

    def run_opt(self, test_covariate_dataset):
        if self.quantile_regressors is None:
            raise ValueError("Missing quantile regressors - run fit first.")
        # For each quantile regressor, compute the corresponding assignments and policy cost
        costs = []
        all_assignments = []
        X_test, r_test = test_covariate_dataset.get_data()
        for quantile in reversed(sorted(self.quantile_regressors.keys())):
            quantile_regressor = self.quantile_regressors[quantile]
            conditional_quantile = quantile_regressor(X_test)
            transfer = np.maximum(self.c_bar - conditional_quantile, 0)
            assignments = {x_idx: [] for x_idx in range(len(X_test))}
            for i in range(len(X_test)):
                assignments[i].append((transfer[i].item(), 1.0))
            cost = policy_cost(test_covariate_dataset, assignments)
            costs.append(cost)
            all_assignments.append(assignments)
        # Get the assignments that have cost lower than budget
        idx = bisect_left(costs, self.budget)
        self.assignments = all_assignments[idx - 1]

        # TODO: Add back interpolation here if idx-1 is between two entries.
        return all_assignments[idx - 1]


class OracleGapTargetedTransfers(TargetedTransfers):
    """
    There is not one well-defined optimal policy to minimize the post-transfer poverty gap: As long as
    every dollar goes to households below the poverty line, the reduction in poverty gap will be the optimal.
    This class implements two policies:
      * lifting as many households as possible to the poverty line, with the restriction that a less-poor
        household does not receive more than a poorer household: this amounts to iteratively raising households
        to the poverty line, starting from the poorest, until the specified tolerance is reached.
      * raising a poverty "floor" until the desired tolerance is reached: a floor is a wealth level below which no
        household is permitted to be. Any household below that floor receives a transfer of the appropriate size
        to raise them to the floor. The floor is set to minimally satisfy the specified tolerance.
    """

    def __init__(self, c_bar=2.15, budget=None, scheme="lift_to_line"):

        assert scheme in ("lift_to_line", "floor")

        super().__init__(c_bar=c_bar, budget=budget)
        self.name = "oracle_gap"
        self.scheme = scheme

    def run_opt(self, test_dataset):

        if self.scheme == "lift_to_line":
            assignments = run_oracle_poverty_gap_lift_to_line_scheme(
                test_dataset, budget=self.budget, c_bar=self.c_bar
            )

        elif self.scheme == "floor":
            assignments = run_oracle_poverty_gap_floor_scheme(
                test_dataset, budget=self.budget, c_bar=self.c_bar
            )
        self.assignments = assignments
        return assignments


class OracleRateTargetedTransfers(TargetedTransfers):
    def __init__(self, c_bar=2.15, budget=None):

        super().__init__(c_bar=c_bar, budget=budget)
        self.name = "oracle_rate"

    def run_opt(self, test_dataset):

        assignments = run_oracle_poverty_rate(
            test_dataset,
            c_bar=self.c_bar,
            budget=self.budget,
        )
        self.assignments = assignments
        return assignments
