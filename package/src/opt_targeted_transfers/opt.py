from opt_targeted_transfers.dataset_utils import Dataset
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
from opt_targeted_transfers.prediction import (
    get_pmt_nn_regressor,
    get_pmt_lasso_regressor,
)
from opt_targeted_transfers.conditional_improvement import (
    get_conditional_improvement_regressor,
    get_avg_estimated_benefit,
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
        c_bar=3.0,
        budget=None,
    ):
        self.c_bar = c_bar
        self.budget = budget
        self.name = None

    def fit(self, train_dataset, validation_dataset, device):
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
        if self.assignments is None:
            raise ValueError("Must run run_opt before evaluate_equity")

        d = len(test_dataset.covs)
        all_transfers_ev = expected_value_transfers(self.assignments)

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

    def __init__(self, c_bar=3.0, budget=None):
        """
        Initialize a new instance of the UnconditionalTargetedTransfers class.
        :param c_bar: The minimum threshold value (poverty line). Defaults to 3.0.
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
        device="cpu",
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
        :param device: The device to use for training the model. Defaults to 'cpu'.
        :type device: str
        """
        density_estimator = get_cond_density_estimator(
            train_dataset,
            validation_dataset,
            n_bins=n_bins,
            n_knots=n_knots,
            degree=degree,
            truncation_upper_value=truncation_upper_value,
            n_epochs=n_epochs,
            device=device,
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
    def __init__(self, c_bar=3.0, budget=None, n_regressors=20):

        super().__init__(c_bar=c_bar, budget=budget)

        self.t_to_household_estimator_map = None
        self.n_regressors = n_regressors
        self.candidate_t_values = np.linspace(0.01, self.c_bar, self.n_regressors)

    def fit(
        self,
        train_dataset,
    ):
        pass

    def get_opt_transfer_sizes_given_budget_grid(self, validation_dataset, budgets):
        """
        Computes transfer size for each budget in the list of budgets. Enables calling
        run_opt for each budget in the list.
        """

        if self.t_to_household_estimator_map is None:
            raise ValueError("Need to run fit before a policy can be computed")

        # For each budget, select optimal transfer value on the test set
        self.budget_to_t_map = dict()

        t_to_ordered_households = dict()
        t_to_estimated_benefits = dict()
        for t in self.candidate_t_values:
            regressor = self.t_to_household_estimator_map[t]
            X_val, _, r_val = validation_dataset.get_data()
            estimated_benefits = regressor(X_val)
            ordered_households = np.argsort(estimated_benefits)[::-1]
            t_to_ordered_households[t] = ordered_households
            t_to_estimated_benefits[t] = estimated_benefits

        for budget in budgets:
            highest_estimated_benefits = -float("inf")
            best_t = None

            candidates_given_budget = reversed(
                self.candidate_t_values[self.candidate_t_values >= budget]
            )
            for t in candidates_given_budget:
                idx_to_receive_transfers = self._get_indices_to_receive_transfers_exact(
                    r_val, t_to_ordered_households[t], t, budget
                )
                total_estimated_benefits = self._get_avg_estimated_benefit(
                    validation_dataset, t, idx_to_receive_transfers
                )

                assert total_estimated_benefits >= 0
                if total_estimated_benefits > highest_estimated_benefits:
                    # if this transfer size has the highest estimated benefits, then this is the best so far.
                    highest_estimated_benefits = total_estimated_benefits
                    best_t = t

            assert best_t is not None
            self.budget_to_t_map[budget] = best_t

    def _get_indices_to_receive_transfers_exact(
        self, r, household_idx_ranked_by_benefit, t, budget
    ):
        pop_weight_receive_transfers = budget / t
        weights_ranked_by_benefit = r[household_idx_ranked_by_benefit]
        cumsum_weights = np.cumsum(weights_ranked_by_benefit)
        # cumsum_weights /= cumsum_weights[
        #     -1
        # ]  # should have no effect if cumsum_weights[-1] ==1.0, but should handle a numerical issue in the case cumsum_weights[-1] is slightly greater than 1.
        indicator_receive_transfers = cumsum_weights <= pop_weight_receive_transfers
        idx_receive_transfers = household_idx_ranked_by_benefit[
            indicator_receive_transfers
        ]
        return idx_receive_transfers

    def run_opt(self, test_covariate_dataset):

        if self.budget_to_t_map is None:
            raise ValueError("Run get_opt_transfer_sizes_given_budget_grid first")

        assert self.budget is not None

        if self.budget not in self.budget_to_t_map.keys():
            raise ValueError(
                f"budget {self.budget} was not included provided to get_opt_transfer_sizes_given_budget_grid"
            )

        X_test, r_test = test_covariate_dataset.get_data()

        best_t = self.budget_to_t_map[self.budget]
        regressor = self.t_to_household_estimator_map[best_t]
        estimated_benefits = regressor(X_test)
        ordered_households = np.argsort(estimated_benefits)[::-1]
        indices_to_receive_transfers = self._get_indices_to_receive_transfers_exact(
            r_test, ordered_households, best_t, self.budget
        )

        assignments = {i: [(0.0, 1.0)] for i in range(len(test_covariate_dataset))}
        for i in indices_to_receive_transfers:
            assignments[i] = [(best_t, 1.0)]
        self.assignments = assignments
        return assignments

    def _get_avg_estimated_benefit(
        self, validation_dataset, t, idx_to_receive_transfers
    ):
        avg_estimated_benefits = get_avg_estimated_benefit(
            validation_dataset, self.name, idx_to_receive_transfers, t, c_bar=self.c_bar
        )
        return avg_estimated_benefits


class BinaryGapTargetedTransfers(BinaryTargetedTransfers):
    def __init__(self, c_bar=3.0, budget=None, n_regressors=20):

        super().__init__(c_bar=c_bar, budget=budget, n_regressors=n_regressors)

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
        device="cpu",
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
                    device=device,
                )
            )


class BinaryRateTargetedTransfers(BinaryTargetedTransfers):
    def __init__(self, c_bar=3.0, budget=None, n_regressors=20):

        super().__init__(c_bar=c_bar, budget=budget, n_regressors=n_regressors)

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
        device="cpu",
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
                    device=device,
                )
            )


class GapTargetedTransfers(TargetedTransfers):
    """
    Poverty-gap targeting.
    """

    def __init__(self, c_bar=3.0, budget=None, n_regressors=20):
        """
        :param c_bar: The minimum threshold value (poverty line). Defaults to 3.0.
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
        device="cpu",
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
        :param device: The device to use for training the model. Defaults to 'cpu'.
        :type device: str
        """

        quantiles = np.linspace(0.0, 1.0, self.n_regressors)
        self.quantile_regressors = dict()
        self.quantiles = quantiles

        for quantile in quantiles[1:]:
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
                device=device,
            )
            self.quantile_regressors[quantile] = quantile_regressor

        def get_zero_quantile_regressor(X):
            return np.zeros((X.shape[0],))

        self.quantile_regressors[0.0] = get_zero_quantile_regressor

    def _interpolate_conditional_quantile(self, X, quantile):
        """
        Evaluates the conditional quantile
        """

        quantile_regressors = self.quantile_regressors
        if quantile_regressors is None:
            raise ValueError("Missing quantile regressors - run fit first.")

        quantiles = list(quantile_regressors.keys())
        quantiles.sort()

        quantile_index = bisect_left(quantiles, quantile)

        # if quantile_index == len(quantiles), then quantile is > all evaluated quantiles.
        # if quantile_index == 0 then quantile is <= all evaluated quantiles.
        # In that case for now I don't attempt to fake interpolation. This means
        # it's important to fit a quantile near or at zero.
        if quantile_index == len(quantiles):

            baseline_quantile_wealth_level = quantile_regressors[
                quantiles[quantile_index - 1]
            ](X)

        elif (quantile == quantiles[quantile_index]) or (quantile_index == 0):

            baseline_quantile_wealth_level = quantile_regressors[
                quantiles[quantile_index]
            ](X)

        # interpolate
        else:
            quantile_index_low = quantile_index - 1
            quantile_index_high = quantile_index

            assert quantile > quantiles[quantile_index_low]
            assert quantile < quantiles[quantile_index_high]

            interpolation_factor = (quantile - quantiles[quantile_index_low]) / (
                quantiles[quantile_index_high] - quantiles[quantile_index_low]
            )

            baseline_quantile_wealth_level_low = quantile_regressors[
                quantiles[quantile_index_low]
            ](X)
            baseline_quantile_wealth_level_high = quantile_regressors[
                quantiles[quantile_index_high]
            ](X)

            baseline_quantile_wealth_level = (
                (1 - interpolation_factor) * baseline_quantile_wealth_level_low
                + interpolation_factor * baseline_quantile_wealth_level_high
            )

        return baseline_quantile_wealth_level

    def _get_assignments_for_lambda(self, X, lambda_):

        quantile_regressors = self.quantile_regressors
        if quantile_regressors is None:
            raise ValueError("Missing quantile regressors - run fit first.")
        conditional_quantiles = self._interpolate_conditional_quantile(X, lambda_)
        transfer = np.maximum(self.c_bar - conditional_quantiles, 0)
        assignments = {x_idx: [] for x_idx in range(len(X))}
        for i in range(len(X)):
            assignments[i].append((transfer[i].item(), 1.0))
        return assignments

    def run_opt(self, test_covariate_dataset):
        if self.quantile_regressors is None:
            raise ValueError("Missing quantile regressors - run fit first.")

        X_test, r_test = test_covariate_dataset.get_data()
        r_test = r_test / r_test.sum()  # normalize weights to sum to

        low = 0
        high = 1.0
        low_assignments = self._get_assignments_for_lambda(X_test, low)
        low_cost = policy_cost(test_covariate_dataset, low_assignments)
        if self.budget >= low_cost:
            # This only happens if budget > poverty line, in which case can just give everyone the same transfer.
            assignments = {
                i: [(self.budget, 1.0)] for i in range(len(test_covariate_dataset))
            }
            self.assignments = assignments
            return assignments

        lambda_value = (high + low) / 2
        assignments = self._get_assignments_for_lambda(X_test, lambda_value)
        lamb_cost = policy_cost(test_covariate_dataset, assignments)

        while np.abs(lamb_cost - self.budget) > 1e-3:
            if lamb_cost > self.budget:
                low = lambda_value
            else:
                high = lambda_value
            next_lambda_value = (high + low) / 2
            assignments = self._get_assignments_for_lambda(X_test, next_lambda_value)
            next_lamb_cost = policy_cost(test_covariate_dataset, assignments)
            lambda_value = next_lambda_value
            lamb_cost = next_lamb_cost
        self.assignments = assignments
        return assignments

        # # For each quantile regressor, compute the corresponding assignments and policy cost
        # costs = []
        # all_assignments = []
        # X_test, r_test = test_covariate_dataset.get_data()
        # r_test = r_test / r_test.sum()  # normalize weights to sum to 1
        # lambda_values = reversed(np.linspace(0.0, 1.0, 200))
        # for lambda_ in lambda_values:
        #     assignments = self._get_assignments_for_lambda(X_test, lambda_)
        #     cost = policy_cost(test_covariate_dataset, assignments)
        #     costs.append(cost)
        #     all_assignments.append(assignments)
        # idx = bisect_left(costs, self.budget)

        # if idx == 0:
        #     self.assignments = all_assignments[0]
        #     return all_assignments[0]
        # if idx >= 1 and idx < len(costs):
        #     if np.abs(costs[idx] - self.budget) < 1e-5:
        #         self.assignments = all_assignments[idx]
        #         return all_assignments[idx]
        #     elif costs[idx] > self.budget:
        #         self.assignments = all_assignments[idx - 1]
        #         return all_assignments[idx - 1]
        # if idx == len(costs):
        #     self.assignments = all_assignments[-1]
        #     return all_assignments[-1]


class OracleGapTargetedTransfers(TargetedTransfers):
    """
    There is not one well-defined optimal policy to minimize the post-transfer poverty gap: As long as
    every dollar goes to households below the poverty line, the reduction in poverty gap will be the optimal.
    This class implements two policies:
      * "lift_to_line" which is the optimal oracle rate-minimizing policy
      * "consumption_floor" which sets the consumption floor for units. Any household below that floor receives a transfer of the appropriate size
        to raise them to the floor. The floor is set to minimally satisfy the specified tolerance. This is the weakly-equitable rate-minimizing policy.
    """

    def __init__(self, c_bar=3.0, budget=None, scheme="lift_to_line"):

        assert scheme in ("lift_to_line", "consumption_floor")

        super().__init__(c_bar=c_bar, budget=budget)
        self.name = "oracle_gap"
        self.scheme = scheme

    def run_opt(self, test_dataset):

        if self.scheme == "lift_to_line":
            assignments = run_oracle_poverty_gap_lift_to_line_scheme(
                test_dataset, budget=self.budget, c_bar=self.c_bar
            )

        elif self.scheme == "consumption_floor":
            assignments = run_oracle_poverty_gap_floor_scheme(
                test_dataset, budget=self.budget, c_bar=self.c_bar
            )
        self.assignments = assignments
        return assignments


class OracleRateTargetedTransfers(TargetedTransfers):
    def __init__(self, c_bar=3.0, budget=None):

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


class UBITargetedTransfers(TargetedTransfers):
    def __init__(self, c_bar, budget=None):
        """
        :param c_bar: The minimum threshold value (poverty line). Defaults to 3.0.
        :type c_bar: float
        """

        super().__init__(c_bar=c_bar, budget=budget)
        self.name = "ubi"

    def run_opt(self, test_covariate_dataset):
        """
        Assign the transfer value to all households
        """
        assert self.budget is not None
        assignments = {
            i: [(self.budget, 1.0)] for i in range(len(test_covariate_dataset))
        }
        self.assignments = assignments
        return assignments


class ModernPMTTargetedTransfers(BinaryTargetedTransfers):
    """
    Modern PMT style targeting
    """

    def __init__(self, c_bar=3.0, budget=None, transfer_value=1.0):
        """
        :param c_bar: The minimum threshold value (poverty line). Defaults to 3.0.
        :type c_bar: float
        """

        super().__init__(c_bar=c_bar, budget=budget)
        self.name = "pmt"
        self.transfer_value = transfer_value

    def fit(
        self,
        train_dataset,
        validation_dataset,
        n_layers=1,
        n_hidden_units=256,
        lr=5e-3,
        n_epochs=300,
        seed=123456,
        device="cpu",
    ):
        """
        :param n_layers: The number of hidden layers in the neural network. Defaults to 1.
        :type n_layers: int
        :param n_hidden_units: The number of hidden units in each hidden layer. Defaults to 256.
        :type n_hidden_units: int
        :param lr: The learning rate for training the neural network. Defaults to 5e-3.
        :type lr: float
        :param n_epochs: The number of epochs for training the neural network. Defaults to 300.
        :type n_epochs: int
        """

        self.consumption_predictor = get_pmt_nn_regressor(
            train_dataset,
            validation_dataset,
            n_layers=n_layers,
            n_hidden_units=n_hidden_units,
            lr=lr,
            n_epochs=n_epochs,
            seed=seed,
            device=device,
        )

    def run_opt(self, test_covariate_dataset):

        assert self.budget is not None

        X_test, r_test = test_covariate_dataset.get_data()

        estimated_benefits = self.consumption_predictor(X_test)
        ordered_households = np.argsort(estimated_benefits)
        indices_to_receive_transfers = self._get_indices_to_receive_transfers_exact(
            r_test, ordered_households, self.transfer_value, self.budget
        )

        assignments = {i: [(0.0, 1.0)] for i in range(len(test_covariate_dataset))}
        for i in indices_to_receive_transfers:
            assignments[i] = [(self.transfer_value, 1.0)]
        self.assignments = assignments
        return assignments


class PMTTargetedTransfers(BinaryTargetedTransfers):
    """
    PMT style targeting
    """

    def __init__(self, c_bar=3.0, budget=None, transfer_value=1.0):
        """
        :param c_bar: The minimum threshold value (poverty line). Defaults to 3.0.
        :type c_bar: float
        """

        super().__init__(c_bar=c_bar, budget=budget)
        self.name = "pmt"
        self.transfer_value = transfer_value

    def fit(self, train_dataset, validation_dataset, alpha=0.1):
        """
        Fitting linear regression
        """

        self.consumption_predictor = get_pmt_lasso_regressor(
            train_dataset, validation_dataset, alpha=alpha
        )

    def run_opt(self, test_covariate_dataset):

        assert self.budget is not None

        X_test, r_test = test_covariate_dataset.get_data()

        estimated_benefits = self.consumption_predictor(X_test)
        ordered_households = np.argsort(estimated_benefits)
        indices_to_receive_transfers = self._get_indices_to_receive_transfers_exact(
            r_test, ordered_households, self.transfer_value, self.budget
        )

        assignments = {i: [(0.0, 1.0)] for i in range(len(test_covariate_dataset))}
        for i in indices_to_receive_transfers:
            assignments[i] = [(self.transfer_value, 1.0)]
        self.assignments = assignments
        return assignments
