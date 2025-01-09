from opt_targeted_transfers.dataset_utils import Dataset
from opt_targeted_transfers.prediction import get_prediction_function
from opt_targeted_transfers.density_estimation import get_cond_density_estimator
from opt_targeted_transfers.knapsack import (
    compute_alpha_opt_policies,
)
from opt_targeted_transfers.oracle import (
    run_oracle_poverty_rate,
    run_oracle_poverty_gap_floor_scheme,
    run_oracle_poverty_gap_lift_to_line_scheme,
)

from opt_targeted_transfers.quantile_regression import get_quantile_regressor
from opt_targeted_transfers.evaluate import (
    post_transfer_metrics,
    expected_value_transfers,
    policy_cost
)
from opt_targeted_transfers.conditional_gap_improvement import get_conditional_gap_improvement_regressor
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
        self, c_bar=2.15, budget=None,
    ):
        self.c_bar = c_bar
        self.budget = budget
        self.opt_policy = None
        self.name = None

    def fit(self, train_dataset):
        pass

    def run_opt(self):
        pass

    def set_budget(self, budget):
        self.budget = budget

    def save_opt_policy(self, name):
        if self.opt_policy is None:
            assert False, "Need to run opt first"
        pickle.dump(
            self.opt_policy,
            open("{}.pickle".format(name), "wb"),
        )

    def evaluate(self, test_dataset):
        """
        Evaluate optimal policy.

        :param test_dataset: The test dataset.
        :type test_dataset: Dataset
        :return: A dictionary of evaluation results.
        :rtype: dict
        """

        if self.opt_policy is None:
            assert False, "Need to first run optimization"

        if "oracle" in self.name:
            result = post_transfer_metrics(
                test_dataset, self.opt_policy, self.c_bar, oracle=True
            )
        else:
            result = post_transfer_metrics(test_dataset, self.opt_policy, self.c_bar)

        d= len(test_dataset.covs)
        result.update(
            {
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
        if self.opt_policy is None:
            assert False, "Need to first run optimization"
        d = len(test_dataset.covs)

        if "oracle" in self.name:
            oracle = True
        else:
            oracle = False

        all_transfers_ev = expected_value_transfers(
            test_dataset, self.opt_policy, oracle=oracle
        )

        _, y_test, _ = test_dataset.get_data()

        for i in range(len(all_transfers_ev)):
            write_result(
                path, {"consumption": y_test[i], "ev_transfer": all_transfers_ev[i]}
            )


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
        super().__init__(
            c_bar=c_bar, budget=budget
        )
        self.name = "unconditional_rate"
        self.density_estimator = None
        self.opt_policy = None

    def fit(
        self,
        train_dataset,
        n_bins=100, 
        n_knots=4, 
        degree=4, 
        truncation_upper_value=10, 
        n_epochs=300
    ):
        """
        Fitting the conditional density.

        :param train_dataset: The dataset used for training the regressors
        :type train_dataset: Dataset
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
            t_alpha_joint_programs,
            total_transfers,
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

        idx = np.argmin(total_transfers)
        t_joint_program_est = t_alpha_joint_programs[idx]
        self.opt_policy = t_joint_program_est
        return t_joint_program_est


class GapTargetedTransfers(TargetedTransfers):
    """
    Poverty-gap targeting.
    """

    def __init__(self, c_bar=2.15, budget=None):
        """
        :param c_bar: The minimum threshold value (poverty line). Defaults to 2.15.
        :type c_bar: float
        """

        super().__init__(
            c_bar=c_bar, budget=budget
        )
        self.name = "gap"
        self.quantile_regressors = None

    def fit(
        self,
        train_dataset,
        n_regressors=20,
        n_layers=1,
        n_hidden_units=256,
        lr=5e-3, 
        n_epochs=300,
        seed=123456  
    ):
        """
        Fitting the quantile regression.

        :param train_dataset: The dataset used for training the regressors
        :type train_dataset: Dataset
        :param lambda_: The quantile for which the regressor is trained. Defaults to 0.5.
        :type lambda_: float
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

        quantiles = np.linspace(0.05, 0.95, n_regressors)
        self.quantile_regressors = dict()

        for quantile in quantiles:
            quantile_regressor = get_quantile_regressor(train_dataset, 
                                                    quantile=quantile, 
                                                    n_layers=n_layers, 
                                                    n_hidden_units=n_hidden_units, 
                                                    lr=lr, 
                                                    n_epochs=n_epochs,
                                                    seed=seed)
            self.quantile_regressor[quantile] = quantile_regressor

    def run_opt(self, test_covariate_dataset):
        if self.quantile_regressor is None:
            raise ValueError("Missing quantile regressors - run fit first.")
        #For each quantile regressor, compute the corresponding policy and policy cost
        costs = []
        policies = []
        for _,quantile_regressor in self.quantile_regressors.items():
            def t(X):
                conditional_quantile = quantile_regressor(X)
                transfer = np.maximum(self.c_bar - conditional_quantile, 0)
                assignments = {x_idx: [] for x_idx in range(len(X))}
                for i in range(len(X)):
                    assignments[i].append((transfer[i].item(), 1.0))
                return assignments
            cost = policy_cost(test_covariate_dataset, t)
            costs.append(cost)
            policies.append(t)
        # Find the policy that has policy cost lower than budget
        idx = bisect_left(costs, self.budget)
        return policies[idx]


class BinaryGapTargetedTransfers(TargetedTransfers):
    # TODO: Hit expected budget exactly by having one stochastic transfer

    def __init__(self, c_bar=2.15, budget=None):

        super().__init__(
            c_bar=c_bar,
            budget=budget
        )

        self.name = "binary_gap"
        self.t_to_household_estimator_map = None
        self.t = None

    def fit(
        self,
        train_dataset,
        n_transfer_values=20,
        n_layers=1, 
        n_hidden_units=256, 
        lr=5e-3, 
        n_epochs=300,
        seed=123456
    ):

        # For each transfer size t, fit benefit estimator using training data
        self.t_to_household_estimator_map = dict()

        transfer_sizes = np.linspace(0., 2.15, n_transfer_values)

        for transfer_size in transfer_sizes:
            self.t_to_household_estimator_map[transfer_size] = get_conditional_gap_improvement_regressor(train_dataset,
                                                                                             t=transfer_size,
                                                                                             c_bar=self.c_bar,
                                                                                             n_layers=n_layers,
                                                                                             n_hidden_units=n_hidden_units,
                                                                                             lr=lr,
                                                                                             n_epochs=n_epochs,
                                                                                             seed=seed)

    def optimize_transfers_for_budget_grid(self, test_covariate_dataset, budgets):
        """
        Computes transfers for each budget in the list of budgets. Enables calling
        run_opt for each budget in the list.
        """

        if self.t_to_household_estimator_map is None:
            raise ValueError("Need to run fit before a policy can be computed")
        
        X_test, r_test = test_covariate_dataset.get_data()

        # for each t, order the households in the test set
        t_to_ordered_households_map = dict()
        t_to_estimated_benefits_map = dict()

        for t, household_benefit_estimator in self.t_to_household_estimator_map.items():

            estimated_benefits = household_benefit_estimator(X_test)
            t_to_estimated_benefits_map[t] = estimated_benefits

            sorting_indices = np.argsort(estimated_benefits)[::-1]
            t_to_ordered_households_map[t] = sorting_indices

        # For each budget, select optimal transfers on the test set
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
                    r_test, t, ordered_households, budget
                )
                total_estimated_benefits = estimated_benefits[
                    indices_to_receive_transfers
                ].sum()

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

        return t_to_estimated_benefits_map, t_to_ordered_households_map

    def run_opt(self, test_covariate_dataset):

        if self.budget_to_households_map is None:
            raise ValueError(
                "Must run optimize_transfers_for_budget_grid before run_opt for "
                "binary gap targeting"
            )

        assert self.budget is not None

        if self.budget not in self.budget_to_households_map.keys():
            raise ValueError(
                f"budget {self.budget} was not included in list provided to "
                "optimize_transfers_for_budget_grid."
            )

        indices_to_receive_transfers = self.budget_to_households_map[self.budget]
        self.t = self.budget_to_t_map[self.budget]

        def transfer_function(X):

            assignments = {i: [(0, 1.0)] for i in range(len(X))}
            for i in indices_to_receive_transfers:
                assignments[i] = [(self.t, 1.0)]
            return assignments

        self.opt_policy = transfer_function
        return transfer_function

    def _get_indices_to_receive_transfers(self, r, t, sorting_indices, budget):

        r = r / r.sum()

        number_of_transfers = len(sorting_indices)
        budget_remaining = budget

        for count, sorting_index in enumerate(sorting_indices):

            cost = r[sorting_index] * t
            if budget_remaining < cost:
                number_of_transfers = count
                break
            budget_remaining -= cost

        return sorting_indices[:number_of_transfers]


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

        super().__init__(
            c_bar=c_bar,
            budget=budget
        )
        self.name = "oracle_gap"
        self.scheme = scheme

    def run_opt(self, y_test, r_test=None):

        dataset = Dataset(X=None, y=y_test, r=r_test)

        if self.scheme == "lift_to_line":

            self.opt_policy = run_oracle_poverty_gap_lift_to_line_scheme(
                dataset, tolerance=self.unconditional_tolerance, c_bar=self.c_bar
            )

        else:
            self.opt_policy = run_oracle_poverty_gap_floor_scheme(
                dataset, tolerance=self.unconditional_tolerance, c_bar=self.c_bar
            )

        return self.opt_policy


class BinaryRateTargetedTransfers(TargetedTransfers):

    def __init__(
        self, c_bar=2.15, budget=None,
    ):

        super().__init__(
            c_bar=c_bar,
            budget=budget
        )
        self.name = "binary_rate"

    def fit(
        self,
        train_dataset,
        n_bins=100,
        n_knots=4,
        degree=4,
        truncation_upper_value=10,
        n_epochs=300
    ):
        density_estimator = get_cond_density_estimator(
            train_dataset,
            n_bins=n_bins,
            n_knots=n_knots,
            degree=degree,
            truncation_upper_value=truncation_upper_value,
            n_epochs=n_epochs,
        )
        self.density_estimator = density_estimator

    def run_opt(self, test_covariate_dataset, n_T=100):
        """
        Run the optimization algorithm.

        :param X_test: The input features of the test data.
        :type X_test: numpy.ndarray
        :param r_test: The sampling weight variable of the test data. Defaults to None.
        :type r_test: numpy.ndarray or None
        """
        if self.density_estimator is None:
            assert False, "Need to first set predictor"
        if self.unconditional_tolerance is None:
            assert False, "Need to first set tolerance"

        Ts = np.linspace(0.0, 2.15, n_T)
        feasible_Ts = []
        policies = []
        costs = []
        X_test, r_test = test_covariate_dataset.get_data()
        cond_dists = self.density_estimator(X_test)

        for T in Ts:
            res = compute_opt_policy_knapsack(
                test_covariate_dataset,
                cond_dists=cond_dists,
                raw_min_transfer_function=None,
                tolerance=self.unconditional_tolerance,
                transfer_amts=np.array([0.0, T]),
                c_bar=self.c_bar,
                compute_cond_density=self.density_estimator,
                deterministic=True,
            )
            if res != False:
                feasible_Ts.append(T)
                policies.append(res[0])
                costs.append(res[1])

        idx = np.argmin(costs)
        opt_binary_policy = policies[idx]
        self.opt_policy = opt_binary_policy
        return opt_binary_policy


class OraclePovertyRateTargetedTransfers(TargetedTransfers):
    def __init__(self, c_bar=2.15, unconditional_tolerance=None):

        super().__init__(
            c_bar=c_bar
        )
        self.name = "oracle_rate"

    def run_opt(self, test_outcome_dataset):

        oracle_policy = run_oracle_poverty_rate(
            test_outcome_dataset, c_bar=self.c_bar, tolerance=self.unconditional_tolerance
        )
        self.opt_policy = oracle_policy
        return oracle_policy
