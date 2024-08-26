from opt_targeted_transfers.dataset_utils import Dataset
from opt_targeted_transfers.prediction import get_prediction_function
from opt_targeted_transfers.density_estimation import get_cond_density_estimator
from opt_targeted_transfers.knapsack import (
    compute_alpha_opt_policies,
    compute_opt_policy_knapsack,
)
from opt_targeted_transfers.oracle import run_oracle_poverty_rate #run_oracle_poverty_gap
from opt_targeted_transfers.quantile_regression import get_quantile_regressor
from opt_targeted_transfers.evaluate import (
    post_transfer_metrics,
    expected_value_transfers,
)
from opt_targeted_transfers.reporting import write_result

import dill as pickle
import numpy as np
from bisect import bisect_left


class TargetedTransfers:
    """
    Base class for TargetedTransfers
    """

    def __init__(
        self, c_bar=2.15, unconditional_tolerance=None, conditional_tolerance=None
    ):
        self.c_bar = c_bar
        self.unconditional_tolerance = unconditional_tolerance
        self.conditional_tolerance = conditional_tolerance
        self.density_estimator = None
        self.opt_policy = None
        self.name = None
        self.nclass = None

    def fit(X_train, y_train, r_train=None):
        pass

    def run_opt(X_test, y_test, r_test=None):
        pass

    def set_unconditional_tolerance(self, unconditional_tolerance):
        """
        Set the unconditional tolerance.
        Note that setting the tolerance to a new value will clear the
        existing optimal policy.

        :param unconditional_tolerance: The unconditional tolerance to set.
        :type unconditional_tolerance: float
        """
        if unconditional_tolerance != self.unconditional_tolerance:
            self.opt_policy = None
        self.unconditional_tolerance = unconditional_tolerance

    def set_conditional_tolerance(self, conditional_tolerance):
        """
        Set the conditional tolerance.
        Note that setting the tolerance to a new value will clear the
        existing optimal policy.

        :param conditional_tolerance: The conditional tolerance to set.
        :type conditional_tolerance: float
        """
        if conditional_tolerance != self.conditional_tolerance:
            self.opt_policy = None
        self.conditional_tolerance = conditional_tolerance

    def set_density_estimator(self, cond_density):
        """
        Set the conditional density estimator for the model.

        :param cond_density: The conditional density estimator that maps numpy array
                             of X values with shape (N, D) to numpy array of ConditionalDistribution
                             objects.
        :type cond_density: Callable[[np.ndarray], np.ndarray]
        """

        self.density_estimator = cond_density

    def save_opt_policy(self, name):
        if self.opt_policy is None:
            assert False, "Need to run opt first"
        pickle.dump(
            self.opt_policy,
            open("{}.pickle".format(name), "wb"),
        )

    def evaluate(self, X_test, y_test, r_test=None):
        """
        Evaluate optimal policy.

        :param X_test: The input features of the test data.
        :type X_test: numpy.ndarray
        :param y_test: The target values of the test data.
        :type y_test: numpy.ndarray
        :param r_test: The response variable of the test data. Defaults to None.
        :type r_test: numpy.ndarray or None
        :return: A dictionary of evaluation results.
        :rtype: dict
        """

        if self.opt_policy is None:
            assert False, "Need to first run optimization"

        dataset = Dataset(X_test, y_test, r_test, normalize_weight_sum=False)

        if "oracle" in self.name:
            result = post_transfer_metrics(
                dataset, self.opt_policy, self.c_bar, oracle=True
            )
        else:
            result = post_transfer_metrics(dataset, self.opt_policy, self.c_bar)

        if len(X_test.shape) > 1:
            d = X_test.shape[1]
        else:
            d = 0
        result.update(
            {
                "method": self.name,
                "unconditional_tolerance": self.unconditional_tolerance,
                "conditional_tolerance": self.conditional_tolerance,
                "d": d,
                "nclass": self.nclass,
            }
        )
        return result

    def evaluate_equity(self, X_test, y_test, r_test=None, path=None):
        """
        Evaluate equity of optimal policy.

        :param X_test: The input features of the test data.
        :type X_test: numpy.ndarray
        :param y_test: The target values of the test data.
        :type y_test: numpy.ndarray
        :param r_test: The response variable of the test data. Defaults to None.
        :type r_test: numpy.ndarray or None
        :return: A dictionary of evaluation results.
        :rtype: dict
        """
        if self.opt_policy is None:
            assert False, "Need to first run optimization"
        dataset = Dataset(X_test, y=y_test, r=r_test)
        if len(X_test.shape) > 1:
            d = X_test.shape[1]
        else:
            d = 0

        if "oracle" in self.name:
            oracle = True
        else:
            oracle=False

        all_transfers_ev = expected_value_transfers(dataset, self.opt_policy, oracle=oracle)

        for i in range(len(all_transfers_ev)):
            write_result(
                path, {"consumption": y_test[i], "ev_transfer": all_transfers_ev[i]}
            )


class ConditionalTargetedTransfers(TargetedTransfers):
    """
    Compute optimal conditional targeted transfers.
    """

    def __init__(self, method="qr", c_bar=2.15, conditional_tolerance=None):
        """
        Initialize a new instance of the UnconditionalTargetedTransfers class.
        :param method: The method used for fitting the nuisance parameter. Either "qr" or "density."
        :type method: str
        :type name: str
        :param c_bar: The minimum threshold value (poverty line). Defaults to 2.15.
        :type c_bar: float
        :param tolerance: The tolerance. Defaults to None.
        :type tolerance: float or None
        """

        super().__init__(
            c_bar=c_bar,
            conditional_tolerance=conditional_tolerance,
            unconditional_tolerance=conditional_tolerance,
        )
        self.name = "conditional_{}".format(method)
        self.method = method
        self.quantile_regressor = None
        self.density_estimator = None

    def fit(
        self,
        X_train,
        y_train,
        r_train=None,
        low_dim=False,
        log_transform=True,
        internal_knots=None,
        n_epochs=300,
    ):
        """
        Fitting the nuisance parameter.

        :param X_train: The input features of the training data.
        :type X_train: numpy.ndarray
        :param y_train: The target values of the training data.
        :type y_train: numpy.ndarray
        :param r_train: The sampling weight variable of the training data. Defaults to None.
        :type r_train: numpy.ndarray or None
        :param log_transform: Whether to perform a log-transform on Y before fitting for "density" method.
                          Defaults to True.
        :type log_transform: bool
        :param knot_quantiles: The quantiles to use as knots for the spline basis functions for "density" method.
                           If None, evenly spaced knots will be used.
                           Defaults to None.
        :type knot_quantiles: numpy.ndarray or None
        :param n_epochs: The number of epochs to train the model. Defaults to 300.
        :type n_epochs: int
        """

        if self.conditional_tolerance is None and self.method == "qr":
            assert (
                False
            ), "First set conditional tolerance before fitting if method is {}".format(
                self.method
            )
        dataset = Dataset(X_train, y_train, r_train)

        if self.method == "density":
            density_estimator = get_cond_density_estimator(
                dataset,
                low_dim=low_dim,
                log_transform=log_transform,
                internal_knots=internal_knots,
                n_epochs=n_epochs,
            )

            pickle.dump(
                density_estimator,
                open("{}_cond_density_estimator.pickle".format(self.name), "wb"),
            )
            self.density_estimator = density_estimator
        elif self.method == "qr":
            quantile_regressor = get_quantile_regressor(
                dataset, self.conditional_tolerance, low_dim=low_dim, n_epochs=n_epochs
            )
            self.quantile_regressor = quantile_regressor

    def set_conditional_tolerance(self, conditional_tolerance):
        """
        Set the tolerance.
        Note that setting the tolerance to a new value will clear the
        existing optimal policy. Furthermore, if the method is "qr,"
        then setting a new tolerance will also clear the quantile
        regressor.

        :param tolerance: The tolerance to set.
        :type tolerance: float
        """
        if conditional_tolerance != self.conditional_tolerance:
            self.opt_policy = None
            if self.method == "qr":
                self.quantile_regressor = None

        self.conditional_tolerance = conditional_tolerance
        self.unconditional_tolerance = conditional_tolerance

    def run_opt(self, X_test, r_test=None):
        """
        Run the optimization algorithm.

        :param X_test: The input features of the test data.
        :type X_test: numpy.ndarray
        :param r_test: The sampling weight variable of the test data. Defaults to None.
        :type r_test: numpy.ndarray or None
        """

        if self.method == "qr":
            if self.quantile_regressor is None:
                assert False, "Need to fit quantile regressor"

            def t(X_test):
                quantile = self.quantile_regressor(X_test)
                transfer = np.maximum(self.c_bar - quantile, 0)
                assignments = {x_idx: [] for x_idx in range(len(X_test))}
                for i in range(len(X_test)):
                    assignments[i].append((transfer[i].item(), 1.0))
                return assignments

        elif self.method == "density":
            if self.density_estimator is None:
                assert False, "Need to fit density function"

            def t(X_test):
                cond_densities = self.density_estimator(X_test)
                assignments = {x_idx: [] for x_idx in range(len(X_test))}
                for i, cond_dist in enumerate(cond_densities):
                    if cond_dist.cdf(self.c_bar) > self.conditional_tolerance:
                        assignments[i] = [
                            (
                                self.c_bar - cond_dist.ppf(self.conditional_tolerance),
                                1.0,
                            )
                        ]
                    else:
                        assignments[i] = [(0.0, 1.0)]
                return assignments

        self.opt_policy = t
        return t


class UnconditionalTargetedTransfers(TargetedTransfers):
    """
    Computes the optimal unconditional targeted transfer policy.
    """

    def __init__(self, c_bar=2.15, unconditional_tolerance=None):
        """
        Initialize a new instance of the UnconditionalTargetedTransfers class.
        :param c_bar: The minimum threshold value (poverty line). Defaults to 2.15.
        :type c_bar: float
        :param tolerance: The tolerance. Defaults to None.
        :type tolerance: float or None
        """
        super().__init__(
            c_bar=c_bar,
            unconditional_tolerance=unconditional_tolerance,
            conditional_tolerance=None,
        )
        self.name = "unconditional"
        self.density_estimator = None
        self.opt_policy = None

    def fit(
        self,
        X_train,
        y_train,
        r_train=None,
        low_dim=False,
        log_transform=True,
        internal_knots=None,
        n_epochs=300,
    ):
        """
        Fitting the conditional density.

        :param X_train: The input features of the training data.
        :type X_train: numpy.ndarray
        :param y_train: The target values of the training data.
        :type y_train: numpy.ndarray
        :param r_train: The sampling weight variable of the training data. Defaults to None.
        :type r_train: numpy.ndarray or None
        :param log_transform: Whether to perform a log-transform on Y before fitting.
                          Defaults to True.
        :type log_transform: bool
        :param knot_quantiles: The quantiles to use as knots for the spline basis functions.
                           If None, evenly spaced knots will be used.
                           Defaults to None.
        :type knot_quantiles: numpy.ndarray or None
        :param n_epochs: The number of epochs to train the model. Defaults to 300.
        :type n_epochs: int
        """
        dataset = Dataset(X_train, y_train, r_train)

        density_estimator = get_cond_density_estimator(
            dataset,
            log_transform=log_transform,
            internal_knots=internal_knots,
            n_epochs=n_epochs,
        )

        pickle.dump(
            density_estimator,
            open("{}_cond_density_estimator.pickle".format(self.name), "wb"),
        )

        self.density_estimator = density_estimator

    def run_opt(
        self,
        X_test,
        r_test=None,
        min_alpha=None,
        max_alpha=None,
        n_alpha=200,
        path=None,
    ):
        """
        Run the optimization algorithm.

        :param X_test: The input features of the test data.
        :type X_test: numpy.ndarray
        :param r_test: The sampling weight variable of the test data. Defaults to None.
        :type r_test: numpy.ndarray or None
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
        dataset = Dataset(X_test, y=None, r=r_test)

        (
            t_alpha_joint_programs,
            total_transfers,
            alphas,
        ) = compute_alpha_opt_policies(
            dataset,
            self.density_estimator,
            tolerance=self.unconditional_tolerance,
            c_bar=self.c_bar,
            min_alpha=min_alpha,
            max_alpha=max_alpha,
            n_alpha=n_alpha,
            min_transfer_function=None,
            path=path,
        )

        idx = np.argmin(total_transfers)
        t_joint_program_est = t_alpha_joint_programs[idx]
        self.opt_policy = t_joint_program_est
        return t_joint_program_est


class HybridTargetedTransfers(TargetedTransfers):
    """
    Computes the optimal unconditional targeted transfer policy.
    """

    def __init__(
        self, c_bar=2.15, unconditional_tolerance=None, conditional_tolerance=None
    ):
        """
        Initialize a new instance of the UnconditionalTargetedTransfers class.
        :param c_bar: The minimum threshold value (poverty line). Defaults to 2.15.
        :type c_bar: float
        :param unconditional_tolerance: The unconditional tolerance. Defaults to None.
        :type unconditional_tolerance: float or None
        :param conditional_tolerance: The conditional tolerance. Defaults to None.
        :type conditional_tolerance: float or None
        """
        super().__init__(
            c_bar=c_bar,
            unconditional_tolerance=unconditional_tolerance,
            conditional_tolerance=conditional_tolerance,
        )
        self.name = "hybrid"
        self.density_estimator = None
        self.opt_policy = None

    def fit(
        self,
        X_train,
        y_train,
        r_train=None,
        low_dim=False,
        log_transform=True,
        internal_knots=None,
        n_epochs=300,
    ):
        """
        Fitting the conditional density.

        :param X_train: The input features of the training data.
        :type X_train: numpy.ndarray
        :param y_train: The target values of the training data.
        :type y_train: numpy.ndarray
        :param r_train: The sampling weight variable of the training data. Defaults to None.
        :type r_train: numpy.ndarray or None
        :param log_transform: Whether to perform a log-transform on Y before fitting.
                          Defaults to True.
        :type log_transform: bool
        :param knot_quantiles: The quantiles to use as knots for the spline basis functions.
                           If None, evenly spaced knots will be used.
                           Defaults to None.
        :type knot_quantiles: numpy.ndarray or None
        :param n_epochs: The number of epochs to train the model. Defaults to 300.
        :type n_epochs: int
        """
        dataset = Dataset(X_train, y_train, r_train)

        density_estimator = get_cond_density_estimator(
            dataset,
            low_dim=low_dim,
            log_transform=log_transform,
            internal_knots=internal_knots,
            n_epochs=n_epochs,
        )

        pickle.dump(
            density_estimator,
            open("{}_cond_density_estimator.pickle".format(self.name), "wb"),
        )

        self.density_estimator = density_estimator

    def run_opt(
        self,
        X_test,
        r_test=None,
        min_alpha=None,
        max_alpha=None,
        n_alpha=200,
        path=None,
    ):
        """
        Run the optimization algorithm.

        :param X_test: The input features of the test data.
        :type X_test: numpy.ndarray
        :param r_test: The sampling weight variable of the test data. Defaults to None.
        :type r_test: numpy.ndarray or None
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
        if self.unconditional_tolerance is None:
            assert False, "Need to first set unconditional tolerance"
        if self.conditional_tolerance is not None:
            assert (
                self.conditional_tolerance >= self.unconditional_tolerance
            ), "Conditional tolerance must be greater than or equal to uncondiitonal tolerance."
        dataset = Dataset(X_test, y=None, r=r_test)

        if self.conditional_tolerance is not None:

            def min_transfer_function(cond_densities):
                min_transfer_values = [
                    np.maximum(
                        self.c_bar - cond_dist.ppf(self.conditional_tolerance), 0
                    ).item()
                    for cond_dist in cond_densities
                ]
                return min_transfer_values

        else:
            min_transfer_function = None

        (
            t_alpha_joint_programs,
            total_transfers,
            alphas,
        ) = compute_alpha_opt_policies(
            dataset,
            self.density_estimator,
            tolerance=self.unconditional_tolerance,
            c_bar=self.c_bar,
            min_alpha=min_alpha,
            max_alpha=max_alpha,
            n_alpha=n_alpha,
            min_transfer_function=min_transfer_function,
            path=path,
        )

        idx = np.argmin(total_transfers)
        t_joint_program_est = t_alpha_joint_programs[idx]
        self.opt_policy = t_joint_program_est
        return t_joint_program_est


class GapTargetedTransfers(TargetedTransfers):
    """
    Poverty-gap targeting

    For now, sweeps out a cost-gap curve.
    """

    def __init__(self, c_bar=2.15):
        """
        Initialize a new instance of the UnconditionalTargetedTransfers class.
        :param method: The method used for fitting the nuisance parameter. Either "qr" or "density."
        :type method: str
        :type name: str
        :param c_bar: The minimum threshold value (poverty line). Defaults to 2.15.
        :type c_bar: float
        """

        super().__init__(
            c_bar=c_bar,
            conditional_tolerance=None,
            unconditional_tolerance=None,
        )
        self.name = "gap"
        self.quantile_regressors = None

    def fit(
        self,
        X_train,
        y_train,
        r_train=None,
        low_dim=False,
        n_epochs=300,
        n_quantiles=20
    ):
        """
        Fitting the quantile regression.

        :param X_train: The input features of the training data.
        :type X_train: numpy.ndarray
        :param y_train: The target values of the training data.
        :type y_train: numpy.ndarray
        :param r_train: The sampling weight variable of the training data. Defaults to None.
        :type r_train: numpy.ndarray or None
        :type log_transform: bool
        :param n_epochs: The number of epochs to train the model. Defaults to 300.
        :type n_epochs: int
        """

        self.quantiles = np.linspace(0, 1, n_quantiles, endpoint=False)

        dataset = Dataset(X_train, y_train, r_train)

        self.quantile_regressors = dict()

        for quantile in self.quantiles:
            self.quantile_regressors[quantile] = get_quantile_regressor(
                dataset, quantile, low_dim=low_dim, n_epochs=n_epochs
            )

    def run_opt(self, X_test, lambda_):
        """
        Run the optimization algorithm.

        :param X_test: The input features of the test data.
        :type X_test: numpy.ndarray
        """

        if self.quantile_regressors is None:
            assert False, "Need to fit quantile regressors"

        quantile_index = bisect_left(self.quantiles, lambda_)

        # if quantile_index == len(self.quantiles), then lambda_ is > all evaluated quantiles.
        # if quantile_index == 0 then lambda_ is <= all evaluated quantiles.
        # In that case for now I don't attempt to fake interpolation.
        if (
            (lambda_ == self.quantiles[quantile_index])
            or (quantile_index == len(self.quantiles))
            or (quantile_index == 0)
        ):
            baseline_lambda_quantile_wealth_level = (
                self.quantile_regressors[self.quantiles[quantile_index]](X_test)
            )

        else:
            quantile_index_low = quantile_index - 1
            quantile_index_high = quantile_index 

            if not (
                (lambda_ > self.quantiles[quantile_index_low])
                & (lambda_ < self.quantiles[quantile_index_high])
            ):
                from IPython import embed
                embed()
            assert lambda_ > self.quantiles[quantile_index_low]
            assert lambda_ < self.quantiles[quantile_index_high]
            
            interpolation_factor = (
                (lambda_ - self.quantiles[quantile_index_low]) 
                / (self.quantiles[quantile_index_high] - self.quantiles[quantile_index_low])
            )

            baseline_lambda_quantile_wealth_level_low = (
                self.quantile_regressors[self.quantiles[quantile_index_low]](X_test)
            )
            baseline_lambda_quantile_wealth_level_high = (
                self.quantile_regressors[self.quantiles[quantile_index_high]](X_test)
            )

            baseline_lambda_quantile_wealth_level = (
                (1 - interpolation_factor) * baseline_lambda_quantile_wealth_level_low
                + interpolation_factor * baseline_lambda_quantile_wealth_level_high
            )

        transfer = np.maximum(self.c_bar - baseline_lambda_quantile_wealth_level, 0)

        def t(X_test):
            assignments = {x_idx: [] for x_idx in range(len(X_test))}
            for i in range(len(X_test)):
                assignments[i].append((transfer[i].item(), 1.0))
            return assignments

        self.opt_policy = t
        return t


# class UnconditionalDiscreteTransfers(TargetedTransfers):

#     def __init__(
#         self, method="lindsey", nclass=None, c_bar=2.15, unconditional_tolerance=None
#     ):
#         super().__init__(
#             c_bar=c_bar,
#             unconditional_tolerance=unconditional_tolerance,
#             conditional_tolerance=None,
#         )
#         self.nclass = nclass
#         if nclass is not None:
#             self.class_thresholds = np.linspace(0.0, self.c_bar, self.nclass)
#         self.method = method
#         self.name = "unconditional_discrete_{}".format(method)

#     def fit(
#         self,
#         X_train,
#         y_train,
#         r_train=None,
#         log_transform=True,
#         knot_quantiles=None,
#         n_epochs=300,
#     ):

#         if self.method == "nn":
#             if self.nclass is None:
#                 assert False, "Method is nn and nclass not set"
#             y_trainclasses = np.searchsorted(self.class_thresholds, y_train) - 1.0
#             dataset = Dataset(X=X_train, y=y_trainclasses, r=r_train)
#             density_estimator = get_prediction_function(
#                 dataset, self.nclass, self.class_thresholds, n_epochs=n_epochs
#             )
#         elif self.method == "lindsey":
#             dataset = Dataset(X_train, y_train, r_train)

#             density_estimator = get_cond_density_estimator(
#                 dataset,
#                 log_transform=log_transform,
#                 knot_quantiles=knot_quantiles,
#                 n_epochs=n_epochs,
#             )
#         self.density_estimator = density_estimator

#     def set_nclass(self, nclass):
#         """
#         Set number of classes.
#         Note that setting the number of classes to a new value will clear the
#         existing optimal policy.

#         """
#         if nclass != self.nclass:
#             self.opt_policy = None
#             if self.method == "nn":
#                 self.density_estimator = None

#         self.nclass = nclass
#         self.class_thresholds = np.linspace(0.0, self.c_bar, self.nclass)

#     def run_opt(self, X_test, r_test=None):
#         """
#         Run the optimization algorithm.

#         :param X_test: The input features of the test data.
#         :type X_test: numpy.ndarray
#         :param r_test: The sampling weight variable of the test data. Defaults to None.
#         :type r_test: numpy.ndarray or None
#         """
#         if self.density_estimator is None:
#             assert False, "Need to first set predictor"
#         if self.unconditional_tolerance is None:
#             assert False, "Need to first set tolerance"
#         if self.nclass is None:
#             assert False, "Need to first set nclass"
#         dataset = Dataset(X_test, y=None, r=r_test)

#         t_opt = compute_opt_policy_knapsack(
#             dataset,
#             self.density_estimator,
#             tolerance=self.unconditional_tolerance,
#             transfer_amts=self.c_bar - self.class_thresholds,
#             c_bar=self.c_bar,
#         )
#         self.opt_policy = t_opt
#         return t_opt



class BinaryTargetedTransfers(TargetedTransfers):

    def __init__(
        self, c_bar=2.15, unconditional_tolerance=None, conditional_tolerance=None
    ):

        super().__init__(
            c_bar=c_bar,
            unconditional_tolerance=unconditional_tolerance,
            conditional_tolerance=conditional_tolerance,
        )
        self.name = "binary"

    def fit(
        self,
        X_train,
        y_train,
        r_train=None,
        low_dim=False,
        log_transform=True,
        internal_knots=None,
        n_epochs=300,
    ):
        dataset = Dataset(X_train, y_train, r_train)

        density_estimator = get_cond_density_estimator(
            dataset,
            low_dim=low_dim,
            log_transform=log_transform,
            internal_knots=internal_knots,
            n_epochs=n_epochs,
        )
        self.density_estimator = density_estimator

    def run_opt(self, X_test, r_test=None, n_T=100):
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
        dataset = Dataset(X_test, y=None, r=r_test)

        if self.conditional_tolerance is not None:

            def raw_min_transfer_function(cond_densities):
                raw_min_transfer_values = [
                    np.maximum(
                        self.c_bar - cond_dist.ppf(self.conditional_tolerance), 0
                    ).item()
                    for cond_dist in cond_densities
                ]
                return raw_min_transfer_values

        else:
            raw_min_transfer_function = None

        Ts = np.linspace(0.50, 2.15, n_T)
        feasible_Ts = []
        policies = []
        costs = []
        cond_dists = self.density_estimator(dataset.X)

        for T in Ts:
            res = compute_opt_policy_knapsack(
                dataset,
                cond_dists=cond_dists,
                raw_min_transfer_function=raw_min_transfer_function,
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


class BinaryConditionalTargetedTransfers(TargetedTransfers):
    """
    Compute optimal conditional targeted transfers.
    """

    def __init__(self, method="qr", c_bar=2.15, conditional_tolerance=None):
        """
        Initialize a new instance of the UnconditionalTargetedTransfers class.
        :param method: The method used for fitting the nuisance parameter. Either "qr" or "density."
        :type method: str
        :type name: str
        :param c_bar: The minimum threshold value (poverty line). Defaults to 2.15.
        :type c_bar: float
        :param tolerance: The tolerance. Defaults to None.
        :type tolerance: float or None
        """

        super().__init__(
            c_bar=c_bar,
            conditional_tolerance=conditional_tolerance,
            unconditional_tolerance=conditional_tolerance,
        )
        self.name = "conditional_{}".format(method)
        self.method = method
        self.quantile_regressor = None
        self.density_estimator = None

    def fit(
        self,
        X_train,
        y_train,
        r_train=None,
        low_dim=False,
        log_transform=True,
        internal_knots=None,
        n_epochs=300,
    ):
        """
        Fitting the nuisance parameter.

        :param X_train: The input features of the training data.
        :type X_train: numpy.ndarray
        :param y_train: The target values of the training data.
        :type y_train: numpy.ndarray
        :param r_train: The sampling weight variable of the training data. Defaults to None.
        :type r_train: numpy.ndarray or None
        :param log_transform: Whether to perform a log-transform on Y before fitting for "density" method.
                          Defaults to True.
        :type log_transform: bool
        :param knot_quantiles: The quantiles to use as knots for the spline basis functions for "density" method.
                           If None, evenly spaced knots will be used.
                           Defaults to None.
        :type knot_quantiles: numpy.ndarray or None
        :param n_epochs: The number of epochs to train the model. Defaults to 300.
        :type n_epochs: int
        """

        if self.conditional_tolerance is None and self.method == "qr":
            assert (
                False
            ), "First set conditional tolerance before fitting if method is {}".format(
                self.method
            )
        dataset = Dataset(X_train, y_train, r_train)

        if self.method == "density":
            density_estimator = get_cond_density_estimator(
                dataset,
                low_dim=low_dim,
                log_transform=log_transform,
                internal_knots=internal_knots,
                n_epochs=n_epochs,
            )

            pickle.dump(
                density_estimator,
                open("{}_cond_density_estimator.pickle".format(self.name), "wb"),
            )
            self.density_estimator = density_estimator
        elif self.method == "qr":
            quantile_regressor = get_quantile_regressor(
                dataset, self.conditional_tolerance, low_dim=low_dim, n_epochs=n_epochs
            )
            self.quantile_regressor = quantile_regressor

    def set_conditional_tolerance(self, conditional_tolerance):
        """
        Set the tolerance.
        Note that setting the tolerance to a new value will clear the
        existing optimal policy. Furthermore, if the method is "qr,"
        then setting a new tolerance will also clear the quantile
        regressor.

        :param tolerance: The tolerance to set.
        :type tolerance: float
        """
        if conditional_tolerance != self.conditional_tolerance:
            self.opt_policy = None
            if self.method == "qr":
                self.quantile_regressor = None

        self.conditional_tolerance = conditional_tolerance
        self.unconditional_tolerance = conditional_tolerance

    def run_opt(self, X_test, r_test=None):
        """
        Run the optimization algorithm.

        :param X_test: The input features of the test data.
        :type X_test: numpy.ndarray
        :param r_test: The sampling weight variable of the test data. Defaults to None.
        :type r_test: numpy.ndarray or None
        """

        if self.method == "qr":
            if self.quantile_regressor is None:
                assert False, "Need to fit quantile regressor"

            def t_unconstrained(X_test):
                quantile = self.quantile_regressor(X_test)
                transfer = np.maximum(self.c_bar - quantile, 0)
                assignments = {x_idx: [] for x_idx in range(len(X_test))}
                for i in range(len(X_test)):
                    assignments[i].append((transfer[i].item(), 1.0))
                return assignments

        elif self.method == "density":
            if self.density_estimator is None:
                assert False, "Need to fit density function"

            def t_unconstrained(X_test):
                cond_densities = self.density_estimator(X_test)
                assignments = {x_idx: [] for x_idx in range(len(X_test))}
                for i, cond_dist in enumerate(cond_densities):
                    if cond_dist.cdf(self.c_bar) > self.conditional_tolerance:
                        assignments[i] = [
                            (
                                self.c_bar - cond_dist.ppf(self.conditional_tolerance),
                                1.0,
                            )
                        ]
                    else:
                        assignments[i] = [(0.0, 1.0)]
                return assignments

        
        def binary_policy(X_test):
            assignments = t_unconstrained(X_test)
            max_transfer_val = max([assignments[key][0][0]  for key in assignments.keys()])
            binary_assignments = {x_idx: [] for x_idx in range(len(X_test))}
            
            for key in assignments.keys():
                l = assignments[key]
                transfer_amt = l[0][0]
                if transfer_amt > 0:
                    binary_assignments[key] = [(max_transfer_val, 1.0)]
                else:
                    binary_assignments[key] = [(0.0, 1.0)]
            return binary_assignments
        
        self.opt_policy = binary_policy
        return binary_policy


class OraclePovertyRateTargetedTransfers(TargetedTransfers):
    def __init__(self, c_bar=2.15, unconditional_tolerance=None):

        super().__init__(
            c_bar=c_bar,
            unconditional_tolerance=unconditional_tolerance,
            conditional_tolerance=None,
        )
        self.name = "oracle_poverty_rate"

    def set_conditional_tolerance(self, conditional_tolerance):
        raise NotImplementedError(
            "OraclePovertyRateTargetedTransfer can't handle conditional tolerances."
        )

    def run_opt(self, y_test, r_test=None):
        dataset = Dataset(X=None, y=y_test, r=r_test)

        oracle_policy = run_oracle_poverty_rate(
            dataset, c_bar=self.c_bar, tolerance=self.unconditional_tolerance
        )
        self.opt_policy = oracle_policy
        return oracle_policy

