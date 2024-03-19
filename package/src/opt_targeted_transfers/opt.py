from opt_targeted_transfers.dataset_utils import Dataset
from opt_targeted_transfers.density_estimation import get_cond_density_estimator
from opt_targeted_transfers.knapsack import compute_alpha_opt_policies
from opt_targeted_transfers.quantile_regression import get_quantile_regressor
from opt_targeted_transfers.evaluate import post_transfer_metrics

import dill as pickle
import numpy as np


class ConditionalTargetedTransfers:
    """
    Compute optimal conditional targeted transfers.
    """

    def __init__(self, method="qr", name="malawi_test", c_bar=2.15, tolerance=None):
        """
        Initialize a new instance of the UnconditionalTargetedTransfers class.
        :param method: The method used for fitting the nuisance parameter. Either "qr" or "density."
        :type method: str
        :param name: The name of the transfer policy. Defaults to "malawi_test".
        :type name: str
        :param c_bar: The minimum threshold value (poverty line). Defaults to 2.15.
        :type c_bar: float
        :param tolerance: The tolerance. Defaults to None.
        :type tolerance: float or None
        """

        self.name = name
        self.method = method
        self.opt_policy = None
        self.c_bar = c_bar
        self.tolerance = tolerance
        self.quantile_regressor = None
        self.density_estimator = None

    def fit(
        self,
        X_train,
        y_train,
        r_train=None,
        log_transform=True,
        knot_quantiles=None,
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

        if self.tolerance is None and self.method == "qr":
            assert False, "First set tolerance before fitting if method is {}".format(
                self.method
            )
        dataset = Dataset(X_train, y_train, r_train)

        if self.method == "density":
            density_estimator = get_cond_density_estimator(
                dataset,
                log_transform=log_transform,
                knot_quantiles=knot_quantiles,
                n_epochs=n_epochs,
            )

            pickle.dump(
                density_estimator,
                open("{}_cond_density_estimator.pickle".format(self.name), "wb"),
            )
            self.density_estimator = density_estimator
        elif self.method == "qr":
            quantile_regressor = get_quantile_regressor(
                dataset, self.tolerance, n_epochs=n_epochs
            )
            self.quantile_regressor = quantile_regressor

    def set_density_estimator(self, cond_density):
        """
        Set the conditional density estimator for the model.

        :param cond_density: The conditional density estimator that maps numpy array
                             of X values with shape (N, D) to numpy array of ConditionalDistribution
                             objects.
        :type cond_density: Callable[[np.ndarray], np.ndarray]
        """

        self.density_estimator = cond_density

    def set_tolerance(self, tolerance):
        """
        Set the tolerance.
        Note that setting the tolerance to a new value will clear the
        existing optimal policy. Furthermore, if the method is "qr,"
        then setting a new tolerance will also clear the quantile
        regressor.

        :param tolerance: The tolerance to set.
        :type tolerance: float
        """
        if tolerance != self.tolerance:
            self.opt_policy = None
            if self.method == "qr":
                self.quantile_regressor = None

        self.tolerance = tolerance

    def run_opt(self, X_test, r_test=None):
        """
        Run the optimization algorithm.

        :param X_test: The input features of the test data.
        :type X_test: numpy.ndarray
        :param r_test: The sampling weight variable of the test data. Defaults to None.
        :type r_test: numpy.ndarray or None
        """

        if self.method == "qr":
            if self.tolerance is None:
                assert False, "Need to specify tolerance"
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
            if self.tolerance is None:
                assert False, "Need to specify tolerance"
            if self.density_estimator is None:
                assert False, "Need to fit density function"

            def t(X_test):
                cond_densities = self.density_estimator(X_test)
                assignments = {x_idx: [] for x_idx in range(len(X_test))}
                for i, cond_dist in enumerate(cond_densities):
                    if cond_dist.cdf(self.c_bar) > self.tolerance:
                        assignments[i] = [
                            (self.c_bar - cond_dist.ppf(self.tolerance), 1.0)
                        ]
                    else:
                        assignments[i] = [(0.0, 1.0)]
                return assignments

        self.opt_policy = t
        return t

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

        dataset = Dataset(X_test, y_test, r_test)
        result = post_transfer_metrics(dataset, self.opt_policy, self.c_bar)
        if len(X_test.shape) > 1:
            d = X_test.shape[1]
        else:
            d = 0
        result.update(
            {
                "method": "conditional_{}".format(self.method),
                "tolerance": self.tolerance,
                "d": d,
            }
        )
        return result


class UnconditionalTargetedTransfers:
    """
    Computes the optimal unconditional targeted transfer policy.
    """

    def __init__(self, name="malawi_test", c_bar=2.15, tolerance=None):
        """
        Initialize a new instance of the UnconditionalTargetedTransfers class.
        :param name: The name of the transfer policy. Defaults to "malawi_test".
        :type name: str
        :param c_bar: The minimum threshold value (poverty line). Defaults to 2.15.
        :type c_bar: float
        :param tolerance: The tolerance. Defaults to None.
        :type tolerance: float or None
        """
        self.name = name
        self.density_estimator = None
        self.opt_policy = None
        self.c_bar = c_bar
        self.tolerance = tolerance

    def fit(
        self,
        X_train,
        y_train,
        r_train=None,
        log_transform=True,
        knot_quantiles=None,
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
            knot_quantiles=knot_quantiles,
            n_epochs=n_epochs,
        )

        pickle.dump(
            density_estimator,
            open("{}_cond_density_estimator.pickle".format(self.name), "wb"),
        )

        self.density_estimator = density_estimator

    def set_density_estimator(self, cond_density):
        """
        Set the conditional density estimator for the model.

        :param cond_density: The conditional density estimator that maps numpy array
                             of X values with shape (N, D) to numpy array of ConditionalDistribution
                             objects.
        :type cond_density: Callable[[np.ndarray], np.ndarray]
        """
        self.density_estimator = cond_density

    def set_tolerance(self, tolerance):
        """
        Set the tolerance.
        Note that setting the tolerance to a new value will clear the
        existing optimal policy.

        :param tolerance: The tolerance to set.
        :type tolerance: float
        """
        if tolerance != self.tolerance:
            self.opt_policy = None
        self.tolerance = tolerance

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
        if self.tolerance is None:
            assert False, "Need to first set tolerance"
        dataset = Dataset(X_test, y=None, r=r_test)

        (
            t_alpha_joint_programs,
            total_transfers,
            alphas,
        ) = compute_alpha_opt_policies(
            dataset,
            self.density_estimator,
            tolerance=self.tolerance,
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

        dataset = Dataset(X_test, y=y_test, r=r_test)
        result = post_transfer_metrics(dataset, self.opt_policy, self.c_bar)
        if len(X_test.shape) > 1:
            d = X_test.shape[1]
        else:
            d = 0
        result.update({"method": "unconditional", "tolerance": self.tolerance, "d": d})
        return result
