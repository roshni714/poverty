# def _get_threshold_to_receive_transfers(
#     self,
#     validation_dataset,
#     estimated_benefits,
#     household_idx_ranked_by_benefit,
#     t,
#     budget,
# ):
#     _, _, r_val = validation_dataset.get_data()

#     pop_weight_receive_transfers = budget / t
#     weights_ranked_by_benefit = r_val[household_idx_ranked_by_benefit]
#     cumsum_weights = np.cumsum(weights_ranked_by_benefit)
#     indicator_receive_transfers = cumsum_weights < pop_weight_receive_transfers
#     idx_receive_transfers = household_idx_ranked_by_benefit[
#         indicator_receive_transfers
#     ]
#     if len(idx_receive_transfers) == 0:
#         threshold = np.inf
#     else:
#         threshold = estimated_benefits[idx_receive_transfers[-1]]
#     return threshold

# def _get_indices_to_receive_transfers_threshold(
#     self, validation_dataset, t, threshold
# ):
#     if self.t_to_household_estimator_map is None:
#         raise ValueError("Need to run fit before a policy can be computed")

#     X_val, _, _ = validation_dataset.get_data()

#     regressor = self.t_to_household_estimator_map[t]
#     estimated_benefits = regressor(X_val)
#     idx_receive_transfers = np.where(estimated_benefits > threshold)[0]
#     return idx_receive_transfers


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
        self.name = "hybrid_rate"
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
        self.name = "conditional_{}_rate".format(method)
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
            max_transfer_val = max(
                [assignments[key][0][0] for key in assignments.keys()]
            )
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


class OraclePovertyGapPolicy:
    def __init__(self, budget, c_bar=2.15):
        self.c_bar = c_bar
        self.budget = budget
        self.opt_policy = None
        self.name = "oracle_poverty_gap"
        self.nclass = None

    def run_opt(X_test, y_test, r_test=None):
        dataset = Dataset(X=None, y=y_test, r=r_test)

        oracle_policy = run_oracle_poverty_rate(
            dataset, c_bar=self.c_bar, tolerance=self.unconditional_tolerance
        )
        self.opt_policy = oracle_policy
        return oracle_policy

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

        dataset = Dataset(X_test, y_test, r_test)
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
        self.name = "conditional_{}_rate".format(method)
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


def check_knapsack_feasibility(
    dataset,
    cond_densities,
    unconditional_tolerance,
    raw_min_transfer_function,
    c_bar,
    max_transfer_value,
):

    if raw_min_transfer_function is not None:
        raw_min_transfer_values = raw_min_transfer_function(cond_densities)
        if any(raw_min_transfer_values > max_transfer_value):
            return False
    probs = np.array(
        [
            cond_density.cdf(c_bar - max_transfer_value)
            for cond_density in cond_densities
        ]
    )
    prob_total = np.sum(probs * dataset.r).item()
    if prob_total > unconditional_tolerance:
        return False
    else:
        return True


def compute_cost(train_dataset, policy):
    assignments = policy(train_dataset.X)

    total_cost = 0.0
    for i in range(len(train_dataset)):
        cost = 0.0
        for j in range(len(assignments[i])):
            cost += assignments[i][j][1] * assignments[i][j][0]
        total_cost += cost * train_dataset.r[i]

    return total_cost


def get_transfer_function(
    transfer_amts, c_bar, eta, lamb, compute_cond_density, deterministic=False
):
    """
    Compute the transfer function.

    :param c_bar: The poverty line.
    :type c_bar: float
    :param eta: The threshold cost-benefit ratio.
    :type eta: float
    :param lamb: The threshold probability.
    :type lamb: float
    :param compute_cond_density: A function to compute the conditional density.
    :type compute_cond_density: Callable[[np.ndarray], np.ndarray]
    :return: The transfer function.
    :rtype: Callable[[np.ndarray], np.ndarray]
    """

    def t(X_test):
        cond_densities = compute_cond_density(X_test)
        assignments = {x_idx: [] for x_idx in range(len(X_test))}

        for j, cond_density in enumerate(cond_densities):
            cvx_hull = cond_density.get_convex_hull(z=transfer_amts, c_bar=c_bar)
            ratios = np.zeros(len(cvx_hull)).astype(np.float64)
            ratios[0] = -np.inf
            for i in range(len(cvx_hull) - 1):
                p1 = cvx_hull[i]
                p2 = cvx_hull[i + 1]
                ratios[i + 1] = (p2[1] - p1[1]) / (p2[0] - p1[0])
            idx = bisect.bisect_left(ratios, eta)

            if (
                idx > 0
                and idx < len(ratios)
                and ratios[idx - 1] < eta
                and ratios[idx] > eta
            ):
                assignments[j] = [(cvx_hull[idx - 1][1], 1.0)]
            elif idx < len(ratios) and ratios[idx] == eta:
                if deterministic:
                    assignments[j] = [
                        (max(cvx_hull[idx - 1][1], cvx_hull[idx][1]), 1.0)
                    ]
                else:
                    assignments[j] = [
                        (cvx_hull[idx - 1][1], lamb),
                        (cvx_hull[idx][1], 1 - lamb),
                    ]
            else:
                assignments[j] = [(0.0, 1.0)]

        return assignments

    return t


def get_alpha_transfer_function(alpha, c_bar, lamb, eta, cond_density_estimator):
    """
    Compute the transfer function.

    :param alpha: The alpha value.
    :type alpha: float
    :param c_bar: The poverty line.
    :type c_bar: float
    :param lamb: The threshold cost-benefit ratio.
    :type lamb: float
    :param eta: The threshold probability
    :type eta: float
    :param cond_density_estimator: A function to compute the conditional density.
    :type cond_density_estimator: Callable[[np.ndarray], np.ndarray]
    :return: The transfer function.
    :rtype: Callable[[np.ndarray], np.ndarray]
    """

    def t(X_test):
        cond_densities = cond_density_estimator(X_test)
        assignments = {x_idx: [] for x_idx in range(len(X_test))}
        cvx_hulls = get_alpha_convex_hulls(
            alpha,
            c_bar,
            cond_dists=cond_densities,
        )

        for j, cond_density in enumerate(cond_densities):
            cvx_hull = cvx_hulls[j]
            ratios = np.zeros(len(cvx_hull)).astype(np.float64)
            ratios[0] = -np.inf
            for i in range(len(cvx_hull) - 1):
                p1 = cvx_hull[i]
                p2 = cvx_hull[i + 1]
                ratios[i + 1] = (p2[1] - p1[1]) / (p2[0] - p1[0])
            idx = bisect.bisect_left(ratios, eta)

            if (
                idx > 0
                and idx < len(ratios)
                and ratios[idx - 1] < lamb
                and ratios[idx] > lamb
            ):
                assignments[j] = [(cvx_hull[idx - 1][1], 1.0)]
            elif idx < len(ratios) and ratios[idx] == lamb:
                assignments[j] = [
                    (cvx_hull[idx - 1][1], eta),
                    (cvx_hull[idx][1], 1 - eta),
                ]
            else:
                assignments[j] = [(0.0, 1.0)]
        return assignments

    return t


def check_assignments_are_equal(assignment1, assignment2):
    assert assignment1.keys() == assignment2.keys()

    for key in assignment1.keys():
        val1 = assignment1[key]
        val2 = assignment2[key]
        assert val1 == val2, "error at key {} bc {} != {}".format(key, val1, val2)


from opt_targeted_transfers.dataset_utils import Dataset
from opt_targeted_transfers.prediction import get_prediction_function
from opt_targeted_transfers.density_estimation import get_cond_density_estimator
from opt_targeted_transfers.knapsack import (
    compute_alpha_opt_policies,
    compute_opt_policy_knapsack,
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
                "policy_type": self.name,
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
            oracle = False

        all_transfers_ev = expected_value_transfers(
            dataset, self.opt_policy, oracle=oracle
        )

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
        self.name = "conditional_{}_rate".format(method)
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
        self.name = "unconditional_rate"
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


class GapTargetedTransfers(TargetedTransfers):
    """
    Poverty-gap targeting.
    """

    def __init__(self, c_bar=2.15):
        """
        :param c_bar: The minimum threshold value (poverty line). Defaults to 2.15.
        :type c_bar: float
        """

        super().__init__(
            c_bar=c_bar, conditional_tolerance=None, unconditional_tolerance=None
        )
        self.name = "gap"
        self.quantile_regressors = None

    def _fit_quantile_regressors(
        self,
        train_dataset,
        low_dim=False,
        n_epochs=300,
        n_quantiles=20,
        hidden_layer_size=64,
    ):
        """
        :param train_dataset: The dataset used for training the regressors
        :type train_dataset: Dataset
        :param low_dim: Whether to use a single-layered nn for regression
        :type low_dim: bool
        :param n_epochs: The number of epochs for training the regressors
        :type n_epochs: int
        :param n_quantiles: The number of (evenly spaced) quantiles to fit
        :type n_quantils: int
        :param hidden_layer_size: size of the hidden layer in the neural net
        :type hidden_layer_size: int
        :return: The quantile regressors.
        :rtype: Dict[int, Callable[[np.ndarray], np.ndarray]]
        """

        quantiles = np.linspace(0, 1, n_quantiles, endpoint=True)

        quantile_regressors = dict()

        for quantile in quantiles:
            quantile_regressors[quantile] = get_quantile_regressor(
                train_dataset,
                quantile,
                low_dim=low_dim,
                n_epochs=n_epochs,
                hidden_layer_size=hidden_layer_size,
            )

        return quantile_regressors

    def set_conditional_tolerance(self, conditional_tolerance):
        raise NotImplementedError("Gap targeting can't use conditional tolerances.")

    def set_unconditional_tolerance(self, unconditional_tolerance):
        raise NotImplementedError("Not yet")

    def fit(
        self,
        X_train,
        y_train,
        r_train=None,
        low_dim=False,
        n_epochs=300,
        n_quantiles=20,
        hidden_layer_size=64,
    ):
        """
        Fitting the quantile regression.

        :param X_train: The input features of the training data.
        :type X_train: numpy.ndarray
        :param y_train: The target values of the training data.
        :type y_train: numpy.ndarray
        :param r_train: The sampling weight variable of the training data. Defaults to None.
        :type r_train: numpy.ndarray or None
        :param n_epochs: The number of epochs to train the model. Defaults to 300.
        :type n_epochs: int
        :param n_quantiles: The number of quantiles to fit.
        :type n_quantiles: int
        :param hidden_layer_size: The size of the hidden layer in the quantile-fitting neural net
        :type hidden_layer_size: int
        """

        self.train_dataset = Dataset(X_train, y_train, r_train)

        self.quantile_regressors = self._fit_quantile_regressors(
            self.train_dataset,
            low_dim=low_dim,
            n_epochs=n_epochs,
            n_quantiles=n_quantiles,
            hidden_layer_size=hidden_layer_size,
        )

        # TODO: Remove. For now, needed for directly checking quantile fits.
        return (
            self.quantile_regressors.keys(),
            self.quantile_regressors,
            self.train_dataset,
        )

    def run_opt(self, X_test, lambda_):

        t = self._get_policy_for_lambda(X_test, lambda_)
        self.opt_policy = t


class BinaryGapTargetedTransfers(TargetedTransfers):
    # TODO: Hit expected budget exactly by having one stochastic transfer

    def __init__(self, c_bar=2.15, num_t_values=20):

        super().__init__(
            c_bar=c_bar,
            unconditional_tolerance=None,
            conditional_tolerance=None,
        )

        self.name = "binary_gap"
        self.candidate_t_values = np.linspace(
            0, self.c_bar, num_t_values, endpoint=True
        )

        self.t_to_household_estimator_map = None
        self.t = None

    def fit(
        self,
        X_train,
        y_train,
        r_train=None,
        low_dim=False,
        n_epochs=300,
        hidden_layer_size=64,
    ):

        dataset = Dataset(X_train, y_train, r_train)

        # For each transfer size t, fit benefit estimator using training data
        self.t_to_household_estimator_map = dict()

        for t in self.candidate_t_values:

            self.t_to_household_estimator_map[t] = (
                self._fit_household_benefit_estimator(
                    dataset.X,
                    dataset.y,
                    dataset.r,
                    t,
                    low_dim=low_dim,
                    n_epochs=n_epochs,
                    hidden_layer_size=hidden_layer_size,
                )
            )

    def optimize_transfers_for_budget_grid(self, X_test, r_test=None, budgets=None):
        """
        Computes transfers for each budget in the list of budgets. Enables calling
        run_opt for each budget in the list.
        """

        if self.t_to_household_estimator_map is None:
            raise ValueError("Need to run fit before a policy can be computed")

        if r_test is None:
            r_test = np.ones(len(X_test))

        r_test = r_test / np.sum(r_test)

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

    def run_opt(self, X_test=None, r_test=None, budget=None):

        if self.budget_to_households_map is None:
            raise ValueError(
                "Must run optimize_transfers_for_budget_grid before run_opt for "
                "binary gap targeting"
            )

        assert budget is not None

        if budget not in self.budget_to_households_map.keys():
            raise ValueError(
                f"budget {budget} was not included in list provided to "
                "optimize_transfers_for_budget_grid."
            )

        indices_to_receive_transfers = self.budget_to_households_map[budget]
        self.t = self.budget_to_t_map[budget]

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

    def __init__(self, c_bar=2.15, unconditional_tolerance=None, scheme="lift_to_line"):

        assert scheme in ("lift_to_line", "floor")

        super().__init__(
            c_bar=c_bar,
            unconditional_tolerance=unconditional_tolerance,
            conditional_tolerance=None,
        )
        self.name = "oracle_gap"
        self.scheme = scheme

    def set_conditional_tolerance(self, conditional_tolerance):
        raise NotImplementedError(
            "OraclePovertyGapTargetedTransfer can't use conditional tolerances."
        )

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


class BinaryTargetedTransfers(TargetedTransfers):

    def __init__(
        self, c_bar=2.15, unconditional_tolerance=None, conditional_tolerance=None
    ):

        super().__init__(
            c_bar=c_bar,
            unconditional_tolerance=unconditional_tolerance,
            conditional_tolerance=conditional_tolerance,
        )
        self.name = "binary_rate"

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


class OraclePovertyRateTargetedTransfers(TargetedTransfers):
    def __init__(self, c_bar=2.15, unconditional_tolerance=None):

        super().__init__(
            c_bar=c_bar,
            unconditional_tolerance=unconditional_tolerance,
            conditional_tolerance=None,
        )
        self.name = "oracle_rate"

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


# def compute_opt_policy_knapsack(
#     test_covariate_dataset,
#     cond_dists,
#     budget,
#     transfer_amts,
#     c_bar,
# ):
#     X_test, r_test = test_covariate_dataset.get_data()

#     cvx_hulls = [
#         cond_dists[i].get_convex_hull(np.array(transfer_amts), c_bar) for i in range(len(cond_dists))
#     ]

#     (opt_assignment, policy_cost, poverty_rate, lamb, eta) = (
#             solve_fractional_mc_knapsack_problem(r_test, cvx_hulls, budget)
#         )

#     return opt_assignment, poverty_rate


def get_transfer_function(c_bar, eta, lamb):
    def t(y_test):
        assignments = {x_idx: [] for x_idx in range(len(y_test))}
        transfers = np.maximum(c_bar - y_test, 0)
        below_line = y_test < c_bar

        for j in range(len(y_test)):
            cvx_hull = get_lower_cvx_hull([(0.0, transfers[j]), (below_line[j], 0.0)])
            ratios = np.zeros(len(cvx_hull)).astype(np.float64)
            ratios[0] = -np.inf
            for i in range(len(cvx_hull) - 1):
                p1 = cvx_hull[i]
                p2 = cvx_hull[i + 1]
                ratios[i + 1] = (p2[1] - p1[1]) / (p2[0] - p1[0])
            idx = bisect.bisect_left(ratios, eta)

            if (
                idx > 0
                and idx < len(ratios)
                and ratios[idx - 1] < eta
                and ratios[idx] > eta
            ):
                assignments[j] = [(cvx_hull[idx - 1][1], 1.0)]
            elif idx < len(ratios) and ratios[idx] == eta:
                if lamb != 0:
                    assignments[j] = [
                        (cvx_hull[idx - 1][1], lamb),
                        (cvx_hull[idx][1], 1 - lamb),
                    ]
                else:
                    assignments[j] = [(cvx_hull[idx][1], 1 - lamb)]
            else:
                assignments[j] = [(0.0, 1.0)]
        return assignments

    return t


# Get the assignments that have cost lower than budget
# idx = bisect_left(costs, self.budget)
# left_cost = costs[idx - 1]

# if left_cost == self.budget:
#     self.assignments = all_assignments[idx - 1]
#     return all_assignments[idx - 1]
# elif self.budget > left_cost and idx <= len(costs) - 1:
#     right_cost = costs[idx]
#     assert self.budget > left_cost and self.budget <= right_cost
#     left_assignments = all_assignments[idx - 1]
#     right_assignments = all_assignments[idx]

#     actual_assignments = {x_idx: [] for x_idx in range(len(X_test))}
#     for i in range(len(X_test)):
#         left_transfer = left_assignments[i][0][0]
#         right_transfer = right_assignments[i][0][0]
#         slope = (right_transfer - left_transfer) / (right_cost - left_cost)
#         intercept = left_transfer - slope * left_cost
#         actual_transfer = slope * self.budget + intercept
#         actual_assignments[i].append((actual_transfer, 1.0))
#     self.assignments = actual_assignments
#     return actual_assignments
# elif self.budget < costs[0]:
#     actual_assignments = {x_idx: [] for x_idx in range(len(X_test))}
#     for i in range(len(X_test)):
#         left_transfer = 0.0
#         right_transfer = all_assignments[0][i][0][0]
#         slope = (right_transfer - left_transfer) / (costs[0])
#         actual_transfer = slope * self.budget
#         actual_assignments[i].append((actual_transfer, 1.0))
#     self.assignments = actual_assignments
#     return actual_assignments
# elif self.budget > costs[-1]:
#     actual_assignments = {x_idx: [] for x_idx in range(len(X_test))}
#     for i in range(len(X_test)):
#         left_transfer = all_assignments[-1][i][0][0]
#         right_transfer = self.c_bar
#         slope = (right_transfer - left_transfer) / (self.c_bar - costs[-1])
#         intercept = left_transfer - slope * costs[-1]
#         actual_transfer = slope * self.budget + intercept
#         actual_assignments[i].append((actual_transfer, 1.0))
#     self.assignments = actual_assignments
#     return actual_assignments


def get_prediction_function(dataset, n_classes, class_thresholds, n_epochs=300):
    """
    Get prediction function for a given dataset.

    :param dataset: The dataset used for training the regressor.
    :type dataset: Dataset
    :param n_epochs: The number of epochs for training the regressor. Defaults to 300.
    :type n_epochs: int
    :return: The quantile regressor.
    :rtype: Callable[[np.ndarray], np.ndarray]
    """
    X = dataset.X
    y = dataset.y
    r = dataset.r

    X, X_mean, X_std = standardize(X)

    np.random.seed(123456)
    torch.manual_seed(123456)

    if X.shape[1] > 0:
        d = X.shape[1]
        h_hat = torch.nn.Sequential(
            torch.nn.Linear(d, 5), torch.nn.ReLU(), torch.nn.Linear(5, n_classes)
        )

        loss_f = torch.nn.CrossEntropyLoss(reduction="none")

        def cross_entropy_loss(h_hat, idx):
            sub_n = len(idx)
            pred = h_hat(torch.Tensor(X[idx, :]))
            return loss_f(pred, torch.Tensor(y[idx]).long())

        optimizer = torch.optim.Adam(h_hat.parameters(), lr=1e-2)
        train_prop = 0.7
        idx_train_set, idx_val_set = list(range(int(train_prop * len(X)))), list(
            range(int(train_prop * len(X)), len(X))
        )

        batch_size = int(len(idx_train_set) / 3)
        print("Fitting predictor via classification with cross entropy loss...")
        pbar = tqdm.tqdm(list(range(n_epochs)))
        val_losses = []
        models = []

        for epoch in pbar:
            if epoch % 25 == 0:
                val_loss = torch.sum(
                    cross_entropy_loss(h_hat, idx_val_set)
                    * torch.Tensor(r[idx_val_set])
                )
                val_losses.append(val_loss.detach().item())
                models.append(copy.deepcopy(h_hat))

            idx = np.random.choice(idx_train_set, size=batch_size)
            optimizer.zero_grad()
            loss = torch.sum(cross_entropy_loss(h_hat, idx) * torch.Tensor(r[idx]))
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"loss": loss.item()})
        best_model_idx = np.argmin(val_losses)
        final_h_hat = models[best_model_idx]

    def prediction_function(X_test):
        if X_test.shape[1] == 0:
            pdf_matrix = torch.zeros(len(X_test), n_classes)
            pdf_matrix[:, 0] = 1.0
        else:
            X_test = (X_test - X_mean) / X_std
            pdf_matrix = (
                torch.nn.functional.softmax(
                    final_h_hat(torch.Tensor(X_test)), dim=1
                ).reshape(X_test.shape[0], n_classes)
            ).detach()
        cdf_matrix = torch.cumsum(pdf_matrix, dim=1)
        zeros = torch.zeros((len(X_test), 1))
        cdf_matrix = torch.cat((zeros, cdf_matrix), dim=1)[:, :-1]
        idx_maxima = argrelextrema(pdf_matrix.numpy(), np.less_equal, axis=1)
        idx_minima = argrelextrema(pdf_matrix.numpy(), np.greater_equal, axis=1)

        best_idx = torch.argmax(pdf_matrix, axis=1)
        modes = class_thresholds[best_idx]
        cond_dists = []

        for i in range(len(X_test)):
            idx_extrema = np.sort(
                np.hstack(
                    (
                        idx_maxima[1][idx_maxima[0] == i],
                        idx_minima[1][idx_minima[0] == i],
                    )
                )
            )
            cdf_function = interp1d(
                class_thresholds,
                cdf_matrix[i].flatten(),
                bounds_error=False,
                kind="previous",
                fill_value=(0.0, 1.0),
            )
            pdf_function = interp1d(
                class_thresholds,
                pdf_matrix[i].flatten(),
                bounds_error=False,
                kind="previous",
                fill_value=0.0,
            )
            ppf_function = interp1d(
                cdf_matrix[i].flatten(),
                class_thresholds,
                bounds_error=False,
                kind="previous",
                fill_value=(class_thresholds[0], class_thresholds[-1]),
            )

            cond_dists.append(
                NonparametricConditionalDistribution(
                    pdf_function,
                    cdf_function,
                    ppf_function,
                    extrema=class_thresholds[idx_extrema].flatten(),
                    outcome_range=(class_thresholds[0], class_thresholds[-1]),
                    mode=modes[i].item(),
                )
            )

        return cond_dists

    return prediction_function
