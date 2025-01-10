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
