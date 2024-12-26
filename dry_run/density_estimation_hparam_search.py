from opt_targeted_transfers import get_cond_density_estimator, Dataset, get_nll
from feature_selection import forward_selection
import pandas as pd


def get_optimal_density_estimation_parameters(
    density_estimation_hparam_ranges, data_generator, ntrain, nval
):
    """
    Get optimal density estimation hyperparameters.

    Args:
        density_estimation_hparam_ranges (dict): Dictionary containing the hyperparameter ranges for density estimation.
        data_generator (DataGenerator): Data generator object.

    Returns:
    """

    if "n_features" in density_estimation_hparam_ranges:
        n_features_range = density_estimation_hparam_ranges["n_features"]
    else:
        n_features_range = [10]

    if "degree" in density_estimation_hparam_ranges:
        degree_range = density_estimation_hparam_ranges["degree"]
    else:
        degree_range = [3]

    if "n_bins" in density_estimation_hparam_ranges:
        n_bins_range = density_estimation_hparam_ranges["n_bins"]
    else:
        n_bins_range = [10]

    if "n_knots" in density_estimation_hparam_ranges:
        n_knots_range = density_estimation_hparam_ranges["n_knots"]
    else:
        n_knots_range = [4]

    train_df = data_generator(nsamples=ntrain, seed=547396234)
    val_df = data_generator(nsamples=nval, seed=79809342)
    train_dataset = Dataset(
        train_df, outcome="consumption_per_capita_per_day", weight="hh_wgt", covs=[]
    )
    val_dataset = Dataset(
        val_df, outcome="consumption_per_capita_per_day", weight="hh_wgt", covs=[]
    )

    print("Running forward selection...")
    ordered_features, _ = forward_selection(
        train_dataset, val_dataset, max_features=max(n_features_range)
    )

    results = []
    for i, n_features in enumerate(n_features_range):
        train_df = data_generator(nsamples=ntrain, seed=1283 * i + 1)
        val_df = data_generator(nsamples=nval, seed=4308 * i + 1)
        train_dataset = Dataset(
            train_df,
            outcome="consumption_per_capita_per_day",
            weight="hh_wgt",
            covs=ordered_features[:n_features],
        )
        val_dataset = Dataset(
            val_df,
            outcome="consumption_per_capita_per_day",
            weight="hh_wgt",
            covs=ordered_features[:n_features],
        )
        for degree in degree_range:
            for n_bins in n_bins_range:
                for n_knots in n_knots_range:
                    density_estimator = get_cond_density_estimator(
                        train_dataset,
                        n_features=n_features,
                        degree=degree,
                        n_bins=n_bins,
                        n_knots=n_knots,
                    )
                    nll = get_nll(val_dataset, density_estimator)
                    results.append(
                        {
                            "n_features": n_features,
                            "degree": degree,
                            "n_bins": n_bins,
                            "n_knots": n_knots,
                            "nll": nll,
                        }
                    )

    df = pd.DataFrame(results)
    optimal_params = df.loc[df["nll"].idxmin()].to_dict()
    del optimal_params["nll"]
    return optimal_params


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
