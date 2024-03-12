from opt_targeted_transfers.dataset_utils import Dataset
from opt_targeted_transfers.density_estimation import get_cond_density_estimator
from opt_targeted_transfers.knapsack import compute_alpha_opt_policies
from opt_targeted_transfers.quantile_regression import get_quantile_regressor
from opt_targeted_transfers.evaluate import post_transfer_metrics

import dill as pickle
import numpy as np

class ConditionalTargetedTransfers:

    def __init__(self, method="qr", name="malawi_test", c_bar=2.15,  budget=0.1):
        self.name = name
        self.method = method
        self.opt_policy = None
        self.c_bar = c_bar
        self.budget = budget
        self.quantile_regressor = None
        self.density_estimator = None
 
    def fit(self, X_train, y_train, r_train=None, log_transform=True, df=None):
        dataset = Dataset(X_train, y_train, r_train)
       
        if self.method == "density": 
            density_estimator = get_cond_density_estimator(dataset, log_transform=log_transform, df=df)

            pickle.dump(
            cond_density_estimator,
            open("{}_cond_density_estimator.pickle".format(self.name), "wb"),
            )
            self.density_estimator = density_estimator
        elif self.method == "qr":
            quantile_regressor = get_quantile_regressor(dataset, self.budget)
            self.quantile_regressor = quantile_regressor

    def set_density_estimator(self, cond_density):
        self.density_estimator = cond_density

    def run_opt(self, X_test, y_test, r_test=None):
        if method == "qr":
            if self.quantile_regressor is None:
                assert False, "Need to fit quantile regressor first"

            def t(X_test):
                quantile = self.quantile_regressor(X_test)
                transfer = np.maximum(c_bar - quantile, 0)
                assignments = {x_idx: [] for x_idx in range(len(X_test))}
                for i in range(len(X_test)):
                    assignments[i].append((transfer[i], 1.0))
                return assignments

        elif method == "density":
            if self.density_estimator is None:
                assert False, "Need to fit density function first"

            def t(X_test):
                cond_densities = self.density_estimator(X_test)
                assignments = {x_idx: [] for x_idx in range(len(X_test))}
                for i, cond_dist in enumerate(cond_densities):
                    if cond_dist.cdf(c_bar) > budget:
                        assignments[i] = [(c_bar - cond_dist.ppf(budget), 1.0)]
                    else:
                        assignments[i] = [(0.0, 1.0)]
                return assignments
        self.opt_policy = t
        return t

    def evaluate(self, X_test, y_test, r_test=None):
        if self.opt_policy is None:
            assert False, "Need to first run optimization"

        dataset = Dataset(X_test, y_test, r_test)
        result = post_transfer_metrics(dataset, self.opt_policy, self.c_bar)
        return result

 
class OptTargetedTransfers:

    def __init__(self, name="malawi_test", c_bar=2.15, budget=None):
        self.name = name
        self.density_estimator = None
        self.opt_policy = None
        self.c_bar = c_bar
        self.budget = budget

    def fit(self, X_train, y_train, r_train=None, log_transform=True, knot_quantiles=None, n_epochs=300):
        dataset = Dataset(X_train, y_train, r_train)
        
        density_estimator = get_cond_density_estimator(dataset, log_transform=log_transform, knot_quantiles=knot_quantiles, n_epochs=n_epochs)

        pickle.dump(
        density_estimator,
        open("{}_cond_density_estimator.pickle".format(self.name), "wb"),
        )

        self.density_estimator = density_estimator

    def set_density_estimator(self, cond_density):
        self.density_estimator = cond_density

    def set_budget(self, budget):
        if budget != self.budget:
            self.opt_policy= None
        self.budget = budget

    def run_opt(self, X_test, y_test, r_test=None, min_alpha=None, max_alpha=None, n_alpha=200, path=None):
        if self.density_estimator is None:
            assert False, "Need to first set density estimator"
        if self.budget is None:
            assert False, "Need to first set budget"
        dataset = Dataset(X_test, y_test, r_test)

        (t_alpha_joint_programs,
        total_transfers,
        alphas,
        ) = compute_alpha_opt_policies(
        dataset,
        self.density_estimator,
        budget=self.budget,
        c_bar=self.c_bar,
        min_alpha = min_alpha,
        max_alpha = max_alpha,
        n_alpha=n_alpha, 
        path=path)

        idx = np.argmin(total_transfers)
        t_joint_program_est = t_alpha_joint_programs[idx]
        self.opt_policy = t_joint_program_est 
        return t_joint_program_est

    def evaluate(self, X_test, y_test, r_test=None):
        if self.opt_policy is None:
            assert False, "Need to first run optimization"

        dataset = Dataset(X_test, y_test, r_test)
        result = post_transfer_metrics(dataset, self.opt_policy, self.c_bar)
        return result 
