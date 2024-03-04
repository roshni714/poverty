from dataset import Dataset
from density_estimation import get_cond_density_estimator
from knapsack import compute_alpha_opt_policies
from quantile_regression import get_quantile_regressor

import dill as pickle


class ConditionalTargetedTransfers:

    def __init__(self, method="qr", name="malawi_test", c_bar=2.15,  budget=0.1):
        self.name = name
        self.method = method
        self.opt_policy = None
        self.c_bar = c_bar
        self.budget = budget

 
    def fit(self, X_train, y_train, r_train=None, log_transform=True, df=None):
        dataset = Dataset(X_train, y_train, r_train)
       
        if self.method == "density": 
            density_estimator = get_cond_density_estimator(dataset, log_transform=log_transform, df=df)

            pickle.dump(
            cond_density_estimator,
            open("{}_cond_density_estimator.pickle".format(self.name), "wb"),
            )
            self.density_estimator = density_estimator
        else:
            quantile_regressor = get_quantile_regressor(dataset, self.budget)

    def set_density_estimator(self, cond_density):
        self.density_estimator = cond_density

        



class OptTargetedTransfers:

    def __init__(self, name="malawi_test", c_bar=2.15, budget=0.1):
        self.name = name
        self.density_estimator = None
        self.opt_policy = None
        self.c_bar = c_bar
        self.budget = budget

    def fit(self, X_train, y_train, r_train=None, log_transform=True, df=None):
        dataset = Dataset(X_train, y_train, r_train)
        
        density_estimator = get_cond_density_estimator(dataset, log_transform=log_transform, df=df)

        pickle.dump(
        cond_density_estimator,
        open("{}_cond_density_estimator.pickle".format(self.name), "wb"),
        )

        self.density_estimator = density_estimator

    def set_density_estimator(self, cond_density):
        self.density_estimator = cond_density

    def run_opt(self, X_test, y_test, r_test=None, alpha_min=None, alpha_max=None, n_alpha=200):
        if self.density_estimator is None:
            assert False, "Need to first set density estimator"
        dataset = Dataset(X_test, y_test, r_test)

        (t_alpha_joint_programs,
        total_transfers,
        alphas,
        ) = compute_alpha_opt_policies(
        dataset,
        self.density_estimator,
        budget=self.budget,
        c_bar=self.c_bar,
        alpha_min = alpha_min,
        alpha_max = alpha_max,
        n_alpha=n_alpha,
        title="{}_joint_opt".format(self.name),
        )

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
