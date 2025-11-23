import unittest
import numpy as np
from opt_targeted_transfers import get_quantile_regressor, GapTargetedTransfers
from opt_targeted_transfers import Dataset, split
from opt_targeted_transfers import post_transfer_metrics
from scipy.stats import norm
import pandas as pd


class TestOptTargetedTransfers(unittest.TestCase):

    def _make_1d_data(self, n=20000, seed=0):
        np.random.seed(seed)
        X = np.random.uniform(1, 5, (n, 1))
        y = 2 * X[:, 0] + np.random.normal(0, 0.1, n)
        r = np.ones(n)

        df = pd.DataFrame(
            np.hstack((X, y.reshape(-1, 1), r.reshape(-1, 1))),
            columns=["feature", "outcome", "weight"],
        )

        dataset = Dataset(df, outcome="outcome", covs=["feature"], weight="weight")
        return dataset

    def _make_2d_data(self, n=20000, seed=0):
        np.random.seed(seed)
        X1 = np.random.uniform(1, 5, (n, 1))
        X2 = np.random.uniform(1, 5, (n, 1))
        y = 2 * X1[:, 0] + 3 * X2[:, 0] + np.random.normal(0, 0.1, n)
        r = np.ones(n)

        df = pd.DataFrame(
            np.hstack((X1, X2, y.reshape(-1, 1), r.reshape(-1, 1))),
            columns=["feature1", "feature2", "outcome", "weight"],
        )

        dataset = Dataset(
            df, outcome="outcome", covs=["feature1", "feature2"], weight="weight"
        )
        return dataset

    def test_get_quantile_regressor_1d_data(self):
        # Add your test implementation here
        n = 20000
        dataset = self._make_1d_data(n=n, seed=42)
        train_dataset, val_dataset = split(dataset, frac=0.6, seed=42)

        quantiles = np.linspace(0.05, 0.95, 5)

        for i in range(len(quantiles)):
            quantile_regressor = get_quantile_regressor(
                train_dataset,
                val_dataset,
                quantiles[i],
                winsorize_outcome=97,
                n_layers=2,
                n_hidden_units=128,
                lr=1e-3,
                n_epochs=100,
                seed=42,
                device="cpu",
            )
            X_test = np.linspace(1, 5, 20).reshape(-1, 1)
            quantile_regressor_output = quantile_regressor(X_test)
            true_res = 2 * np.linspace(1, 5, 20) + norm.ppf(quantiles[i], loc=0, scale=0.1)
            np.testing.assert_allclose(quantile_regressor_output, true_res, rtol=1e-1)

    def test_get_quantile_regressor_2d_data(self):
        # Add your test implementation here
        n = 20000
        dataset = self._make_2d_data(n=n, seed=42)
        train_dataset, val_dataset = split(dataset, frac=0.6, seed=42)

        quantiles = np.linspace(0.05, 0.95, 5)

        for i in range(len(quantiles)):
            quantile_regressor = get_quantile_regressor(
                train_dataset,
                val_dataset,
                quantiles[i],
                winsorize_outcome=97,
                n_layers=2,
                n_hidden_units=128,
                lr=1e-3,
                n_epochs=100,
                seed=42,
                device="cpu",
            )
            X1_test = np.linspace(1, 5, 5)
            X2_test = np.linspace(1, 5, 5)
            X_test = np.array([[x1, x2] for x1 in X1_test for x2 in X2_test])
            quantile_regressor_output = quantile_regressor(X_test)
            true_res = 2 * X_test[:, 0] + 3 * X_test[:, 1] + norm.ppf(quantiles[i], loc=0, scale=0.1)
            np.testing.assert_allclose(quantile_regressor_output, true_res, rtol=1e-1)

    def test_gap_targeting(self):
        # Add your test implementation here

        n = 20000
        dataset = self._make_2d_data(n=n, seed=42)
        train_dataset, val_dataset = split(dataset, frac=0.6, seed=42)
        test_dataset = self._make_2d_data(n=5000, seed=24)
        test_covariate_dataset = Dataset(
            test_dataset.df,
            outcome=None,
            covs=test_dataset.covs,
            weight=test_dataset.weight,
        )

        c_bar = 3

        gtt = GapTargetedTransfers(c_bar=c_bar, n_regressors=10)
        gtt.fit(train_dataset, val_dataset)
        budgets = np.linspace(0, 5, 3)

        for budget in budgets:
            gtt.set_budget(budget)
            assignments = gtt.run_opt(test_covariate_dataset)
            metrics = post_transfer_metrics(test_dataset=test_dataset, assignments=assignments, c_bar=c_bar)
            np.testing.assert_allclose(metrics['policy_cost_per_capita'], budget, rtol=1e-2)

    def test_double_gap_targeting(self):
        # Add your test implementation here

        n = 20000
        dataset = self._make_2d_data(n=n, seed=42)
        train_dataset, val_dataset = split(dataset, frac=0.6, seed=42)
        test_dataset = self._make_2d_data(n=5000, seed=24)
        test_covariate_dataset = Dataset(
            test_dataset.df,
            outcome=None,
            covs=test_dataset.covs,
            weight=test_dataset.weight,
        )

        c_bar = 3

        gtt = GapTargetedTransfers(c_bar=c_bar, n_regressors=10)
        gtt.fit(train_dataset, val_dataset)
        budget = 0.5

        gtt.set_budget(budget)
        assignments = gtt.run_opt(test_covariate_dataset)

        dataset.df["outcome"] *= 2.0
        train_dataset, val_dataset = split(dataset, frac=0.6, seed=42)
        test_dataset.df["outcome"] *= 2.0
        test_covariate_dataset = Dataset(
            test_dataset.df,
            outcome=None,
            covs=test_dataset.covs,
            weight=test_dataset.weight,
        )

        double_cbar = 6
        double_gtt = GapTargetedTransfers(c_bar=double_cbar, n_regressors=10)
        double_gtt.fit(train_dataset, val_dataset)
        double_gtt.set_budget(budget * 2)
        double_assignments = double_gtt.run_opt(test_covariate_dataset)
        for i in range(len(assignments)):
            for j in range(len(assignments[i])):
                self.assertAlmostEqual(
                    assignments[i][j][0] * 2.0, double_assignments[i][j][0], delta=1e-2
                )


if __name__ == "__main__":
    unittest.main()
