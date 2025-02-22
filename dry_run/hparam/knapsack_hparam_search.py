from opt_targeted_transfers import RateTargetedTransfers
from opt_targeted_transfers import Dataset, split
from feature_selection import forward_selection
import pandas as pd
from constants import C_BAR, BUDGETS


def get_optimal_knapsack_parameters(
    n_alpha_range,
    data_generator,
    val_df,
    device,
    original_cols,
    ntrain,
    outcome,
    weight,
    density_estimation_params,
    savepath,
):
    """
    Get optimal density estimation hyperparameters.

    Args:
    n_alpha_range: Range of n_alpha values to search over.
    data_generator (DataGenerator): Data generator object.
    device (str): Device to train the density estimator on.
    original_cols (list): List of columns in dataset before one-hot encoding.
    ntrain (int): Number of training samples.
    nval (int): Number of validation samples.
    outcome (str): Outcome variable.
    weight (str): Weight variable.
    savepath (str): Path to save results.


    Returns:
        opt_params (dict): Optimal hyperparameters for knapsack.
    """

    results = []
    for trial in range(3):
        feature_list = original_cols.copy()
        feature_list.remove(outcome)
        feature_list.remove(weight)
        train_df = data_generator(nsamples=ntrain, seed=547396234 + trial)
        train_dataset = Dataset(
            train_df, outcome=outcome, weight=weight, covs=feature_list
        )

        tt = RateTargetedTransfers(c_bar=C_BAR)
        new_train_dataset, new_val_dataset = split(train_dataset)

        features, _ = forward_selection(
            train_dataset=new_train_dataset,
            validation_dataset=new_val_dataset,
            max_features=density_estimation_params["n_features"],
        )
        new_train_dataset.covs = features
        new_val_dataset.covs = features

        tt.fit(
            new_train_dataset,
            new_val_dataset,
            n_bins=density_estimation_params["n_bins"],
            n_knots=density_estimation_params["n_knots"],
            degree=density_estimation_params["degree"],
            device=device,
        )

        test_covariate_dataset = Dataset(
            val_df, outcome=None, weight=weight, covs=features
        )
        test_dataset = Dataset(val_df, outcome=outcome, weight=weight, covs=features)

        for n_alpha in n_alpha_range:
            res = tt.compute_auc(
                test_dataset=test_dataset,
                test_covariate_dataset=test_covariate_dataset,
                metrics=["post_transfer_poverty_rate"],
                budgets=BUDGETS,
                n_alpha=n_alpha,
            )
            results.append(
                {
                    "n_alpha": n_alpha,
                    "trial": trial,
                    "auc": res["post_transfer_poverty_rate"]["auc"],
                }
            )

    df = pd.DataFrame.from_records(results)
    df.to_csv(savepath, index=False)
    df = df.groupby(["n_alpha"]).mean().reset_index()
    optimal_params = df.loc[df["auc"].idxmin()].to_dict()
    for hparam in optimal_params:
        optimal_params[hparam] = int(optimal_params[hparam])
    return optimal_params["n_alpha"]
