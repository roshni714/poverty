from opt_targeted_transfers import (
    BinaryRateTargetedTransfers,
    BinaryGapTargetedTransfers,
    GapTargetedTransfers,
    WelfareTargetedTransfers,
)
from opt_targeted_transfers import Dataset, split
import pandas as pd
import numpy as np


def get_optimal_n_regressors(
    n_regressors_range,
    loss_type,
    povertyline,
    data_generator,
    val_df,
    device,
    original_cols,
    ntrain,
    outcome,
    weight,
    neural_network_params,
    savepath,
):
    """
    Get optimal density estimation hyperparameters.

    Args:
    n_regressors_range: Range of n_regressors values to search over.
    loss_type (str): Type of loss function to use.
    data_generator (DataGenerator): Data generator object.
    original_cols (list): List of columns in dataset before one-hot encoding.
    ntrain (int): Number of training samples.
    nval (int): Number of validation samples.
    outcome (str): Outcome variable.
    weight (str): Weight variable.
    savepath (str): Path to save results.


    Returns:
        opt_params (dict): Optimal hyperparameters for knapsack.
    """

    feature_list = original_cols.copy()
    feature_list.remove(outcome)
    feature_list.remove(weight)
    test_covariate_dataset = Dataset(
        val_df, outcome=None, weight=weight, covs=feature_list
    )
    test_dataset = Dataset(val_df, outcome=outcome, weight=weight, covs=feature_list)

    if loss_type == "binary_rate":
        TT = BinaryRateTargetedTransfers
        metric = "post_transfer_poverty_rate"
    elif loss_type == "binary_gap":
        TT = BinaryGapTargetedTransfers
        metric = "post_transfer_poverty_gap"
    elif loss_type == "continuous_gap":
        TT = GapTargetedTransfers
        metric = "post_transfer_poverty_gap"
    elif loss_type == "welfare":
        TT = WelfareTargetedTransfers
        metric = "post_transfer_welfare"

    results = []
    for trial in range(3):
        train_df = data_generator(nsamples=ntrain, seed=547396234 + trial)
        train_dataset = Dataset(
            train_df, outcome=outcome, weight=weight, covs=feature_list
        )

        new_train_dataset, new_val_dataset = split(train_dataset)
        budgets = np.linspace(0.05, povertyline, 15)
        for n_regressors in n_regressors_range:
            tt = TT(c_bar=povertyline, n_regressors=n_regressors)
            tt.fit(
                new_train_dataset,
                new_val_dataset,
                device=device,
                **neural_network_params,
            )
            if "binary" in loss_type:
                tt.get_opt_transfer_sizes_given_budget_grid(new_val_dataset, budgets)
            res = tt.compute_auc(
                test_dataset=test_dataset,
                test_covariate_dataset=test_covariate_dataset,
                metrics=[metric],
                budgets=budgets,
            )
            results.append(
                {
                    "n_regressors": n_regressors,
                    "trial": trial,
                    "auc": res[metric]["auc"],
                }
            )

    df = pd.DataFrame.from_records(results)
    df.to_csv(savepath, index=False)
    df = df.groupby(["n_regressors"]).mean().reset_index()
    optimal_params = df.loc[df["auc"].idxmin()].to_dict()
    for hparam in optimal_params:
        optimal_params[hparam] = int(optimal_params[hparam])
    return optimal_params["n_regressors"]
