from opt_targeted_transfers import (
    BinaryRateTargetedTransfers,
    BinaryGapTargetedTransfers,
    GapTargetedTransfers,
)
from opt_targeted_transfers import Dataset, split
import pandas as pd
from constants import C_BAR, BUDGETS


def get_optimal_n_regressors(
    n_regressors_range,
    loss_type,
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

    results = []
    for trial in range(3):
        train_df = data_generator(nsamples=ntrain, seed=547396234)
        train_dataset = Dataset(
            train_df, outcome=outcome, weight=weight, covs=feature_list
        )

        new_train_dataset, new_val_dataset = split(train_dataset)
        for n_regressors in n_regressors_range:
            tt = TT(c_bar=C_BAR, n_regressors=n_regressors)
            tt.fit(
                new_train_dataset,
                new_val_dataset,
                device=device,
                **neural_network_params,
            )
            if "binary" in loss_type:
                tt.get_opt_transfer_sizes_given_budget_grid(new_val_dataset, BUDGETS)
            res = tt.compute_auc(
                test_dataset=test_dataset,
                test_covariate_dataset=test_covariate_dataset,
                metrics=[metric],
                budgets=BUDGETS,
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
