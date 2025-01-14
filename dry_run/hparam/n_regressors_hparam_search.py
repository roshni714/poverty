from opt_targeted_transfers import BinaryRateTargetedTransfers, BinaryGapTargetedTransfers, GapTargetedTransfers
from opt_targeted_transfers import Dataset
import pandas as pd
from constants import C_BAR, BUDGETS


def get_optimal_n_regressors(
    n_regressors_range,
    loss_type,
    data_generator,
    original_cols,
    ntrain,
    nval,
    ntest,
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

    train_df = data_generator(nsamples=ntrain, seed=547396234)
    val_df = data_generator(nsamples=nval, seed=79809342)
    feature_list = original_cols
    feature_list.remove(outcome)
    feature_list.remove(weight)
    train_dataset = Dataset(train_df, outcome=outcome, weight=weight, covs=feature_list)
    val_dataset = Dataset(val_df, outcome=outcome, weight=weight, covs=feature_list)

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
        test_df = data_generator(nsamples=ntest, seed=79809242 + trial)
        test_covariate_dataset = Dataset(
            test_df, outcome=None, weight=weight, covs=feature_list
        )
        test_dataset = Dataset(
            test_df, outcome=outcome, weight=weight, covs=feature_list
        )
        for n_regressors in n_regressors_range:
            tt = TT(c_bar=C_BAR, n_regressors = n_regressors)
            tt.fit(train_dataset, val_dataset, **neural_network_params)
            tt.optimize_transfers_for_budget_grid(test_covariate_dataset=test_covariate_dataset, budgets=BUDGETS)

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
    df = df.groupby(["n_transfer_values"]).mean().reset_index()
    optimal_params = df.loc[df["auc"].idxmin()].to_dict()
    return optimal_params["n_transfer_values"]
