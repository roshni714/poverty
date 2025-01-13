from opt_targeted_transfers import RateTargetedTransfers
from opt_targeted_transfers import Dataset
import pandas as pd
from constants import C_BAR, BUDGETS


def get_optimal_knapsack_parameters(
    n_alpha_range,
    data_generator,
    original_cols,
    ntrain,
    nval,
    ntest,
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

    tt = RateTargetedTransfers(c_bar=C_BAR)
    tt.fit(train_dataset, val_dataset)

    results = []
    for trial in range(3):
        test_df = data_generator(nsamples=ntest, seed=79809242 + trial)
        test_covariate_dataset = Dataset(
            test_df, outcome=None, weight=weight, covs=feature_list
        )
        test_dataset = Dataset(
            test_df, outcome=outcome, weight=weight, covs=feature_list
        )

        for n_alpha in n_alpha_range:
            res = tt.compute_auc(
                test_dataset=test_dataset,
                test_covariate_dataset=test_covariate_dataset,
                metrics=["post_transfer_poverty_rate"],
                budgets=BUDGETS,
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
    del optimal_params["auc"]
    del optimal_params["trial"]
    return optimal_params
