from opt_targeted_transfers import get_cond_density_estimator, Dataset, get_nll, split
from feature_selection import forward_selection
import pandas as pd


def get_optimal_density_estimation_parameters(
    density_estimation_hparam_ranges,
    data_generator,
    original_cols,
    ntrain,
    nval,
    outcome,
    weight,
    savepath,
):
    """
    Get optimal density estimation hyperparameters.

    Args:
        density_estimation_hparam_ranges (dict): Dictionary containing the hyperparameter ranges for density estimation.
        data_generator (DataGenerator): Data generator object.
        original_cols (list): List of columns in dataset before one-hot encoding.
        ntrain (int): Number of training samples.
        nval (int): Number of validation samples.
        outcome (str): Outcome variable.
        weight (str): Weight variable.
        savepath (str): Path to save results.


    Returns:
        opt_params (dict): Optimal hyperparameters for density estimation.
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
        n_bins_range = [100]

    if "n_knots" in density_estimation_hparam_ranges:
        n_knots_range = density_estimation_hparam_ranges["n_knots"]
    else:
        n_knots_range = [4]

    train_df = data_generator(nsamples=ntrain, seed=547396234)
    val_df = data_generator(nsamples=nval, seed=79809342)
    feature_list = original_cols
    feature_list.remove(outcome)
    feature_list.remove(weight)
    train_dataset = Dataset(train_df, outcome=outcome, weight=weight, covs=feature_list)
    val_dataset = Dataset(val_df, outcome=outcome, weight=weight, covs=feature_list)

    print("Running forward selection...")
    ordered_features, _ = forward_selection(
        train_dataset, val_dataset, max_features=max(n_features_range)
    )

    results = []
    for trial in range(3):
        train_df = data_generator(nsamples=ntrain, seed=1283 + trial)
        val_df = data_generator(nsamples=nval, seed=4308 + trial)

        for n_features in n_features_range:
            train_dataset = Dataset(
                train_df,
                outcome=outcome,
                weight=weight,
                covs=ordered_features[:n_features],
            )
            big_val_dataset = Dataset(
                val_df,
                outcome=outcome,
                weight=weight,
                covs=ordered_features[:n_features],
            )
            for degree in degree_range:
                for n_bins in n_bins_range:
                    for n_knots in n_knots_range:
                        train_dataset, val_dataset = split(train_dataset)
                        density_estimator = get_cond_density_estimator(
                            train_dataset=train_dataset,
                            validation_dataset=val_dataset,
                            degree=degree,
                            n_bins=n_bins,
                            n_knots=n_knots,
                        )
                        nll = get_nll(big_val_dataset, density_estimator)
                        results.append(
                            {
                                "n_features": n_features,
                                "degree": degree,
                                "n_bins": n_bins,
                                "n_knots": n_knots,
                                "nll": nll,
                                "trial": trial,
                            }
                        )
                        print(results[-1])
    df = pd.DataFrame.from_records(results)
    df.to_csv(savepath, index=False)
    df = df.groupby(["n_features", "degree", "n_bins", "n_knots"]).mean().reset_index()
    optimal_params = df.loc[df["nll"].idxmin()].to_dict()
    del optimal_params["nll"]
    del optimal_params["trial"]
    return optimal_params
