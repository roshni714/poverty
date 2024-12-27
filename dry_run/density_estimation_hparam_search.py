from opt_targeted_transfers import get_cond_density_estimator, Dataset, get_nll
from feature_selection import forward_selection
import pandas as pd


def get_optimal_density_estimation_parameters(
    density_estimation_hparam_ranges,
    data_generator,
    ntrain,
    nval,
    outcome,
    weight,
    savedir,
):
    """
    Get optimal density estimation hyperparameters.

    Args:
        density_estimation_hparam_ranges (dict): Dictionary containing the hyperparameter ranges for density estimation.
        data_generator (DataGenerator): Data generator object.
        ntrain (int): Number of training samples.
        nval (int): Number of validation samples.
        outcome (str): Outcome variable.
        weight (str): Weight variable.
        savedir (str): Directory to save the results.

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
    train_dataset = Dataset(train_df, outcome=outcome, weight=weight, covs=[])
    val_dataset = Dataset(val_df, outcome=outcome, weight=weight, covs=[])

    print("Running forward selection...")
    ordered_features, _ = forward_selection(
        train_dataset, val_dataset, max_features=max(n_features_range)
    )

    train_df = data_generator(nsamples=ntrain, seed=1283)
    val_df = data_generator(nsamples=nval, seed=4308)

    results = []
    for n_features in n_features_range:
        train_dataset = Dataset(
            train_df,
            outcome=outcome,
            weight=weight,
            covs=ordered_features[:n_features],
        )
        val_dataset = Dataset(
            val_df,
            outcome=outcome,
            weight=weight,
            covs=ordered_features[:n_features],
        )
        for degree in degree_range:
            for n_bins in n_bins_range:
                for n_knots in n_knots_range:
                    density_estimator = get_cond_density_estimator(
                        train_dataset,
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
    df.to_csv("{}/density_estimation.csv".format(savedir))
    optimal_params = df.loc[df["nll"].idxmin()].to_dict()
    del optimal_params["nll"]
    return optimal_params
