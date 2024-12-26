from opt_targeted_transfers import get_cond_density_estimator, Dataset, get_nll
from feature_selection import forward_selection
import pandas as pd


def get_optimal_density_estimation_parameters(
    density_estimation_hparam_ranges, data_generator, ntrain, nval
):
    """
    Get optimal density estimation hyperparameters.

    Args:
        density_estimation_hparam_ranges (dict): Dictionary containing the hyperparameter ranges for density estimation.
        data_generator (DataGenerator): Data generator object.

    Returns:
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
        n_bins_range = [10]

    if "n_knots" in density_estimation_hparam_ranges:
        n_knots_range = density_estimation_hparam_ranges["n_knots"]
    else:
        n_knots_range = [4]

    train_df = data_generator(nsamples=ntrain, seed=547396234)
    val_df = data_generator(nsamples=nval, seed=79809342)
    train_dataset = Dataset(
        train_df, outcome="consumption_per_capita_per_day", weight="hh_wgt", covs=[]
    )
    val_dataset = Dataset(
        val_df, outcome="consumption_per_capita_per_day", weight="hh_wgt", covs=[]
    )

    print("Running forward selection...")
    ordered_features, _ = forward_selection(
        train_dataset, val_dataset, max_features=max(n_features_range)
    )

    results = []
    for i, n_features in enumerate(n_features_range):
        train_df = data_generator(nsamples=ntrain, seed=1283 * i + 1)
        val_df = data_generator(nsamples=nval, seed=4308 * i + 1)
        train_dataset = Dataset(
            train_df,
            outcome="consumption_per_capita_per_day",
            weight="hh_wgt",
            covs=ordered_features[:n_features],
        )
        val_dataset = Dataset(
            val_df,
            outcome="consumption_per_capita_per_day",
            weight="hh_wgt",
            covs=ordered_features[:n_features],
        )
        for degree in degree_range:
            for n_bins in n_bins_range:
                for n_knots in n_knots_range:
                    density_estimator = get_cond_density_estimator(
                        train_dataset,
                        n_features=n_features,
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
    optimal_params = df.loc[df["nll"].idxmin()].to_dict()
    del optimal_params["nll"]
    return optimal_params
