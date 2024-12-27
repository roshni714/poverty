from opt_targeted_transfers import (
    Dataset,
    get_quantile_regressor,
    get_quantile_loss,
    get_conditional_gap_improvement_regressor,
    get_conditional_gap_improvement_loss,
)
import pandas as pd
import numpy as np


def get_optimal_nn_gap_improvement_parameters(
    nn_hparam_ranges, data_generator, ntrain, nval, outcome, weight, savedir
):
    """
    Get optimal neural network hyperparameters for gap improvement.

    Args:
        nn_hparam_ranges (dict): Hyperparameter ranges for neural network.
        data_generator (generator): Data generator.
        ntrain (int): Number of training samples.
        nval (int): Number of validation samples.
        outcome (str): Outcome variable.
        weight (str): Weight variable.

    Returns:
        opt_params: Optimal hyperparameters for neural network.
    """

    if "n_layers" in nn_hparam_ranges:
        n_layers_range = nn_hparam_ranges["n_layers"]
    else:
        n_layers_range = [1]

    if "n_hidden_units" in nn_hparam_ranges:
        n_hidden_units_range = nn_hparam_ranges["n_hidden_units"]
    else:
        n_hidden_units_range = [64]
    if "lr" in nn_hparam_ranges:
        lr_range = nn_hparam_ranges["lr"]
    else:
        lr_range = [5e-3]

    results = []
    transfer_sizes = np.linspace(0.01, 2.0, 5)

    for trial in range(5):
        train_df = data_generator(nsamples=ntrain, seed=54734234 + trial)
        val_df = data_generator(nsamples=nval, seed=7959342 + trial)
        covs = train_df.columns.difference([outcome, weight]).tolist()
        train_dataset = Dataset(train_df, outcome=outcome, weight=weight, covs=covs)
        val_dataset = Dataset(val_df, outcome=outcome, weight=weight, covs=covs)

        for transfer_size in transfer_sizes:
            for n_layers in n_layers_range:
                for n_hidden_units in n_hidden_units_range:
                    for lr in lr_range:
                        print(
                            f"Training neural network with {n_layers} layers, {n_hidden_units} hidden units, and learning rate {lr} for transfer size {transfer_size} during trial {trial}..."
                        )
                        model = get_conditional_gap_improvement_regressor(
                            dataset=train_dataset,
                            t=transfer_size,
                            c_bar=2.15,
                            n_layers=n_layers,
                            n_hidden_units=n_hidden_units,
                            lr=lr,
                        )
                        loss = get_conditional_gap_improvement_loss(
                            val_dataset, model, t=transfer_size
                        ).item()
                        results.append(
                            {
                                "transfer_size": transfer_size,
                                "n_layers": n_layers,
                                "n_hidden_units": n_hidden_units,
                                "lr": lr,
                                "loss": loss,
                                "trial": trial,
                            }
                        )
                        print(loss)
    df = pd.DataFrame.from_records(results)
    df = df.groupby(["n_layers", "n_hidden_units", "lr"]).mean().reset_index()
    df.to_csv(f"{savedir}/nn_gap_improvement_results.csv", index=False)
    optimal_params = df.loc[df["loss"].idxmin()].to_dict()
    del optimal_params["loss"]
    del optimal_params["transfer_size"]
    del optimal_params["trial"]
    return optimal_params


def get_optimal_nn_quantile_regression_parameters(
    nn_hparam_ranges, data_generator, ntrain, nval, outcome, weight, savedir
):
    """
    Get optimal neural network hyperparameters for quantile regression.

    Args:
        nn_hparam_ranges (dict): Hyperparameter ranges for neural network.
        data_generator (generator): Data generator.
        ntrain (int): Number of training samples.
        nval (int): Number of validation samples.
        outcome (str): Outcome variable.
        weight (str): Weight variable.

    Returns:
        opt_params: Optimal hyperparameters for neural network.
    """

    if "n_layers" in nn_hparam_ranges:
        n_layers_range = nn_hparam_ranges["n_layers"]
    else:
        n_layers_range = [1]

    if "n_hidden_units" in nn_hparam_ranges:
        n_hidden_units_range = nn_hparam_ranges["n_hidden_units"]
    else:
        n_hidden_units_range = [64]
    if "lr" in nn_hparam_ranges:
        lr_range = nn_hparam_ranges["lr"]
    else:
        lr_range = [5e-3]

    results = []
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

    for trial in range(5):
        train_df = data_generator(nsamples=ntrain, seed=54734234 + trial)
        val_df = data_generator(nsamples=nval, seed=7959342 + trial)
        covs = train_df.columns.difference([outcome, weight]).tolist()
        train_dataset = Dataset(train_df, outcome=outcome, weight=weight, covs=covs)
        val_dataset = Dataset(val_df, outcome=outcome, weight=weight, covs=covs)

        for quantile in quantiles:
            for n_layers in n_layers_range:
                for n_hidden_units in n_hidden_units_range:
                    for lr in lr_range:
                        print(
                            f"Training neural network with {n_layers} layers, {n_hidden_units} hidden units, and learning rate {lr} for quantile {quantile} during trial {trial}..."
                        )
                        model = get_quantile_regressor(
                            dataset=train_dataset,
                            quantile=quantile,
                            n_layers=n_layers,
                            n_hidden_units=n_hidden_units,
                            lr=lr,
                        )
                        pinball_loss = get_quantile_loss(
                            val_dataset=val_dataset,
                            quantile_regressor=model,
                            quantile=quantile,
                        ).item()
                        results.append(
                            {
                                "quantile": quantile,
                                "n_layers": n_layers,
                                "n_hidden_units": n_hidden_units,
                                "lr": lr,
                                "pinball_loss": pinball_loss,
                                "trial": trial,
                            }
                        )
                        print(pinball_loss)
    df = pd.DataFrame.from_records(results)
    df = df.groupby(["n_layers", "n_hidden_units", "lr"]).mean().reset_index()
    df.to_csv(f"{savedir}/nn_quantile_regression_results.csv", index=False)
    optimal_params = df.loc[df["pinball_loss"].idxmin()].to_dict()
    del optimal_params["pinball_loss"]
    del optimal_params["quantile"]
    del optimal_params["trial"]
    return optimal_params
