from opt_targeted_transfers import (
    Dataset,
    split,
    get_quantile_regressor,
    get_quantile_loss,
    get_conditional_improvement_regressor,
    get_conditional_improvement_loss,
)
import pandas as pd
import numpy as np
from constants import C_BAR


def get_optimal_nn_improvement_parameters(
    loss_type,
    nn_hparam_ranges,
    data_generator,
    device,
    original_cols,
    ntrain,
    nval,
    outcome,
    weight,
    savepath,
):
    """
    Get optimal neural network hyperparameters for gap improvement.

    Args:
        nn_hparam_ranges (dict): Hyperparameter ranges for neural network.
        data_generator (generator): Data generator.
        device (str): Device to train the neural network on.
        original_cols (list): Original columns before one-hot encoding.
        ntrain (int): Number of training samples.
        nval (int): Number of validation samples.
        outcome (str): Outcome variable.
        weight (str): Weight variable.
        savepath (str): Path to save results.

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
        n_hidden_units_range = [2 ** round(np.log(original_cols), 1)]
    if "lr" in nn_hparam_ranges:
        lr_range = nn_hparam_ranges["lr"]
    else:
        lr_range = [5e-3]

    results = []
    transfer_sizes = [0.5, 1.0, 1.5]

    for trial in range(3):
        train_df = data_generator(nsamples=ntrain, seed=54734234 + trial)
        val_df = data_generator(nsamples=nval, seed=7959342 + trial)
        covs = original_cols.copy()
        covs.remove(outcome)
        covs.remove(weight)
        train_dataset = Dataset(train_df, outcome=outcome, weight=weight, covs=covs)
        big_val_dataset = Dataset(val_df, outcome=outcome, weight=weight, covs=covs)
        new_train_dataset, new_val_dataset = split(train_dataset)
        for transfer_size in transfer_sizes:
            for n_layers in n_layers_range:
                for n_hidden_units in n_hidden_units_range:
                    for lr in lr_range:
                        print(
                            f"Training neural network with {n_layers} layers, {n_hidden_units} hidden units, and learning rate {lr} for transfer size {transfer_size} during trial {trial}..."
                        )

                        model = get_conditional_improvement_regressor(
                            loss_type=loss_type,
                            train_dataset=new_train_dataset,
                            validation_dataset=new_val_dataset,
                            t=transfer_size,
                            c_bar=C_BAR,
                            n_layers=n_layers,
                            n_hidden_units=n_hidden_units,
                            lr=lr,
                            device=device,
                        )
                        loss = get_conditional_improvement_loss(
                            big_val_dataset,
                            loss_type=loss_type,
                            predictor=model,
                            t=transfer_size,
                            c_bar=C_BAR,
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
    df = pd.DataFrame.from_records(results)
    df.to_csv(savepath, index=False)
    df = df.groupby(["n_layers", "n_hidden_units", "lr"]).mean().reset_index()
    optimal_params = df.loc[df["loss"].idxmin()].to_dict()
    del optimal_params["loss"]
    del optimal_params["transfer_size"]
    del optimal_params["trial"]
    for hparam in ["n_hidden_units", "n_layers"]:
        optimal_params[hparam] = int(optimal_params[hparam])
    return optimal_params


def get_optimal_nn_quantile_regression_parameters(
    nn_hparam_ranges,
    data_generator,
    device,
    original_cols,
    ntrain,
    nval,
    outcome,
    weight,
    savepath,
):
    """
    Get optimal neural network hyperparameters for quantile regression.

    Args:
        nn_hparam_ranges (dict): Hyperparameter ranges for neural network.
        data_generator (generator): Data generator.
        device (str): Device to train the neural network on.
        original_cols (list): Original columns before one-hot encoding.
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
        n_hidden_units_range = [2 ** round(np.log(original_cols), 1)]
    if "lr" in nn_hparam_ranges:
        lr_range = nn_hparam_ranges["lr"]
    else:
        lr_range = [5e-3]

    results = []
    quantiles = [0.25, 0.5, 0.75]

    for trial in range(3):
        train_df = data_generator(nsamples=ntrain, seed=54734234 + trial)
        val_df = data_generator(nsamples=nval, seed=7959342 + trial)
        covs = original_cols.copy()
        covs.remove(outcome)
        covs.remove(weight)
        train_dataset = Dataset(train_df, outcome=outcome, weight=weight, covs=covs)
        big_val_dataset = Dataset(val_df, outcome=outcome, weight=weight, covs=covs)
        new_train_dataset, new_val_dataset = split(train_dataset)
        for quantile in quantiles:
            for n_layers in n_layers_range:
                for n_hidden_units in n_hidden_units_range:
                    for lr in lr_range:
                        print(
                            f"Training neural network with {n_layers} layers, {n_hidden_units} hidden units, and learning rate {lr} for quantile {quantile} during trial {trial}..."
                        )
                        model = get_quantile_regressor(
                            train_dataset=new_train_dataset,
                            validation_dataset=new_val_dataset,
                            quantile=quantile,
                            n_layers=n_layers,
                            n_hidden_units=n_hidden_units,
                            lr=lr,
                            device=device,
                        )
                        pinball_loss = get_quantile_loss(
                            validation_dataset=big_val_dataset,
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
    df = pd.DataFrame.from_records(results)
    df.to_csv(savepath, index=False)
    df = df.groupby(["n_layers", "n_hidden_units", "lr"]).mean().reset_index()
    optimal_params = df.loc[df["pinball_loss"].idxmin()].to_dict()
    del optimal_params["pinball_loss"]
    del optimal_params["quantile"]
    del optimal_params["trial"]
    for hparam in ["n_hidden_units", "n_layers"]:
        optimal_params[hparam] = int(optimal_params[hparam])
    return optimal_params
