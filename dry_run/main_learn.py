import yaml
import argh
import numpy as np
from feature_selection import forward_selection
from opt_targeted_transfers import (
    RateTargetedTransfers,
    GapTargetedTransfers,
    BinaryRateTargetedTransfers,
    BinaryGapTargetedTransfers,
    write_result,
    split,
)
from data_loader import load_datasets
from constants import C_BAR, BUDGETS


def run_evaluation(tt, test_covariate_dataset, test_dataset, savepath):
    for budget in BUDGETS:
        tt.set_budget(budget)
        tt.run_opt(test_covariate_dataset)
        res = tt.evaluate(test_dataset)
        write_result(savepath + ".csv", res)
    auc_res = tt.compute_auc(
        test_dataset=test_dataset,
        test_covariate_dataset=test_covariate_dataset,
        metrics=["post_transfer_poverty_rate", "post_transfer_poverty_gap"],
        budgets=BUDGETS,
    )
    for metric in auc_res:
        del auc_res[metric]["results"]
    write_result(savepath + "_auc.csv", auc_res)


def learn_continuous_rate(
    train_dataset,
    validation_dataset,
    test_covariate_dataset,
    test_dataset,
    continuous_rate_params,
    savepath,
):
    """
    Learn the continuous rate targeted transfers
    """
    print("Learning continuous rate targeted transfers...")
    print("Run forward selection...")
    features, _ = forward_selection(
        train_dataset,
        validation_dataset,
        max_features=int(continuous_rate_params["density_estimation"]["n_features"]),
    )
    train_dataset.covs = features
    validation_dataset.covs = features
    test_covariate_dataset.covs = features
    test_dataset.covs = features

    tt = RateTargetedTransfers(c_bar=C_BAR)
    tt.fit(
        train_dataset,
        validation_dataset,
        n_knots=int(continuous_rate_params["density_estimation"]["n_knots"]),
        n_bins=int(continuous_rate_params["density_estimation"]["n_bins"]),
        degree=int(continuous_rate_params["density_estimation"]["degree"]),
    )
    for budget in BUDGETS:
        tt.set_budget(budget)
        tt.run_opt(
            test_covariate_dataset, n_alpha=int(continuous_rate_params["n_alpha"])
        )
        res = tt.evaluate(test_dataset)
        write_result(savepath + ".csv", res)
    auc_res = tt.compute_auc(
        test_dataset=test_dataset,
        test_covariate_dataset=test_covariate_dataset,
        metrics=["post_transfer_poverty_rate", "post_transfer_poverty_gap"],
        budgets=BUDGETS,
    )
    for metric in auc_res:
        del auc_res[metric]["results"]
    write_result(savepath + "_auc.csv", auc_res)


def learn_binary_rate(
    train_dataset,
    validation_dataset,
    test_covariate_dataset,
    test_dataset,
    binary_rate_params,
    savepath,
):
    """
    Learn the binary rate targeted transfers
    """
    print("Learning binary rate targeted transfers...")
    tt = BinaryRateTargetedTransfers(
        c_bar=C_BAR, n_transfer_values=int(binary_rate_params["n_transfer_values"])
    )
    for hparam in binary_rate_params["neural_network"]:
        binary_rate_params["neural_network"][hparam] = int(
            binary_rate_params["neural_network"][hparam]
        )
    tt.fit(train_dataset, validation_dataset, **binary_rate_params["neural_network"])
    tt.optimize_transfers_for_budget_grid(test_covariate_dataset, BUDGETS)
    run_evaluation(tt, test_covariate_dataset, test_dataset, savepath)


def learn_continuous_gap(
    train_dataset,
    validation_dataset,
    test_covariate_dataset,
    test_dataset,
    continuous_gap_params,
    savepath,
):
    """
    Learn the continuous gap targeted transfers
    """
    print("Learning continuous gap targeted transfers...")
    tt = GapTargetedTransfers(c_bar=C_BAR)
    for hparam in continuous_gap_params["neural_network"]:
        continuous_gap_params["neural_network"][hparam] = int(
            continuous_gap_params["neural_network"][hparam]
        )
    tt.fit(train_dataset, validation_dataset, **continuous_gap_params["neural_network"])
    run_evaluation(tt, test_covariate_dataset, test_dataset, savepath)


def learn_binary_gap(
    train_dataset,
    validation_dataset,
    test_covariate_dataset,
    test_dataset,
    binary_gap_params,
    savepath,
):
    """
    Learn the binary gap targeted transfers
    """
    print("Learning binary gap targeted transfers...")
    tt = BinaryGapTargetedTransfers(
        c_bar=C_BAR, n_transfer_values=int(binary_gap_params["n_transfer_values"])
    )
    for hparam in binary_gap_params["neural_network"]:
        binary_gap_params["neural_network"][hparam] = int(
            binary_gap_params["neural_network"][hparam]
        )
    tt.fit(train_dataset, validation_dataset, **binary_gap_params["neural_network"])
    tt.optimize_transfers_for_budget_grid(test_covariate_dataset, BUDGETS)
    run_evaluation(tt, test_covariate_dataset, test_dataset, savepath)


@argh.arg("--config", default="hparam_results/output_gan_continuous_rate.yaml")
@argh.arg("--trainpath", default="data/train.parquet")
@argh.arg("--testpath", default="data/test.parquet")
@argh.arg("--savedir", default="learn_results")
def main(
    config="hparam_results/output_gan_continuous_rate.yaml",
    trainpath="data/train.parquet",
    testpath="data/test.parquet",
    savedir="learn_results",
):
    """
    Main function to learn and evaluate targeted transfers.
    """
    with open(config) as stream:
        try:
            config_hparam = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    config_keys = list(config_hparam.keys())
    assert "data" in config_keys
    assert all(
        [
            key
            in [
                "continuous_rate",
                "binary_rate",
                "continuous_gap",
                "binary_gap",
                "data",
            ]
            for key in config_keys
        ]
    )

    data_config = config_hparam["data"]

    train_dataset, validation_dataset, test_covariate_dataset, test_dataset = (
        load_datasets(
            trainpath,
            testpath,
            outcome=data_config["outcome"],
            weight=data_config["weight"],
        )
    )

    name = config.split("/")[1].split(".yaml")[0]
    savepath = savedir + "/" + name

    for key in config_keys:
        if key == "continuous_rate":
            continuous_rate_params = config_hparam[key]
            learn_continuous_rate(
                train_dataset,
                validation_dataset,
                test_covariate_dataset,
                test_dataset,
                continuous_rate_params,
                savepath,
            )
        elif key == "binary_rate":
            binary_rate_params = config_hparam[key]
        elif key == "continuous_gap":
            continuous_gap_params = config_hparam[key]
        elif key == "binary_gap":
            binary_gap_params = config_hparam[key]


if __name__ == "__main__":
    _parser = argh.ArghParser()
    _parser.add_commands([main])
    _parser.dispatch()
