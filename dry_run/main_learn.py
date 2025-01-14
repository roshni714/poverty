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
)
from learn.data_loader import load_datasets
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
    new_dic = {}
    for metric in auc_res:
        new_dic[metric] = auc_res[metric]["auc"]
    write_result(savepath + "_auc.csv", new_dic)


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
    all_res = []
    for budget in BUDGETS:
        tt.set_budget(budget)
        tt.run_opt(
            test_covariate_dataset, n_alpha=int(continuous_rate_params["n_alpha"])
        )
        res = tt.evaluate(test_dataset)
        write_result(savepath + ".csv", res)
        all_res.append(res)

    auc_res = {}
    metrics = ["post_transfer_poverty_rate", "post_transfer_poverty_gap"]
    for metric in metrics:
        y_items = []
        for res in all_res:
            y_items.append(res[metric])
        auc = np.trapz(y_items, x=BUDGETS)
        auc_res[metric] = auc

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
        c_bar=C_BAR, n_regressors=int(binary_rate_params["n_regressors"])
    )
    for hparam in binary_rate_params["neural_network"]:
        binary_rate_params["neural_network"][hparam] = int(
            binary_rate_params["neural_network"][hparam]
        )
    tt.fit(
        train_dataset,
        validation_dataset,
        n_layers=binary_rate_params["neural_network"]["n_layers"],
        n_hidden_units=binary_rate_params["neural_network"]["n_hidden_units"],
        lr=binary_rate_params["neural_network"]["lr"],
    )
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
    tt = GapTargetedTransfers(
        c_bar=C_BAR, n_regressors=int(continuous_gap_params["n_regressors"])
    )
    for hparam in continuous_gap_params["neural_network"]:
        continuous_gap_params["neural_network"][hparam] = int(
            continuous_gap_params["neural_network"][hparam]
        )
    tt.fit(
        train_dataset,
        validation_dataset,
        n_layers=continuous_gap_params["neural_network"]["n_layers"],
        n_hidden_units=continuous_gap_params["neural_network"]["n_hidden_units"],
        lr=continuous_gap_params["neural_network"]["lr"],
    )
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
        c_bar=C_BAR, n_regressors=int(binary_gap_params["n_regressors"])
    )
    for hparam in binary_gap_params["neural_network"]:
        binary_gap_params["neural_network"][hparam] = int(
            binary_gap_params["neural_network"][hparam]
        )
    tt.fit(
        train_dataset,
        validation_dataset,
        n_layers=binary_gap_params["neural_network"]["n_layers"],
        n_hidden_units=binary_gap_params["neural_network"]["n_hidden_units"],
        lr=binary_gap_params["neural_network"]["lr"],
    )
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

    METHODS = {
        "continuous_rate": learn_continuous_rate,
        "binary_rate": learn_binary_rate,
        "continuous_gap": learn_continuous_gap,
        "binary_gap": learn_binary_gap,
    }

    for key in config_keys:
        if key in METHODS:
            method = METHODS[key]
            method(
                train_dataset,
                validation_dataset,
                test_covariate_dataset,
                test_dataset,
                config_hparam[key],
                savepath,
            )
        else:
            continue


if __name__ == "__main__":
    _parser = argh.ArghParser()
    _parser.add_commands([main])
    _parser.dispatch()
