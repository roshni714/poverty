import yaml
import argh
import numpy as np
from feature_selection import forward_selection
from opt_targeted_transfers import (
    RateTargetedTransfers,
    GapTargetedTransfers,
    write_result,
)
from learn.data_loader import load_datasets
import os
import pandas as pd


def run_evaluation(tt, test_covariate_dataset, test_dataset, budgets, savepath):
    results = []
    for budget in budgets:
        tt.set_budget(budget)
        tt.run_opt(test_covariate_dataset)
        res = tt.evaluate(test_dataset)
        results.append(res)
    df = pd.DataFrame(results)
    ys = [results[-1]["initial_poverty_rate"]] + list(df["post_transfer_poverty_rate"])
    xs = [0.0] + list(df["policy_cost_per_capita"])
    auc = np.trapz(ys, x=xs)
    write_result(
        savepath + "_comparison" + ".csv",
        {
            "AUC": auc,
            "d": results[-1]["d"],
            "initial_poverty_rate": results[-1]["initial_poverty_rate"],
        },
    )


def learn_continuous_rate(
    train_dataset,
    validation_dataset,
    test_covariate_dataset,
    test_dataset,
    covariate_dimensions,
    continuous_rate_params,
    povertyline,
    budgets,
    device,
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
        max_features=max(covariate_dimensions),
    )

    if os.path.exists(savepath + "_comparison" + ".csv"):
        os.remove(savepath + "_comparison" + ".csv")

    for cov_dim in covariate_dimensions:
        train_dataset.covs = features[:cov_dim]
        validation_dataset.covs = features[:cov_dim]
        test_covariate_dataset.covs = features[:cov_dim]
        test_dataset.covs = features[:cov_dim]

        tt = RateTargetedTransfers(c_bar=povertyline)
        tt.fit(
            train_dataset,
            validation_dataset,
            n_knots=int(continuous_rate_params["density_estimation"]["n_knots"]),
            n_bins=int(continuous_rate_params["density_estimation"]["n_bins"]),
            degree=int(continuous_rate_params["density_estimation"]["degree"]),
            kde_fft=continuous_rate_params["density_estimation"].get("kde_fft", False),
            winsorize=continuous_rate_params["density_estimation"].get(
                "winsorize", False
            ),
            device=device,
        )
        results = []
        for budget in budgets:
            tt.set_budget(budget)
            tt.run_opt(
                test_covariate_dataset, n_alpha=continuous_rate_params["n_alpha"]
            )
            res = tt.evaluate(test_dataset)
            results.append(res)
        df = pd.DataFrame(results)
        ys = [results[-1]["initial_poverty_rate"]] + list(
            df["post_transfer_poverty_rate"]
        )
        xs = [0.0] + list(df["policy_cost_per_capita"])
        auc = np.trapz(ys, x=xs)
        write_result(
            savepath + "_comparison" + ".csv",
            {
                "AUC": auc,
                "d": results[-1]["d"],
                "initial_poverty_rate": results[-1]["initial_poverty_rate"],
            },
        )


def learn_continuous_gap(
    train_dataset,
    validation_dataset,
    test_covariate_dataset,
    test_dataset,
    covariate_dimensions,
    continuous_gap_params,
    povertyline,
    budgets,
    device,
    savepath,
):
    """
    Learn the continuous gap targeted transfers
    """
    print("Learning continuous gap targeted transfers...")

    print("Run forward selection...")
    features, _ = forward_selection(
        train_dataset,
        validation_dataset,
        max_features=max(covariate_dimensions),
    )

    if os.path.exists(savepath + "_comparison" + ".csv"):
        os.remove(savepath + "_comparison" + ".csv")

    for cov_dim in covariate_dimensions:
        train_dataset.covs = features[:cov_dim]
        validation_dataset.covs = features[:cov_dim]
        test_covariate_dataset.covs = features[:cov_dim]
        test_dataset.covs = features[:cov_dim]
        tt = GapTargetedTransfers(
            c_bar=povertyline, n_regressors=continuous_gap_params["n_regressors"]
        )
        tt.fit(
            train_dataset,
            validation_dataset,
            device=device,
            **continuous_gap_params["neural_network"],
        )
        run_evaluation(tt, test_covariate_dataset, test_dataset, budgets, savepath)


@argh.arg("--config", default="hparam_results/output_gan_continuous_rate.yaml")
@argh.arg("--povertyline", default=3.0)
@argh.arg("--year", default=2021)
@argh.arg("--country", default="malawi")
@argh.arg("--trainpath", default=None)
@argh.arg("--testpath", default=None)
@argh.arg("--summarypath", default="data/summary_2019.parquet")
@argh.arg("--device", default="cpu")
def main(
    config="hparam_results/output_gan_continuous_rate.yaml",
    povertyline=3.0,
    year=2021,
    country="malawi",
    trainpath=None,
    testpath=None,
    summarypath=None,
    device="cpu",
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
                "oracle_gap",
                "data",
                "savedir",
                "pmt",
                "ubi",
                "modern_pmt",
            ]
            for key in config_keys
        ]
    )

    data_config = config_hparam["data"]
    savedir = config_hparam["savedir"]

    train_dataset, validation_dataset, test_covariate_dataset, test_dataset = (
        load_datasets(
            trainpath,
            testpath,
            summarypath,
            geo_extrapolation=data_config["geo_extrapolation"],
            outcome=data_config["outcome"],
            weight=data_config["weight"],
            country=country,
            year=year,
        )
    )

    name = config.split("/")[-1].split(".yaml")[0]
    savepath = savedir + "/" + "year=" + str(year) + "/" + name

    LEARNING_METHODS = {
        "continuous_rate": learn_continuous_rate,
        "continuous_gap": learn_continuous_gap,
    }

    budgets = np.linspace(0.05, 2.15, 15)
    COVARIATE_DIMENSIONS = np.linspace(1, len(train_dataset.covs), 6, dtype=int)

    for key in config_keys:
        if key in LEARNING_METHODS:
            method = LEARNING_METHODS[key]
            method(
                train_dataset,
                validation_dataset,
                test_covariate_dataset,
                test_dataset,
                COVARIATE_DIMENSIONS,
                config_hparam[key],
                povertyline=povertyline,
                budgets=budgets,
                device=device,
                savepath=savepath,
            )
        else:
            continue


if __name__ == "__main__":
    _parser = argh.ArghParser()
    _parser.add_commands([main])
    _parser.dispatch()
