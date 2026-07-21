import yaml
import argh
import numpy as np
from feature_selection import forward_selection
from opt_targeted_transfers import (
    RateTargetedTransfers,
    GapTargetedTransfers,
    BinaryRateTargetedTransfers,
    BinaryGapTargetedTransfers,
    UBITargetedTransfers,
    OracleGapTargetedTransfers,
    OracleRateTargetedTransfers,
    PMTTargetedTransfers,
    ModernPMTTargetedTransfers,
    WelfareTargetedTransfers,
    PMTGapTargetedTransfers,
    write_result,
    bootstrap_subsample,
)
from learn.data_loader import load_datasets
import os


def run_evaluation(tt, test_covariate_dataset, test_dataset, budgets, savepath):
    if os.path.exists(savepath + ".csv"):
        os.remove(savepath + ".csv")

    for budget in budgets:
        tt.set_budget(budget)
        if "oracle" in tt.name:
            tt.run_opt(test_dataset)
        else:
            tt.run_opt(test_covariate_dataset)
        res = tt.evaluate(test_dataset)
        print(res)
        write_result(savepath + ".csv", res)
        if os.path.exists(savepath + f"_budget={budget}.csv"):
            os.remove(savepath + f"_budget={budget}.csv")
        tt.evaluate_equity(test_dataset, savepath + f"_budget={budget}.csv")

    auc_res = tt.compute_auc(
        test_dataset=test_dataset,
        test_covariate_dataset=test_covariate_dataset,
        metrics=["post_transfer_poverty_rate", "post_transfer_poverty_gap"],
        budgets=budgets,
    )
    if os.path.exists(savepath + "_auc.csv"):
        os.remove(savepath + "_auc.csv")
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
        max_features=continuous_rate_params["density_estimation"]["n_features"],
    )
    train_dataset.covs = features
    validation_dataset.covs = features
    test_covariate_dataset.covs = features
    test_dataset.covs = features

    tt = RateTargetedTransfers(c_bar=povertyline)
    tt.fit(
        train_dataset,
        validation_dataset,
        n_knots=int(continuous_rate_params["density_estimation"]["n_knots"]),
        n_bins=int(continuous_rate_params["density_estimation"]["n_bins"]),
        degree=int(continuous_rate_params["density_estimation"]["degree"]),
        device=device,
    )
    all_res = []
    if os.path.exists(savepath + ".csv"):
        os.remove(savepath + ".csv")
    for budget in budgets:
        tt.set_budget(budget)
        tt.run_opt(test_covariate_dataset, n_alpha=continuous_rate_params["n_alpha"])
        res = tt.evaluate(test_dataset)
        write_result(savepath + ".csv", res)
        if os.path.exists(savepath + f"_budget={budget}.csv"):
            os.remove(savepath + f"_budget={budget}.csv")
        tt.evaluate_equity(test_dataset, savepath + f"_budget={budget}.csv")
        all_res.append(res)

    if os.path.exists(savepath + "_auc.csv"):
        os.remove(savepath + "_auc.csv")

    auc_res = {}
    metrics = ["post_transfer_poverty_rate", "post_transfer_poverty_gap"]
    for metric in metrics:
        y_items = []
        for res in all_res:
            y_items.append(res[metric])
        auc = np.trapz(y_items, x=budgets)
        auc_res[metric] = auc

    write_result(savepath + "_auc.csv", auc_res)


def learn_binary_rate(
    train_dataset,
    validation_dataset,
    test_covariate_dataset,
    test_dataset,
    binary_rate_params,
    povertyline,
    budgets,
    device,
    savepath,
):
    """
    Learn the binary rate targeted transfers
    """
    print("Learning binary rate targeted transfers...")
    tt = BinaryRateTargetedTransfers(
        c_bar=povertyline, n_regressors=binary_rate_params["n_regressors"]
    )
    tt.fit(
        train_dataset,
        validation_dataset,
        device=device,
        **binary_rate_params["neural_network"],
    )
    tt.get_opt_transfer_sizes_given_budget_grid(validation_dataset, budgets)
    run_evaluation(tt, test_covariate_dataset, test_dataset, budgets, savepath)


def learn_continuous_gap(
    train_dataset,
    validation_dataset,
    test_covariate_dataset,
    test_dataset,
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


def learn_binary_gap(
    train_dataset,
    validation_dataset,
    test_covariate_dataset,
    test_dataset,
    binary_gap_params,
    povertyline,
    budgets,
    device,
    savepath,
):
    """
    Learn the binary gap targeted transfers
    """
    print("Learning binary gap targeted transfers...")
    tt = BinaryGapTargetedTransfers(
        c_bar=povertyline, n_regressors=binary_gap_params["n_regressors"]
    )
    tt.fit(
        train_dataset,
        validation_dataset,
        device=device,
        **binary_gap_params["neural_network"],
    )
    tt.get_opt_transfer_sizes_given_budget_grid(validation_dataset, budgets)
    run_evaluation(tt, test_covariate_dataset, test_dataset, budgets, savepath)


def learn_modern_pmt(
    train_dataset,
    validation_dataset,
    test_covariate_dataset,
    test_dataset,
    modern_pmt_params,
    povertyline,
    budgets,
    device,
    savepath,
):
    """
    Learn the modern PMT targeted transfers
    """
    _, y_train, r_train = train_dataset.get_data()
    weight = r_train[y_train <= povertyline]
    z = (y_train[y_train <= povertyline] * weight).sum() / weight.sum() * 0.2

    print("Learning modern PMT targeted transfers...")
    tt = ModernPMTTargetedTransfers(c_bar=povertyline, transfer_value=z)
    tt.fit(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        device=device,
        **modern_pmt_params["neural_network"],
    )
    run_evaluation(tt, test_covariate_dataset, test_dataset, budgets, savepath)


def learn_pmt(
    train_dataset,
    validation_dataset,
    test_covariate_dataset,
    test_dataset,
    pmt_params,
    povertyline,
    budgets,
    device,
    savepath,
):
    """
    Learn PMT targeted transfers
    """
    print("Learning PMT targeted transfers...")
    _, y_train, r_train = train_dataset.get_data()
    weight = r_train[y_train <= povertyline]
    z = (y_train[y_train <= povertyline] * weight).sum() / weight.sum() * 0.2
    tt = PMTTargetedTransfers(c_bar=povertyline, transfer_value=z)
    tt.fit(
        train_dataset,
        validation_dataset,
        alpha=pmt_params["lasso"]["alpha"],
    )
    run_evaluation(tt, test_covariate_dataset, test_dataset, budgets, savepath)


def learn_pmt_gap(
    train_dataset,
    validation_dataset,
    test_covariate_dataset,
    test_dataset,
    pmt_params,
    povertyline,
    budgets,
    device,
    savepath,
):
    """
    Learn PMT targeted transfers
    """
    print("Learning PMT targeted transfers...")
    tt = PMTGapTargetedTransfers(c_bar=povertyline)
    tt.fit(
        train_dataset,
        validation_dataset,
        alpha=pmt_params["lasso"]["alpha"],
    )
    tt.get_opt_transfer_sizes_given_budget_grid(validation_dataset, budgets)
    run_evaluation(tt, test_covariate_dataset, test_dataset, budgets, savepath)


def learn_ubi(test_covariate_dataset, test_dataset, povertyline, budgets, savepath):
    """
    Learn UBI targeted transfers
    """
    print("Learning UBI targeted transfers...")
    tt = UBITargetedTransfers(c_bar=povertyline)
    run_evaluation(tt, test_covariate_dataset, test_dataset, budgets, savepath)


def learn_oracle_gap(
    test_covariate_dataset,
    test_dataset,
    povertyline,
    budgets,
    savepath,
):
    """
    Learn the oracle gap targeted transfers
    """
    print("Learning oracle gap targeted transfers...")
    tt = OracleGapTargetedTransfers(c_bar=povertyline, scheme="consumption_floor")
    run_evaluation(tt, test_covariate_dataset, test_dataset, budgets, savepath)

def learn_oracle_rate(
    test_covariate_dataset,
    test_dataset,
    povertyline,
    budgets,
    savepath,
):
    """
    Learn the weakly equitable oracle rate targeted transfers
    """
    print("Learning weakly equitable oracle rate targeted transfers...")
    tt = OracleRateTargetedTransfers(c_bar=povertyline, scheme="rate_weakly_equitable")
    run_evaluation(tt, test_covariate_dataset, test_dataset, budgets, savepath)


def learn_welfare(
    train_dataset,
    validation_dataset,
    test_covariate_dataset,
    test_dataset,
    welfare_params,
    povertyline,
    budgets,
    device,
    savepath,
):
    """
    Learn the welfare targeted transfers
    """
    print("Learning welfare targeted transfers...")
    tt = WelfareTargetedTransfers(c_bar=povertyline, budget=2)
    tt.fit(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        device=device,
        **welfare_params["neural_network"],
    )
    run_evaluation(tt, test_covariate_dataset, test_dataset, budgets, savepath)


@argh.arg("--config", default="hparam_results/output_gan_continuous_rate.yaml")
@argh.arg("--povertyline", default=3.0)
@argh.arg("--year", default=2021)
@argh.arg("--geo", action="store_true", help="Whether to use geo only features")
@argh.arg("--nfeatures", default=None, type=int)
@argh.arg("--trainpath", default=None)
@argh.arg("--testpath", default=None)
@argh.arg("--auxpath", default="data/auxiliary_data/auxiliary_data_20260511.csv")
@argh.arg("--summarypath", default="data/summary_2019.parquet")
@argh.arg("--device", default="cpu")
def main(
    config="hparam_results/output_gan_continuous_rate.yaml",
    povertyline=3.0,
    year=2021,
    geo=False,
    nfeatures=None,
    auxpath="data/auxiliary_data/auxiliary_data_20260511.csv",
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
                "welfare",
                "data",
                "savedir",
                "pmt",
                "pmt_gap",
                "ubi",
                "modern_pmt",
                "oracle_rate",
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
            auxpath,
            geo_only=geo,
            outcome=data_config["outcome"],
            weight=data_config["weight"],
            year=year,
        )
    )

    if nfeatures is not None:
        features, _ = forward_selection(
            train_dataset,
            validation_dataset,
            max_features=nfeatures,
        )
        train_dataset.covs = features
        validation_dataset.covs = features
        test_covariate_dataset.covs = features
        test_dataset.covs = features

    name = config.split("/")[-1].split(".yaml")[0]

    if nfeatures is not None:
        savepath = (
            savedir + "/" + "year=" + str(year) + "_d=" + str(nfeatures) + "/" + name
        )
    elif geo:
        savepath = savedir + "/" + "year=" + str(year) + "_geo_only" + "/" + name
    else:
        savepath = savedir + "/" + "year=" + str(year) + "/" + name

    LEARNING_METHODS = {
        "continuous_rate": learn_continuous_rate,
        "binary_rate": learn_binary_rate,
        "continuous_gap": learn_continuous_gap,
        "binary_gap": learn_binary_gap,
        "pmt": learn_pmt,
        "modern_pmt": learn_modern_pmt,
        "welfare": learn_welfare,
        "pmt_gap": learn_pmt_gap,
    }

    NONLEARNING_METHODS = {"oracle_gap": learn_oracle_gap, "ubi": learn_ubi, "oracle_rate": learn_oracle_rate}

    _, y_test, r_test = test_dataset.get_data()
    pov_gap = np.sum(np.maximum(povertyline - y_test, 0) * r_test)
    oracle_budgets = np.linspace(0.0, pov_gap, 15)
    budgets = np.linspace(0.05, povertyline, 15)

    _, y_train, r_train = train_dataset.get_data()
    weight = r_train[y_train <= povertyline]
    z = (y_train[y_train <= povertyline] * weight).sum() / weight.sum() * 0.2
    pmt_budgets = np.linspace(0.05, z, 15)

    for key in config_keys:
        if key in LEARNING_METHODS:
            learn_budgets = budgets
            if key in ["pmt", "modern_pmt"]:
                learn_budgets = pmt_budgets
            method = LEARNING_METHODS[key]
            method(
                train_dataset,
                validation_dataset,
                test_covariate_dataset,
                test_dataset,
                config_hparam[key],
                povertyline=povertyline,
                budgets=learn_budgets,
                device=device,
                savepath=savepath,
            )
        elif key in NONLEARNING_METHODS:
            if "oracle" in key:
                learn_budgets = oracle_budgets
            else:
                learn_budgets = budgets
            method = NONLEARNING_METHODS[key]
            method(
                test_covariate_dataset,
                test_dataset,
                povertyline=povertyline,
                budgets=learn_budgets,
                savepath=savepath,
            )
        else:
            continue


@argh.arg("--config", default="hparam_results/output_gan_continuous_rate.yaml")
@argh.arg("--povertyline", default=3.0)
@argh.arg("--year", default=2021)
@argh.arg("--trainfraction", default=1.0, type=float)
@argh.arg("--seed", default=42, type=int)
@argh.arg("--trainpath", default=None)
@argh.arg("--testpath", default=None)
@argh.arg("--auxpath", default="data/auxiliary_data/auxiliary_data_20260511.csv")
@argh.arg("--summarypath", default="data/summary_2019.parquet")
@argh.arg("--device", default="cpu")
def main_sample_size(
    config="hparam_results/output_gan_continuous_rate.yaml",
    povertyline=3.0,
    year=2021,
    geo=False,
    trainfraction=1.0,
    auxpath="data/auxiliary_data/auxiliary_data_20251207.csv",
    trainpath=None,
    testpath=None,
    summarypath=None,
    device="cpu",
    seed=42,
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
                "welfare",
                "data",
                "savedir",
                "pmt",
                "pmt_gap",
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
            auxpath,
            geo_only=False,
            outcome=data_config["outcome"],
            weight=data_config["weight"],
            year=year,
        )
    )
    name = config.split("/")[-1].split(".yaml")[0]
    savepath = (
        savedir
        + "/"
        + "year="
        + str(year)
        + "_sample_size"
        + "/"
        + name
        + "_nprop="
        + str(trainfraction)
        + "_seed="
        + str(seed)
    )
    train_dataset = bootstrap_subsample(train_dataset, frac=trainfraction, seed=seed)

    LEARNING_METHODS = {
        "continuous_rate": learn_continuous_rate,
        "binary_rate": learn_binary_rate,
        "continuous_gap": learn_continuous_gap,
        "binary_gap": learn_binary_gap,
        "pmt": learn_pmt,
        "modern_pmt": learn_modern_pmt,
        "welfare": learn_welfare,
        "pmt_gap": learn_pmt_gap,
    }

    NONLEARNING_METHODS = {"oracle_gap": learn_oracle_gap, "ubi": learn_ubi}

    _, y_test, r_test = test_dataset.get_data()
    pov_gap = np.sum(np.maximum(povertyline - y_test, 0) * r_test)
    oracle_budgets = np.linspace(0.0, pov_gap, 15)
    budgets = np.linspace(0.05, povertyline, 15)

    for key in config_keys:
        if key in LEARNING_METHODS:

            method = LEARNING_METHODS[key]
            method(
                train_dataset,
                validation_dataset,
                test_covariate_dataset,
                test_dataset,
                config_hparam[key],
                povertyline=povertyline,
                budgets=budgets,
                device=device,
                savepath=savepath,
            )
        elif key in NONLEARNING_METHODS:
            if "oracle" in key:
                learn_budgets = oracle_budgets
            else:
                learn_budgets = budgets
            method = NONLEARNING_METHODS[key]
            method(
                test_covariate_dataset,
                test_dataset,
                povertyline=povertyline,
                budgets=learn_budgets,
                savepath=savepath,
            )
        else:
            continue


if __name__ == "__main__":
    _parser = argh.ArghParser()
    _parser.add_commands([main, main_sample_size])
    _parser.dispatch()
