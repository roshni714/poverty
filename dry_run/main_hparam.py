import yaml
import argh
from hparam.data_generators import (
    get_wgan_data_generator,
    get_gt_train_data_generator,
)
from hparam.density_estimation_hparam_search import (
    get_optimal_density_estimation_parameters,
)
from hparam.knapsack_hparam_search import get_optimal_knapsack_parameters
from hparam.n_regressors_hparam_search import get_optimal_n_regressors
from hparam.nn_hparam_search import (
    get_optimal_nn_quantile_regression_parameters,
    get_optimal_nn_improvement_parameters,
    get_optimal_nn_welfare_parameters,
    get_optimal_nn_pmt_parameters,
    get_optimal_lasso_parameters,
)


@argh.arg("--config", default="hparam/configs/hparam_config.yml")
@argh.arg("--learnsavedir", default="learn/results")
def main(config="hparam_config.yaml", learnsavedir="learn/results"):
    """
    Main function to optimize hyperparameters.

    Args:
        hparamconfig (str): Path to the hyperparameter configuration file. This file contains all hyperparameter ranges that to be optimized and what data to use for the hyperparameter search.
    """
    with open(config) as stream:
        try:
            config_hparams = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    savedir = config_hparams["savedir"]
    device = config_hparams["device"]
    geo_extrapolation = config_hparams["data"]["geo_extrapolation"]

    opt_hparams = {}
    opt_hparams["savedir"] = learnsavedir

    data_config_params = config_hparams["data"]
    opt_hparams["data"] = {}
    opt_hparams["data"]["outcome"] = data_config_params["outcome"]
    opt_hparams["data"]["weight"] = data_config_params["weight"]
    opt_hparams["data"]["geo_extrapolation"] = geo_extrapolation

    if "gt" in data_config_params:
        gt_config_params = data_config_params["gt"]
        trainpath = gt_config_params["trainpath"]
        summarypath = gt_config_params["summarypath"]

        train_data_generator, ntrain, val_df, original_cols = (
            get_gt_train_data_generator(
                trainpath,
                summarypath=summarypath,
                auxpath=gt_config_params["auxpath"],
                outcome=data_config_params["outcome"],
                year=data_config_params["year"],
                geo_extrapolation=geo_extrapolation,
                val_split=0.33,
            )
        )

    outcome = data_config_params["outcome"]
    weight = data_config_params["weight"]
    name = config.split("/")[-1].split(".yaml")[0]
    print(config_hparams)

    if "continuous_rate" in config_hparams:
        opt_hparams["continuous_rate"] = {}
        rate = config_hparams["continuous_rate"]
        if "density_estimation" in rate:
            opt_density_estimation_hparams = get_optimal_density_estimation_parameters(
                density_estimation_hparam_ranges=rate["density_estimation"],
                data_generator=train_data_generator,
                device=device,
                original_cols=original_cols,
                ntrain=ntrain,
                val_df=val_df,
                outcome=outcome,
                weight=weight,
                savepath=f"{savedir}/density_estimation_{name}.csv",
            )
            opt_hparams["continuous_rate"][
                "density_estimation"
            ] = opt_density_estimation_hparams

            opt_n_alpha = get_optimal_knapsack_parameters(
                rate["n_alpha"],
                povertyline=data_config_params["povertyline"],
                data_generator=train_data_generator,
                device=device,
                original_cols=original_cols,
                ntrain=ntrain,
                val_df=val_df,
                outcome=outcome,
                weight=weight,
                density_estimation_params=opt_density_estimation_hparams,
                savepath=f"{savedir}/n_alpha_{name}.csv",
            )

            opt_hparams["continuous_rate"]["n_alpha"] = opt_n_alpha
    if "binary_rate" in config_hparams:
        opt_hparams["binary_rate"] = {}
        binary_rate = config_hparams["binary_rate"]
        opt_nn_hparams = get_optimal_nn_improvement_parameters(
            loss_type="binary_rate",
            povertyline=data_config_params["povertyline"],
            nn_hparam_ranges=binary_rate["neural_network"],
            data_generator=train_data_generator,
            device=device,
            original_cols=original_cols,
            ntrain=ntrain,
            val_df=val_df,
            outcome=outcome,
            weight=weight,
            savepath=f"{savedir}/nn_{name}.csv",
        )
        opt_hparams["binary_rate"]["neural_network"] = opt_nn_hparams
        print(opt_nn_hparams)

        opt_n_regressors = get_optimal_n_regressors(
            binary_rate["n_regressors"],
            loss_type="binary_rate",
            povertyline=data_config_params["povertyline"],
            data_generator=train_data_generator,
            device=device,
            original_cols=original_cols,
            ntrain=ntrain,
            val_df=val_df,
            outcome=outcome,
            weight=weight,
            neural_network_params=opt_nn_hparams,
            savepath=f"{savedir}/n_regressors_{name}.csv",
        )
        opt_hparams["binary_rate"]["n_regressors"] = opt_n_regressors
    if "continuous_gap" in config_hparams:
        opt_hparams["continuous_gap"] = {}
        continuous_gap = config_hparams["continuous_gap"]
        opt_nn_hparams = get_optimal_nn_quantile_regression_parameters(
            nn_hparam_ranges=continuous_gap["neural_network"],
            data_generator=train_data_generator,
            device=device,
            original_cols=original_cols,
            ntrain=ntrain,
            val_df=val_df,
            outcome=outcome,
            weight=weight,
            savepath=f"{savedir}/nn_{name}.csv",
        )
        print(opt_nn_hparams)

        opt_hparams["continuous_gap"]["neural_network"] = opt_nn_hparams

        opt_n_regressors = get_optimal_n_regressors(
            continuous_gap["n_regressors"],
            loss_type="continuous_gap",
            povertyline=data_config_params["povertyline"],
            data_generator=train_data_generator,
            device=device,
            original_cols=original_cols,
            ntrain=ntrain,
            val_df=val_df,
            outcome=outcome,
            weight=weight,
            neural_network_params=opt_nn_hparams,
            savepath=f"{savedir}/n_regressors_{name}.csv",
        )
        opt_hparams["continuous_gap"]["n_regressors"] = opt_n_regressors
    if "binary_gap" in config_hparams:
        opt_hparams["binary_gap"] = {}
        binary_gap = config_hparams["binary_gap"]
        opt_nn_hparams = get_optimal_nn_improvement_parameters(
            loss_type="binary_gap",
            povertyline=data_config_params["povertyline"],
            nn_hparam_ranges=binary_gap["neural_network"],
            data_generator=train_data_generator,
            device=device,
            original_cols=original_cols,
            ntrain=ntrain,
            val_df=val_df,
            outcome=outcome,
            weight=weight,
            savepath=f"{savedir}/nn_{name}.csv",
        )
        print(opt_nn_hparams)
        opt_hparams["binary_gap"]["neural_network"] = opt_nn_hparams
        opt_n_regressors = get_optimal_n_regressors(
            binary_gap["n_regressors"],
            loss_type="binary_gap",
            povertyline=data_config_params["povertyline"],
            data_generator=train_data_generator,
            device=device,
            original_cols=original_cols,
            ntrain=ntrain,
            val_df=val_df,
            outcome=outcome,
            weight=weight,
            neural_network_params=opt_nn_hparams,
            savepath=f"{savedir}/n_regressors_{name}.csv",
        )
        opt_hparams["binary_gap"]["n_regressors"] = opt_n_regressors

    if "modern_pmt" in config_hparams:
        opt_hparams["modern_pmt"] = {}
        modern_pmt = config_hparams["modern_pmt"]
        opt_nn_hparams = get_optimal_nn_pmt_parameters(
            nn_hparam_ranges=modern_pmt["neural_network"],
            data_generator=train_data_generator,
            device=device,
            original_cols=original_cols,
            ntrain=ntrain,
            val_df=val_df,
            outcome=outcome,
            weight=weight,
            savepath=f"{savedir}/nn_{name}.csv",
        )
        print(opt_nn_hparams)
        opt_hparams["modern_pmt"]["neural_network"] = opt_nn_hparams
        opt_hparams["modern_pmt"]["transfer_value"] = data_config_params["povertyline"]
    
    if "welfare" in config_hparams:
        opt_hparams["welfare"] = {}
        welfare = config_hparams["welfare"]
        opt_nn_hparams = get_optimal_nn_welfare_parameters(
            loss_type="welfare",
            povertyline=data_config_params["povertyline"],
            nn_hparam_ranges=welfare["neural_network"],
            data_generator=train_data_generator,
            device=device,
            original_cols=original_cols,
            ntrain=ntrain,
            val_df=val_df,
            outcome=outcome,
            weight=weight,
            savepath=f"{savedir}/nn_{name}.csv",
        )
        print(opt_nn_hparams)
        opt_hparams["welfare"]["neural_network"] = opt_nn_hparams

        opt_n_regressors = get_optimal_n_regressors(
            welfare["n_regressors"],
            loss_type="welfare",
            povertyline=data_config_params["povertyline"],
            data_generator=train_data_generator,
            device=device,
            original_cols=original_cols,
            ntrain=ntrain,
            val_df=val_df,
            outcome=outcome,
            weight=weight,
            neural_network_params=opt_nn_hparams,
            savepath=f"{savedir}/n_regressors_{name}.csv",
        )
        opt_hparams["welfare"]["n_regressors"] = opt_n_regressors

    if "pmt" in config_hparams:
        opt_hparams["pmt"] = {}
        pmt = config_hparams["pmt"]
        opt_lasso_hparams = get_optimal_lasso_parameters(
            lasso_hparam_ranges=pmt["lasso"],
            data_generator=train_data_generator,
            device=device,
            original_cols=original_cols,
            ntrain=ntrain,
            val_df=val_df,
            outcome=outcome,
            weight=weight,
            savepath=f"{savedir}/lasso_{name}.csv",
        )
        print(opt_lasso_hparams)
        opt_hparams["pmt"]["lasso"] = opt_lasso_hparams
        opt_hparams["pmt"]["transfer_value"] = data_config_params["povertyline"]

    with open(f"{savedir}/output_{name}.yaml", "w") as file:
        yaml.dump(opt_hparams, file, default_flow_style=False)


if __name__ == "__main__":
    _parser = argh.ArghParser()
    _parser.add_commands([main])
    _parser.dispatch()
