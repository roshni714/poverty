import yaml
import argh
from hparam.data_generators import (
    get_wgan_data_generator,
    get_gt_data_generator,
)
from hparam.density_estimation_hparam_search import (
    get_optimal_density_estimation_parameters,
)
from hparam.knapsack_hparam_search import get_optimal_knapsack_parameters
from hparam.n_regressors_hparam_search import get_optimal_n_regressors
from hparam.nn_hparam_search import (
    get_optimal_nn_quantile_regression_parameters,
    get_optimal_nn_improvement_parameters,
)


@argh.arg("--config", default="hparam/configs/hparam_config.yml")
def main(config="hparam_config.yaml"):
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

    opt_hparams = {}
    opt_hparams["savedir"] = "learn/results"

    data_config_params = config_hparams["data"]
    opt_hparams["data"] = {}
    opt_hparams["data"]["outcome"] = data_config_params["outcome"]
    opt_hparams["data"]["weight"] = data_config_params["weight"]
    if "gan" in data_config_params:
        gan_config_params = data_config_params["gan"]
        objectspath = gan_config_params["objectspath"]
        summarypath = gan_config_params["summarypath"]
        data_generator, original_cols = get_wgan_data_generator(
            objectspath, summarypath=summarypath
        )

    elif "gt" in data_config_params:
        gt_config_params = data_config_params["gt"]
        trainpath = gt_config_params["trainpath"]
        summarypath = gt_config_params["summarypath"]
        data_generator, original_cols = get_gt_data_generator(
            trainpath, summarypath=summarypath
        )

    ntrain = data_config_params["ntrain"]
    nval = data_config_params["nval"]
    ntest = data_config_params["ntest"]
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
                data_generator=data_generator,
                original_cols=original_cols,
                ntrain=ntrain,
                nval=nval,
                outcome=outcome,
                weight=weight,
                savepath=f"{savedir}/density_estimation_{name}.csv",
            )
            opt_hparams["continuous_rate"][
                "density_estimation"
            ] = opt_density_estimation_hparams

            opt_n_alpha = get_optimal_knapsack_parameters(
                rate["n_alpha"],
                data_generator=data_generator,
                original_cols=original_cols,
                ntrain=ntrain,
                ntest=ntest,
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
            nn_hparam_ranges=binary_rate["neural_network"],
            data_generator=data_generator,
            original_cols=original_cols,
            ntrain=ntrain,
            nval=nval,
            outcome=outcome,
            weight=weight,
            savepath=f"{savedir}/nn_{name}.csv",
        )
        opt_hparams["binary_rate"]["neural_network"] = opt_nn_hparams
        print(opt_nn_hparams)

        opt_n_regressors = get_optimal_n_regressors(
            binary_rate["n_regressors"],
            loss_type="binary_rate",
            data_generator=data_generator,
            original_cols=original_cols,
            ntrain=ntrain,
            ntest=ntest,
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
            data_generator=data_generator,
            original_cols=original_cols,
            ntrain=ntrain,
            nval=nval,
            outcome=outcome,
            weight=weight,
            savepath=f"{savedir}/nn_{name}.csv",
        )
        print(opt_nn_hparams)

        opt_hparams["continuous_gap"]["neural_network"] = opt_nn_hparams

        opt_n_regressors = get_optimal_n_regressors(
            continuous_gap["n_regressors"],
            loss_type="continuous_gap",
            data_generator=data_generator,
            original_cols=original_cols,
            ntrain=ntrain,
            ntest=ntest,
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
            nn_hparam_ranges=binary_gap["neural_network"],
            data_generator=data_generator,
            original_cols=original_cols,
            ntrain=ntrain,
            nval=nval,
            outcome=outcome,
            weight=weight,
            savepath=f"{savedir}/nn_{name}.csv",
        )
        print(opt_nn_hparams)
        opt_hparams["binary_gap"]["neural_network"] = opt_nn_hparams
        opt_n_regressors = get_optimal_n_regressors(
            binary_gap["n_regressors"],
            loss_type="binary_gap",
            data_generator=data_generator,
            original_cols=original_cols,
            ntrain=ntrain,
            ntest=ntest,
            outcome=outcome,
            weight=weight,
            neural_network_params=opt_nn_hparams,
            savepath=f"{savedir}/n_regressors_{name}.csv",
        )
        opt_hparams["binary_gap"]["n_regressors"] = opt_n_regressors

    with open(f"{savedir}/output_{name}.yaml", "w") as file:
        yaml.dump(opt_hparams, file, default_flow_style=False)


if __name__ == "__main__":
    _parser = argh.ArghParser()
    _parser.add_commands([main])
    _parser.dispatch()
