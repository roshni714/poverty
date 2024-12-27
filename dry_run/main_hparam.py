import yaml
import argh
from data_generators import get_wgan_data_generator, get_gt_data_generator
from density_estimation_hparam_search import get_optimal_density_estimation_parameters
import os

# from nn_hparam_search import get_optimal_nn_quantile_regression_parameters, get_optimal_nn_gap_improvement_parameters


@argh.arg("--hparamconfig", default="configs/hparam_config.yml")
def main(hparamconfig="hparam_config.yaml"):
    """
    Main function to optimize hyperparameters.

    Args:
        hparamconfig (str): Path to the hyperparameter configuration file. This file contains all hyperparameter ranges that to be optimized and what data to use for the hyperparameter search.
    """
    with open(hparamconfig) as stream:
        try:
            config_hparams = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    savedir = config_hparams["savedir"]
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    opt_hparams = {}

    data_config_params = config_hparams["data"]
    if "gan" in data_config_params:
        gan_config_params = data_config_params["gan"]
        generatorpath = gan_config_params["generatorpath"]
        datawrapperpath = gan_config_params["datawrapperpath"]
        data_generator = get_wgan_data_generator(generatorpath, datawrapperpath)

    elif "gt" in data_config_params:
        gt_config_params = data_config_params["gt"]
        trainpath = gt_config_params["trainpath"]
        data_generator = get_gt_data_generator(trainpath)

    ntrain = data_config_params["ntrain"]
    nval = data_config_params["nval"]
    outcome = data_config_params["outcome"]
    weight = data_config_params["weight"]

    if "rate" in config_hparams:
        opt_hparams["rate"] = {}
        rate = config_hparams["rate"]
        if "density_estimation" in rate:
            opt_density_estimation_hparams = get_optimal_density_estimation_parameters(
                density_estimation_hparam_ranges=rate["density_estimation"],
                data_generator=data_generator,
                ntrain=ntrain,
                nval=nval,
                outcome=outcome,
                weight=weight,
                savedir=savedir,
            )
            opt_hparams["rate"]["density_estimation"] = opt_density_estimation_hparams
            print(opt_density_estimation_hparams)
    # if "continuous_gap" in config_hparams:
    #     opt_hparams["continuous_gap"] = {}
    #     continuous_gap = config_hparams["continuous_gap"]
    #     opt_nn_hparams = get_optimal_nn_quantile_regression_parameters(nn_hparam_ranges=continuous_gap["neural_network"],
    #                                                                    data_generator=data_generator,
    #                                                                    ntrain=ntrain,
    #                                                                    nval=nval,
    #                                                                    outcome=outcome,
    #                                                                    weight=weight,
    #                                                                    savedir=savedir)
    #     opt_hparams["continuous_gap"]["neural_network"] = opt_nn_hparams
    #     print(opt_nn_hparams)
    # if "binary_gap" in config_hparams:
    #     opt_hparams["binary_gap"] = {}
    #     binary_gap = config_hparams["binary_gap"]
    #     opt_nn_hparams = get_optimal_nn_gap_improvement_parameters(nn_hparam_ranges=binary_gap["neural_network"],
    #                                                                    data_generator=data_generator,
    #                                                                    ntrain=ntrain,
    #                                                                    nval=nval,
    #                                                                    outcome=outcome,
    #                                                                    weight=weight,
    #                                                                    savedir=savedir)
    #     opt_hparams["binary_gap"]["neural_network"] = opt_nn_hparams
    #     print(opt_nn_hparams)

    with open("output.yaml", "w") as file:
        yaml.dump(opt_hparams, file, default_flow_style=False)


if __name__ == "__main__":
    _parser = argh.ArghParser()
    _parser.add_commands([main])
    _parser.dispatch()
