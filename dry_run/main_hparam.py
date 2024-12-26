import yaml
import argh
from data_generators import get_wgan_data_generator, get_gt_data_generator
from density_estimation_hparam_search import get_optimal_density_estimation_parameters
from feature_selection_hparam_search import get_optimal_num_features


@argh.arg("--hparamconfig", default="hparam_config.yml")
def main(hparamconfig="hparam_config.yaml", defaultconfig="default_config.yml"):
    """
    Main function to optimize hyperparameters.

    Args:
        hparamconfig (str): Path to the hyperparameter configuration file. This file contains all hyperparameter ranges that to be optimized and what data to use for the hyperparameter search.
        defaultconfig (str): Path to the default configuration file. This file contains default hyperparameter settings that are used when the hyperparameter configuration file does not specify a hyperparameter range.
    """
    with open(hparamconfig) as stream:
        try:
            config_hparams = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    with open(defaultconfig) as stream:
        try:
            default_params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

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

    if "rate" in config_hparams:
        opt_hparams["rate"] = {}
        rate = config_hparams["rate"]

        if "density_estimation" in rate:
            opt_density_estimation_hparams = get_optimal_density_estimation_parameters(
                density_estimation_hparam_ranges=rate["density_estimation"],
                data_generator=data_generator,
            )
            opt_hparams["rate"]["density_estimation"] = opt_density_estimation_hparams


if __name__ == "__main__":
    _parser = argh.ArghParser()
    _parser.add_commands([main])
    _parser.dispatch()
