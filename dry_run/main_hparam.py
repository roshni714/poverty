import yaml
import argh
from data_generators import get_wgan_data_generator, get_gt_data_generator


@argh.arg("--config", default="config.yml")
def main(config="config.yaml"):
    with open(config) as stream:
        try:
            config_params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    data_config_params = config_params["data"]
    if "gan" in data_config_params:
        gan_config_params = data_config_params["gan"]
        generatorpath = gan_config_params["generator_path"]
        datawrapperpath = gan_config_params["datawrapper_path"]
        data_generator = get_wgan_data_generator(generatorpath, datawrapperpath)

    elif "gt" in data_config_params:
        gt_config_params = data_config_params["gt"]
        trainpath = gt_config_params["trainpath"]
        data_generator = get_gt_data_generator(trainpath)


if __name__ == "__main__":
    _parser = argh.ArghParser()
    _parser.add_commands([main])
    _parser.dispatch()
