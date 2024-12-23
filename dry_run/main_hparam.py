from gan_data_gen import generate_synthetic_data
import yaml
import argh
import dill


@argh.arg("--config", default="config.yaml")
@argh.arg("--timestamp", default="12-23-2024")
def main(config="config.yaml", timestamp="12-23-2024"):
    with open("example.yaml") as stream:
        try:
            print(yaml.safe_load(stream))
        except yaml.YAMLError as exc:
            print(exc)

    with open("pickled/generator_{}.pickle".format(timestamp), "rb") as dill_file:
        generator = dill.load(dill_file)

    with open("pickled/data_wrapper_{}.pickle".format(timestamp), "rb") as dill_file:
        data_wrapper = dill.load(dill_file)
