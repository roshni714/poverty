from gan_data_gen import load_data_for_wgan, train_wgan, generate_synthetic_data
import argh
import dill
from datetime import datetime
import numpy as np


@argh.arg("--trainpath", default="data/train.parquet")
@argh.arg("--savedir", default="pickled")
@argh.arg("--device", default="cuda")
def train(trainpath="data/train.parquet", savedir="models", device="cuda"):
    data, data_wrapper = load_data_for_wgan(trainpath)
    generator = train_wgan(data, data_wrapper, device)

    with open(
        savedir + "/generator_{}.pickle".format(datetime.now().strftime("%m-%d-%Y")),
        "wb",
    ) as dill_file:
        dill.dump(generator, dill_file)

    with open(
        savedir + "/data_wrapper_{}.pickle".format(datetime.now().strftime("%m-%d-%Y")),
        "wb",
    ) as dill_file:
        dill.dump(data_wrapper, dill_file)


@argh.arg("--timestamp", default=datetime.now().strftime("%m-%d-%Y"))
@argh.arg("--nsamples", type=int, default=1000)
@argh.arg("--savedir", default="data")
@argh.arg("--seed", type=int, default=534543897)
def generate(
    timestamp="12-23-2024",
    nsamples=20000,
    savedir="data",
    modeldir="pickled",
    seed=534543897,
):
    with open(modeldir + "/generator_{}.pickle".format(timestamp), "rb") as dill_file:
        generator = dill.load(dill_file)

    with open(
        modeldir + "/data_wrapper_{}.pickle".format(timestamp), "rb"
    ) as dill_file:
        data_wrapper = dill.load(dill_file)

    synthetic_df = generate_synthetic_data(generator, data_wrapper, nsamples, seed=seed)
    synthetic_df.to_csv(
        savedir + "/synthetic_{}_n={}.parquet".format(timestamp, nsamples), index=False
    )


if __name__ == "__main__":
    _parser = argh.ArghParser()
    _parser.add_commands([train, generate])
    _parser.dispatch()
