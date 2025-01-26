from gan.gan_data_gen import (
    load_data_for_wgan,
    train_wgan,
)
from hparam.data_generators import (
    generate_synthetic_data,
)
import argh
import dill


@argh.arg("--dropout", default=0.1)
@argh.arg("--batchsize", default=512)
@argh.arg("--maxepochs", default=2000)
@argh.arg("--lr", default=1e-3)
@argh.arg("--trainpath", default="data/train.parquet")
@argh.arg("--summarypath", default="data/summary_2019.parquet")
@argh.arg("--savedir", default="pickled")
@argh.arg("--device", default="cuda")
@argh.arg("--trial", default=0)
def train(
    trainpath="data/train.parquet",
    summarypath="data/summary_2019.parquet",
    savedir="models",
    device="cuda",
    maxepochs=2000,
    lr=1e-3,
    batchsize=512,
    dropout=0.1,
    trial=0,
):
    data, data_wrapper, categorical_mapping = load_data_for_wgan(trainpath, summarypath)
    generator = train_wgan(
        data,
        data_wrapper,
        device,
        max_epochs=maxepochs,
        lr=lr,
        batch_size=batchsize,
        dropout=dropout,
    )

    objects = {
        "generator": generator,
        "data_wrapper": data_wrapper,
        "categorical_mapping": categorical_mapping,
    }

    with open(
        savedir
        + f"/objects-maxepochs={maxepochs}_lr={lr}_batchsize={batchsize}_dropout={dropout}_trial={trial}.pickle",
        "wb",
    ) as dill_file:
        dill.dump(objects, dill_file)


@argh.arg("--objectspath", default="pickled/objects_12-23-2024.pickle")
@argh.arg("--nsamples", type=int, default=1000)
@argh.arg("--savedir", default="data")
@argh.arg("--seed", type=int, default=534543897)
def generate(
    objectspath="pickled/objects_12-23-2024.pickle",
    nsamples=20000,
    savedir="data",
    seed=534543897,
):
    with open(objectspath, "rb") as dill_file:
        objects = dill.load(dill_file)
        generator = objects["generator"]
        data_wrapper = objects["data_wrapper"]
        categorical_mapping = objects["categorical_mapping"]

    name = objectspath.split("objects-")[1].split(".pickle")[0]

    synthetic_df = generate_synthetic_data(
        generator, data_wrapper, categorical_mapping, nsamples, seed=seed
    )
    synthetic_df.to_parquet(
        savedir + "/synthetic-{}_n={}.parquet".format(name, nsamples), index=False
    )


if __name__ == "__main__":
    _parser = argh.ArghParser()
    _parser.add_commands([train, generate])
    _parser.dispatch()
