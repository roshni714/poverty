# Malawi Dry Run
This module implements the Malawi dry-run of the data use plan.

# Datasets
The datasets for reproducing the results from the Malawi dry-run are available in the Google Drive folder [here](https://drive.google.com/drive/folders/1t0s0ef7UZGCAzgWFKzBWEZ032BuP5LWP?usp=sharing).

# Installation
Here are the installation instructions for the dry-run.

```
python3 -m venv dryrun
source dryrun/bin/activate
pip install git+https://github.com/gsbDBI/ds-wgan
cd poverty/package
pip install -e .
```
Before running the final command, there are a few extra packages that need to be installed; they are listed [here](https://github.com/roshni714/poverty/blob/master/dry_run/requirements.txt) are installed. At a later point, we'll need to make a single installation script.

# GAN Training
Running the following command will train a WGAN on 50% of the data stored in `trainpath` and save pickled outputs (generator, data wrapper) to the folder `savedir`. These objects are saved so that we can generate synthetic data without retraining.

```
mkdir pickled
python main.py train --device cpu --savedir pickled --trainpath data/train.parquet --maxepochs 3000 --lr 1e-3 --batchsize 256 --dropout 0.1
```

# Synthetic Data Generation
Running the following command will unpickle the generator and datawrapper saved at `generatorpath` and `datawrapperpath`, respectively and generate a synthetic dataset with `nsamples` and save it to `savedir`.

```
python main.py generate --generatorpath pickled/generator-maxepochs=3000_lr=0.001_batchsize=256_dropout=0.1.pickle --datawrapperpath pickled/datawrapper-maxepochs=1000_lr=0.001_batchsize=256_dropout=0.1.pickle --nsamples 20000 --savedir data
```

# Hyperparameter Search
Running the following command will run a hyperparameter search for all procedures and hyperparameters specified in the `hparamconfig` YAML file. [This]((https://github.com/roshni714/poverty/blob/21ccd2be215338756a8ce1015082382a6eaac924/dry_run/configs/hparam_config.yml)) is an example config file. If a certain hyperparameter is not specified in the config file, the script uses a default value; see default hyperparameter values [here](https://github.com/roshni714/poverty/blob/21ccd2be215338756a8ce1015082382a6eaac924/dry_run/configs/default_config.yml). This script will produce a YAML file with the optimal hyperparameter values.

```
python main_hp.py main --hparamconfig hparam_config.yml
```


