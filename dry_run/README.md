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
pip install -r requirements.txt
pip install -e .
```
May need to check that the additional packages [here](https://github.com/roshni714/poverty/blob/master/dry_run/requirements.txt) are installed. At a later point, we'll need to make a single installation script.

# GAN Training
Running the following command will train a WGAN on 50% of the data stored in `trainpath` and save pickled outputs (generator, data wrapper) to the folder `savedir`. These objects are saved so that we can generate synthetic data without retraining.

```
mkdir pickled
python main.py train --device cuda --savedir pickled --trainpath data/train.parquet --maxepochs 1000 --lr 1e-3 --batchsize 256 --dropout 0.2
```

# Synthetic Data Generation
Running the following command will unpickle the generator and datawrapper saved at `generatorpath` and `datawrapperpath`, respectively and generate a synthetic dataset with `nsamples` and save it to `savedir`.

```
python main.py generate --generatorpath pickled/generator-maxepochs=1000_lr=0.001_batchsize=256_dropout=0.2.pickle --datawrapperpath pickled/datawrapper-maxepochs=1000_lr=0.001_batchsize=256_dropout=0.2.pickle --nsamples 20000 --savedir data
```


