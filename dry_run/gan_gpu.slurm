#!/bin/bash

# Example slurm script to train pytorch DL model on Yen GPU

#SBATCH -J train-gpu
#SBATCH -p gpu
#SBATCH -c 20
#SBATCH -N 1
#SBATCH -t 1-             # limit of 1 day runtime
#SBATCH -G 1              # limit of 2 GPU's per user
#SBATCH -o train-gpu-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rsahoo@stanford.edu

python main.py train --device cuda --savedir pickled --trainpath data/train.parquet --maxepochs 2000 --lr 1e-4 --batchsize 512
python main.py generate --generatorpath pickled/generator-maxepochs=2000_lr=0.0001_batchsize=512.pickle --datawrapperpath pickled/datawrapper-maxepochs=2000_lr=0.0001_batchsize=512.pickle --nsamples 20000 --savedir data