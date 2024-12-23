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

python main.py train --device cuda --save pickled --trainpath data/train.parquet
python main.py generate --timestamp 12-23-2024 --nsamples 20000 --savedir data --modeldir pickled