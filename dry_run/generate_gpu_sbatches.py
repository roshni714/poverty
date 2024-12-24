import itertools
import glob
import os


SBATCH_PREFACE = """#!/bin/bash
#SBATCH -J train-gpu
#SBATCH -p gpu
#SBATCH -c 20
#SBATCH -N 1
#SBATCH -t 1-             # limit of 1 day runtime
#SBATCH -G 1              # limit of 2 GPU's per user
#SBATCH -o train-gpu-%j.out
#SBATCH --job-name="{}.sh"
#SBATCH --error="{}/{}_err.log"
#SBATCH --output="{}/{}_out.log"\n
"""

OUTPUT_PATH = (
    "/home/users/rsahoo/zfs/gsb/intermediate-yens/rsahoo/poverty/dry_run/scripts"
)


def generate_wgan_runs():

    batchsize_range = [256]
    maxepochs_range = [2000, 3000, 4000, 5000]
    lr_range = [1e-3]
    dropout_range = [0.0, 0.1, 0.2, 0.3]
    for batchsize in batchsize_range:
        for maxepochs in maxepochs_range:
            for lr in lr_range:
                for dropout in dropout_range:
                    exp_id = "wgan_maxepochs={}_lr={}_batchsize={}_dropout={}".format(
                        maxepochs, lr, batchsize, dropout
                    )
                    script_fn = os.path.join(OUTPUT_PATH, "{}.sh".format(exp_id))
                    with open(script_fn, "w") as f:
                        print(
                            SBATCH_PREFACE.format(
                                exp_id, OUTPUT_PATH, exp_id, OUTPUT_PATH, exp_id
                            ),
                            file=f,
                        )
                        base_cmd = "python main.py train --device cuda --savedir pickled --trainpath data/train.parquet --maxepochs {} --lr {} --batchsize {} --dropout {}".format(
                            maxepochs, lr, batchsize, dropout
                        )
                        print(base_cmd, file=f)

                        base_cmd = "python main.py generate --generatorpath pickled/generator-maxepochs={maxepochs}_lr={lr}_batchsize={batchsize}_dropout={dropout}.pickle --datawrapperpath pickled/datawrapper-maxepochs={maxepochs}_lr={lr}_batchsize={batchsize}_dropout={dropout}.pickle --nsamples 20000 --savedir data".format(
                            maxepochs=maxepochs,
                            lr=lr,
                            batchsize=batchsize,
                            dropout=dropout,
                        )
                        print(base_cmd, file=f)

                        print("sleep 1", file=f)


generate_wgan_runs()
