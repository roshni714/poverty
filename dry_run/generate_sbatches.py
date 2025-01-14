import itertools
import glob
import os


GPU_SBATCH_PREFACE = """#!/bin/bash
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


SBATCH_PREFACE = """#!/bin/bash
#SBATCH -t 12:00:00
#SBATCH -c 1
#SBATCH --mem 10GB
#SBATCH -p normal
#SBATCH --exclude=yen11
#SBATCH --ntasks-per-node=1
#SBATCH --job-name="{}.sh"
#SBATCH --error="{}/{}_err.log"
#SBATCH --output="{}/{}_out.log"\n
"""

OUTPUT_PATH = (
    "/home/users/rsahoo/zfs/gsb/intermediate-yens/rsahoo/poverty/dry_run/scripts"
)


def generate_learn_run():

    configs = [
        "output_gan_continuous_rate.yaml",
        "output_gan_binary_rate.yaml",
        "output_gan_binary_gap.yaml",
        "output_gan_continuous_gap.yaml",
    ]

    for config in configs:
        exp_id = config
        script_fn = os.path.join(OUTPUT_PATH, "{}.sh".format(exp_id))
        with open(script_fn, "w") as f:
            print(
                SBATCH_PREFACE.format(exp_id, OUTPUT_PATH, exp_id, OUTPUT_PATH, exp_id),
                file=f,
            )
            base_cmd = f"python main_learn.py main --config hparam_results/{config} --trainpath data/train.parquet --testpath data/test.parquet --savedir learn_results"
            print(base_cmd, file=f)
            print("sleep 1", file=f)


def generate_hparam_run():

    configs = [
        "gan_continuous_rate.yaml",
        "gt_continuous_rate.yaml",
        "gan_binary_rate.yaml",
        "gt_binary_rate.yaml",
        "gan_binary_gap.yaml",
        "gan_continuous_gap.yaml",
        "gt_binary_gap.yaml",
        "gt_continuous_gap.yaml",
    ]

    for config in configs:
        exp_id = config
        script_fn = os.path.join(OUTPUT_PATH, "{}.sh".format(exp_id))
        with open(script_fn, "w") as f:
            print(
                SBATCH_PREFACE.format(exp_id, OUTPUT_PATH, exp_id, OUTPUT_PATH, exp_id),
                file=f,
            )
            base_cmd = f"python main_hparam.py main --hparamconfig hparam/test_configs/{config}"
            print(base_cmd, file=f)
            print("sleep 1", file=f)


def generate_wgan_run():
    maxepochs = 3000
    lr = 1e-3
    batchsize = 256
    dropout = 0.1

    exp_id = (
        f"wgan_maxepochs={maxepochs}_lr={lr}_batchsize={batchsize}_dropout={dropout}"
    )

    script_fn = os.path.join(OUTPUT_PATH, "{}.sh".format(exp_id))
    with open(script_fn, "w") as f:
        print(
            GPU_SBATCH_PREFACE.format(exp_id, OUTPUT_PATH, exp_id, OUTPUT_PATH, exp_id),
            file=f,
        )
        base_cmd = "python main_gan.py train --device cuda --savedir gan/pickled --trainpath data/train.parquet --maxepochs {} --lr {} --batchsize {} --dropout {}".format(
            maxepochs, lr, batchsize, dropout
        )
        print(base_cmd, file=f)

        base_cmd = "python main_gan.py generate --objectspath gan/pickled/objects-maxepochs={maxepochs}_lr={lr}_batchsize={batchsize}_dropout={dropout}_trial=0.pickle --nsamples 20000 --savedir data".format(
            maxepochs=maxepochs,
            lr=lr,
            batchsize=batchsize,
            dropout=dropout,
        )
        print(base_cmd, file=f)
        print("sleep 1", file=f)


def generate_wgan_hparam_runs():
    batchsize_range = [128, 256, 512]
    maxepochs_range = [2500, 5000, 7500]
    lr_range = [1e-3]
    dropout_range = [0.0]
    n_trials = 5

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
                            GPU_SBATCH_PREFACE.format(
                                exp_id, OUTPUT_PATH, exp_id, OUTPUT_PATH, exp_id
                            ),
                            file=f,
                        )
                        for trial in range(n_trials):
                            base_cmd = "python main.py train --device cuda --savedir pickled --trainpath data/train.parquet --maxepochs {} --lr {} --batchsize {} --dropout {} --trial {}".format(
                                maxepochs, lr, batchsize, dropout, trial
                            )
                            print(base_cmd, file=f)

                            base_cmd = "python main.py generate --objectspath pickled/objects-maxepochs={maxepochs}_lr={lr}_batchsize={batchsize}_dropout={dropout}_trial={trial}.pickle --nsamples 20000 --savedir data".format(
                                maxepochs=maxepochs,
                                lr=lr,
                                batchsize=batchsize,
                                dropout=dropout,
                                trial=trial,
                            )
                            print(base_cmd, file=f)

                            print("sleep 1", file=f)


# generate_learn_run()
generate_hparam_run()
# generate_wgan_run()
# generate_wgan_hparam_runs()
