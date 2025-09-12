import itertools
import glob
import os


GPU_SBATCH_PREFACE = """#!/bin/bash
#SBATCH -J train-gpu
#SBATCH -p gpu
#SBATCH -c 1
#SBATCH --mem 20GB
#SBATCH -N 1
#SBATCH -t 12:00:00           # limit of 1 day runtime
#SBATCH -G 1              # limit of 2 GPU's per user
#SBATCH -o train-gpu-%j.out
#SBATCH --job-name="{}.sh"
#SBATCH --error="{}/{}_err.log"
#SBATCH --output="{}/{}_out.log"\n
"""


SBATCH_PREFACE = """#!/bin/bash
#SBATCH -t 3:00:00             # limit of 1 day runtime
#SBATCH -c 1
#SBATCH --mem 5GB
#SBATCH -p normal
#SBATCH --exclude=yen15
#SBATCH --ntasks-per-node=1
#SBATCH --job-name="{}.sh"
#SBATCH --error="{}/{}_err.log"
#SBATCH --output="{}/{}_out.log"\n
"""

OUTPUT_PATH = (
    "/home/users/rsahoo/zfs/gsb/intermediate-yens/rsahoo/poverty/dry_run/scripts3"
)


def generate_learn_run_2017_povertyline():
    countries = ["nigeria", "india", "colombia"]
    geo_extrapolation = [True]
    configs = [
        # "output_gt_continuous_gap.yaml",
        # "output_gt_binary_rate.yaml",
        # "output_gt_binary_gap.yaml",
        "output_gt_continuous_gap.yaml",
        # "output_gt_modern_pmt.yaml",
        # "oracle_gap.yaml",
        # "output_gt_pmt.yaml",
        # "ubi.yaml",
    ]

    for country in countries:
        for geo in geo_extrapolation:
            if geo:
                subfolder = "geo_extrapolation"
            else:
                subfolder = "geo_interpolation"
            for config in configs:
                exp_id = country + "_" + subfolder + "_" + "_2017_" + config
                script_fn = os.path.join(OUTPUT_PATH, "{}.sh".format(exp_id))
                with open(script_fn, "w") as f:
                    print(
                        SBATCH_PREFACE.format(
                            exp_id, OUTPUT_PATH, exp_id, OUTPUT_PATH, exp_id
                        ),
                        file=f,
                    )
                    base_cmd = f"python main_learn.py main --config hparam/results/{country}/{subfolder}/{config} --trainpath data/{country}/train.parquet --testpath data/{country}/test.parquet --summarypath data/{country}/summary.parquet --device cpu --country {country} --povertyline 2.15 --year 2017"
                    print(base_cmd, file=f)
                    print("sleep 1", file=f)


def generate_learn_run_2021_povertyline():
    countries = ["ethiopia"]
    geo_extrapolation = [True]
    configs = [
        "output_gt_continuous_rate.yaml",
        # "output_gt_binary_rate.yaml",
        # "output_gt_binary_gap.yaml",
        # "output_gt_continuous_gap.yaml",
        # "output_gt_modern_pmt.yaml",
        # "oracle_gap.yaml",
        # "output_gt_pmt.yaml",
        # "ubi.yaml",
    ]

    for country in countries:
        for geo in geo_extrapolation:
            if geo:
                subfolder = "geo_extrapolation"
            else:
                subfolder = "geo_interpolation"
            for config in configs:
                exp_id = country + "_" + subfolder + "_" + "_2021_" + config
                script_fn = os.path.join(OUTPUT_PATH, "{}.sh".format(exp_id))
                with open(script_fn, "w") as f:
                    print(
                        SBATCH_PREFACE.format(
                            exp_id, OUTPUT_PATH, exp_id, OUTPUT_PATH, exp_id
                        ),
                        file=f,
                    )
                    base_cmd = f"python main_learn.py main --config hparam/results/{country}/{subfolder}/{config} --trainpath data/{country}/train.parquet --testpath data/{country}/test.parquet --summarypath data/{country}/summary.parquet --device cpu --country {country} --povertyline 3.0 --year 2021"
                    print(base_cmd, file=f)
                    print("sleep 1", file=f)


def generate_hparam_run():

    countries = ["india", "colombia"]
    geo_extrapolation = [True]
    configs = [
        "gt_continuous_rate.yaml",
        "gt_binary_rate.yaml",
        "gt_binary_gap.yaml",
        "gt_continuous_gap.yaml",
        "gt_modern_pmt.yaml",
        "gt_pmt.yaml",
    ]

    # script_fn = os.path.join(OUTPUT_PATH, "a_make_hparamdir.sh")
    # with open(script_fn, "w") as f:
    #     print(
    #         SBATCH_PREFACE.format(
    #             "a_make_hparamdir",
    #             OUTPUT_PATH,
    #             "a_make_hparamdir",
    #             OUTPUT_PATH,
    #             "a_make_hparamdir",
    #         ),
    #         file=f,
    #     )
    #     for country in countries:
    #         print(f"mkdir hparam/results/{country}", file=f)
    #         print("sleep 1", file=f)

    for country in countries:
        for geo in geo_extrapolation:
            if geo:
                subfolder = "geo_extrapolation"
            else:
                subfolder = "geo_interpolation"
            for config in configs:
                exp_id = "{}_{}_{}".format(country, subfolder, config)
                script_fn = os.path.join(OUTPUT_PATH, "{}.sh".format(exp_id))
                with open(script_fn, "w") as f:
                    print(
                        GPU_SBATCH_PREFACE.format(
                            exp_id, OUTPUT_PATH, exp_id, OUTPUT_PATH, exp_id
                        ),
                        file=f,
                    )
                    base_cmd = f"python main_hparam.py main --config hparam/configs/{country}/{subfolder}/{config} --learnsavedir learn/results/{country}/{subfolder}"
                    print(base_cmd, file=f)
                    print("sleep 1", file=f)

    script_fn = os.path.join(OUTPUT_PATH, "make_learnsavedir.sh")
    with open(script_fn, "w") as f:
        print(
            SBATCH_PREFACE.format(
                "make_learnsavedir",
                OUTPUT_PATH,
                "make_learnsavedir",
                OUTPUT_PATH,
                "make_learnsavedir",
            ),
            file=f,
        )
        if not os.path.exists(f"learn/results"):
            print(f"mkdir learn/results", file=f)
        for country in countries:
            if not os.path.exists(f"learn/results/{country}"):
                print(f"mkdir learn/results/{country}", file=f)
                print("sleep 1", file=f)
            for geo in geo_extrapolation:
                if geo:
                    subfolder = "geo_extrapolation"
                else:
                    subfolder = "geo_interpolation"
                if not os.path.exists(f"learn/results/{country}/{subfolder}"):
                    print(f"mkdir learn/results/{country}/{subfolder}", file=f)
                    print("sleep 1", file=f)
                    for year in [2017, 2021]:
                        if not os.path.exists(
                            f"learn/results/{country}/{subfolder}/year={year}"
                        ):
                            print(
                                f"mkdir learn/results/{country}/{subfolder}/year={year}",
                                file=f,
                            )
                            print("sleep 1", file=f)


def make_learnsavedir():
    countries = [
        "benin",
        "burkina_faso",
        "cote_divoire",
        "ghana",
        "guinea_bissau",
        "kenya",
        "malawi",
        "mali",
        "niger",
        "nigeria",
        "senegal",
        "south_africa",
        "tanzania",
        "togo",
        "uganda",
    ]
    geo_extrapolation = [True]
    years = [2017]

    script_fn = os.path.join(OUTPUT_PATH, "make_learnsavedir.sh")
    with open(script_fn, "w") as f:
        print(
            SBATCH_PREFACE.format(
                "make_learnsavedir",
                OUTPUT_PATH,
                "make_learnsavedir",
                OUTPUT_PATH,
                "make_learnsavedir",
            ),
            file=f,
        )
        if not os.path.exists(f"learn/results"):
            print(f"mkdir learn/results", file=f)
        for country in countries:
            if not os.path.exists(f"learn/results/{country}"):
                print(f"mkdir learn/results/{country}", file=f)
                print("sleep 1", file=f)
            for geo in geo_extrapolation:
                if geo:
                    subfolder = "geo_extrapolation"
                else:
                    subfolder = "geo_interpolation"
                if not os.path.exists(f"learn/results/{country}/{subfolder}"):
                    print(f"mkdir learn/results/{country}/{subfolder}", file=f)
                    print("sleep 1", file=f)
                for year in years:
                    if not os.path.exists(
                        f"learn/results/{country}/{subfolder}/year={year}"
                    ):
                        print(
                            f"mkdir learn/results/{country}/{subfolder}/year={year}",
                            file=f,
                        )
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
        base_cmd = "python main_gan.py train --device cpu --savedir gan/pickled --trainpath data/preprocessed_2025_02_11/train.parquet --summarypath data/preprocessed_2025_02_11/summary_2019.parquet --maxepochs {} --lr {} --batchsize {} --dropout {}".format(
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
                            base_cmd = "python main.py train --device cuda --savedir pickled --trainpath data/preprocessed_2025_02_11/train.parquet --summarypath data/preprocessed_2025_02_11/summary_2019.parquet --maxepochs {} --lr {} --batchsize {} --dropout {} --trial {}".format(
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


def generate_gt_run():
    hparam_configs = [
        "gt_continuous_rate.yaml",
        "gt_binary_rate.yaml",
        "gt_binary_gap.yaml",
        "gt_continuous_gap.yaml",
    ]

    learn_configs = [
        "output_gt_continuous_rate.yaml",
        "output_gt_binary_rate.yaml",
        "output_gt_binary_gap.yaml",
        "output_gt_continuous_gap.yaml",
    ]

    script_fn = os.path.join("gt_script.sh")
    with open(script_fn, "w") as f:
        print("#!/bin/bash", file=f)
        print("# Hyperparameter search", file=f)
        for config in hparam_configs:
            base_cmd = f"python main_hparam.py main --config hparam/configs/{config}"
            print(base_cmd, file=f)
            print("sleep 1", file=f)

        print("# Learning", file=f)
        print("mkdir learn/results", file=f)
        for config in learn_configs:
            base_cmd = f"python main_learn.py main --config hparam/results/{config} --trainpath data/preprocessed_2025_02_11/train.parquet --testpath data/preprocessed_2025_02_11/test.parquet --device cuda"
            print(base_cmd, file=f)
            print("sleep 1", file=f)


# generate_gt_run()
generate_hparam_run()
# generate_learn_run_2017_povertyline()
# generate_learn_run_2021_povertyline()
# generate_learn_run()
# make_learnsavedir()
# generate_learn_run()
# generate_wgan_run()
# generate_wgan_hparam_runs()
