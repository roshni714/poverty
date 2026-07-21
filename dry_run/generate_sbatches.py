import itertools
import glob
import os
import argparse


GPU_SBATCH_PREFACE = """#!/bin/bash
#SBATCH -J train-gpu
#SBATCH -p gpu
#SBATCH -c 1
#SBATCH --mem 15GB
#SBATCH -N 1
#SBATCH -t 24:00:00           # limit of 1 day runtime
#SBATCH -G 1              # limit of 2 GPU's per user
#SBATCH -o train-gpu-%j.out
#SBATCH --job-name="{}.sh"
#SBATCH --error="{}/{}_err.log"
#SBATCH --output="{}/{}_out.log"\n
"""


HPARAM_NORMAL_SBATCH_PREFACE = """#!/bin/bash
#SBATCH -t 24:00:00             # limit of 1 day runtime
#SBATCH -c 1
#SBATCH --mem 10GB
#SBATCH -p normal
#SBATCH --exclude=yen15
#SBATCH --ntasks-per-node=1
#SBATCH --job-name="{}.sh"
#SBATCH --error="{}/{}_err.log"
#SBATCH --output="{}/{}_out.log"\n
"""

SBATCH_PREFACE = """#!/bin/bash
#SBATCH -t 6:00:00             # limit of 1 day runtime
#SBATCH -c 1
#SBATCH --mem 10GB
#SBATCH -p normal
#SBATCH --exclude=yen15
#SBATCH --ntasks-per-node=1
#SBATCH --job-name="{}.sh"
#SBATCH --error="{}/{}_err.log"
#SBATCH --output="{}/{}_out.log"\n
"""

CONTRATE_LONG_SBATCH_PREFACE = """#!/bin/bash
#SBATCH -t 96:00:00             # limit of 1 day runtime
#SBATCH -c 1
#SBATCH --mem 30GB
#SBATCH -p long
#SBATCH --exclude=yen15
#SBATCH --ntasks-per-node=1
#SBATCH --job-name="{}.sh"
#SBATCH --error="{}/{}_err.log"
#SBATCH --output="{}/{}_out.log"\n
"""

SBATCH_ARRAY_PREFACE = """#!/bin/bash
#SBATCH -t 6:00:00             # limit of 1 day runtime
#SBATCH -c 1
#SBATCH --mem 10GB
#SBATCH -p normal
#SBATCH --exclude=yen15
#SBATCH --ntasks-per-node=1
#SBATCH --job-name="{}.sh"
#SBATCH --error="{}/{}_err.log"
#SBATCH --output="{}/{}_out.log"
#SBATCH --array=0-14\n
"""


COUNTRIES = [
    "BDI",
    "BEN",
    "BFA",
    "BGD",
    "CAF",
    "CIV",
    "COD",
    "COL",
    "ETH",
    "GHA",
    "GNB",
    "IDN",
    "IND",
    "KEN",
    "LBR",
    "MDG",
    "MEX",
    "MLI",
    "MWI",
    "NAM",
    "NER",
    "NGA",
    "PAK",
    "RWA",
    "SDN",
    "SLE",
    "SEN",
    "TGO",
    "TLS",
    "TZA",
    "UGA",
    "YEM",
    "ZAF",
    "ZWE",
]


def generate_learn_run_2017_restricted_features_povertyline(output_path):
    geo_extrapolation = [True]
    configs = ["output_gt_continuous_gap.yaml"]

    for country in COUNTRIES:
        for geo in geo_extrapolation:
            if geo:
                subfolder = "geo_extrapolation"
            else:
                subfolder = "geo_interpolation"
            for config in configs:
                exp_id = (
                    country
                    + "_learn"
                    + "_"
                    + subfolder
                    + "_"
                    + "_2017_"
                    + config
                    + "_restricted_features"
                )
                script_fn = os.path.join(output_path, "{}.sh".format(exp_id))
                with open(script_fn, "w") as f:
                    print(
                        SBATCH_PREFACE.format(
                            exp_id, output_path, exp_id, output_path, exp_id
                        ),
                        file=f,
                    )
                    base_cmd = f"python main_learn.py main --config hparam/results/{country}/{subfolder}/{config} --trainpath data/{country}/train.parquet --testpath data/{country}/test.parquet --summarypath data/{country}/summary.parquet --device cpu --povertyline 2.15 --year 2017 --nfeatures 20"
                    print(base_cmd, file=f)
                    print("sleep 1", file=f)


def generate_satellite_learn_run(output_path):

    countries = ["TGO_alpha_earth", "TGO_alpha_earth_and_survey"]
    configs = ["output_gt_continuous_gap.yaml"]

    for country in countries:
        for config in configs:
            exp_id = (
                country
                + "_learn_satellite"
                + "_geo_extrapolation"
                + "_"
                + "_2017_"
                + config
            )
            script_fn = os.path.join(output_path, "{}.sh".format(exp_id))
            with open(script_fn, "w") as f:
                print(
                    SBATCH_PREFACE.format(
                        exp_id, output_path, exp_id, output_path, exp_id
                    ),
                    file=f,
                )
                base_cmd = f"python main_learn.py main --config hparam/results/{country}/geo_extrapolation/{config} --trainpath data/{country}/train.parquet --testpath data/{country}/test.parquet --summarypath data/{country}/summary.parquet --device cpu --povertyline 2.15 --year 2017"
                print(base_cmd, file=f)
                print("sleep 1", file=f)


def generate_satellite_hparam_run(output_path):
    countries = ["TGO_alpha_earth_and_survey", "TGO_alpha_earth"]
    configs = ["gt_continuous_gap.yaml"]

    for country in countries:
        for config in configs:
            exp_id = (
                country + "_hparam" + "_geo_extrapolation" + "_" + "_2017_" + config
            )
            script_fn = os.path.join(output_path, "{}.sh".format(exp_id))
            with open(script_fn, "w") as f:
                print(
                    SBATCH_PREFACE.format(
                        exp_id, output_path, exp_id, output_path, exp_id
                    ),
                    file=f,
                )
                base_cmd = f"python main_hparam.py main --config hparam/configs/{country}/geo_extrapolation/{config} --learnsavedir learn/results/{country}/geo_extrapolation"
                print(base_cmd, file=f)
                print("sleep 1", file=f)


def generate_geographic_learn_run(output_path):
    configs = ["output_gt_continuous_gap.yaml"]

    for country in COUNTRIES:
        for config in configs:
            exp_id = country + "_learn_geo" + "_" + "_2017_" + config
            script_fn = os.path.join(output_path, "{}.sh".format(exp_id))
            with open(script_fn, "w") as f:
                print(
                    SBATCH_PREFACE.format(
                        exp_id, output_path, exp_id, output_path, exp_id
                    ),
                    file=f,
                )
                base_cmd = f"python main_learn.py main --geo --config hparam/results/{country}/geo_extrapolation/{config} --trainpath data/{country}/train.parquet --testpath data/{country}/test.parquet --summarypath data/{country}/summary.parquet --device cpu --povertyline 2.15 --year 2017"
                print(base_cmd, file=f)
                print("sleep 1", file=f)


def generate_sample_size_run_2017_povertyline(output_path):
    geo_extrapolation = [True]
    train_fractions = [0.05, 0.1, 0.2, 0.5, 0.7, 0.9, 1.0]

    for country in COUNTRIES:
        for geo in geo_extrapolation:
            if geo:
                subfolder = "geo_extrapolation"
            else:
                subfolder = "geo_interpolation"
            config = "output_gt_continuous_gap.yaml"
            for frac in train_fractions:
                exp_id = (
                    country
                    + "_learn_"
                    + "sample_size_"
                    + str(frac)
                    + "_"
                    + subfolder
                    + "_"
                    + "_2017_"
                    + config
                )
                script_fn = os.path.join(output_path, "{}.sh".format(exp_id))
                with open(script_fn, "w") as f:
                    print(
                        SBATCH_ARRAY_PREFACE.format(
                            exp_id, output_path, exp_id, output_path, exp_id
                        ),
                        file=f,
                    )
                    base_cmd = f"python main_learn.py main-sample-size --config hparam/results/{country}/{subfolder}/{config} --trainpath data/{country}/train.parquet --testpath data/{country}/test.parquet --summarypath data/{country}/summary.parquet --device cpu --povertyline 2.15 --year 2017 --trainfraction {frac} --seed $SLURM_ARRAY_TASK_ID"
                    print(base_cmd, file=f)
                    print("sleep 1", file=f)


def generate_learn_run_2017_povertyline(output_path):
    geo_extrapolation = [True]
    configs = [
        #"output_gt_continuous_gap.yaml",
        #"output_gt_binary_rate.yaml",
        #"output_gt_binary_gap.yaml",
        #"output_gt_continuous_rate.yaml",
        #"output_gt_pmt.yaml",
        #"output_gt_modern_pmt.yaml",
        "oracle_rate.yaml",
        #"output_gt_pmt_gap.yaml",
        #"ubi.yaml",
        #"output_gt_welfare.yaml",
    ]

    for country in COUNTRIES:
        for geo in geo_extrapolation:
            if geo:
                subfolder = "geo_extrapolation"
            else:
                subfolder = "geo_interpolation"
            for config in configs:
                exp_id = country + "_learn_" + subfolder + "_" + "_2017_" + config
                script_fn = os.path.join(output_path, "{}.sh".format(exp_id))
                with open(script_fn, "w") as f:
                    print(
                        SBATCH_PREFACE.format(
                            exp_id, output_path, exp_id, output_path, exp_id
                        ),
                        file=f,
                    )
                    base_cmd = f"python main_learn.py main --config hparam/results/{country}/{subfolder}/{config} --trainpath data/{country}/train.parquet --testpath data/{country}/test.parquet --summarypath data/{country}/summary.parquet --device cpu --povertyline 2.15 --year 2017"
                    print(base_cmd, file=f)
                    print("sleep 1", file=f)


def generate_learn_run_2021_povertyline(output_path):
    geo_extrapolation = [True]
    configs = [
        # "output_gt_pmt.yaml",
        # "output_gt_pmt_gap.yaml",
        # "output_gt_continuous_rate.yaml",
        # "output_gt_binary_rate.yaml",
        # "output_gt_binary_gap.yaml",
        # "output_gt_continuous_gap.yaml",
        # "output_gt_modern_pmt.yaml",
        # "ubi.yaml",
        "oracle_rate.yaml",
    ]

    for country in COUNTRIES:
        for geo in geo_extrapolation:
            if geo:
                subfolder = "geo_extrapolation"
            else:
                subfolder = "geo_interpolation"
            for config in configs:
                exp_id = country + "_learn_" + subfolder + "_" + "_2021_" + config
                script_fn = os.path.join(output_path, "{}.sh".format(exp_id))
                with open(script_fn, "w") as f:
                    print(
                        SBATCH_PREFACE.format(
                            exp_id, output_path, exp_id, output_path, exp_id
                        ),
                        file=f,
                    )
                    base_cmd = f"python main_learn.py main --config hparam/results/{country}/{subfolder}/{config} --trainpath data/{country}/train.parquet --testpath data/{country}/test.parquet --summarypath data/{country}/summary.parquet --device cpu --povertyline 3.0 --year 2021"
                    print(base_cmd, file=f)
                    print("sleep 1", file=f)


def generate_hparam_run(output_path):
    geo_extrapolation = [True]
    configs = [
        "gt_pmt_gap.yaml",
        "gt_welfare.yaml",
        "gt_pmt.yaml",
        "gt_modern_pmt.yaml",
        "gt_continuous_rate.yaml",
        "gt_binary_rate.yaml",
        "gt_binary_gap.yaml",
        "gt_continuous_gap.yaml",
    ]

    for country in COUNTRIES:
        for geo in geo_extrapolation:
            if geo:
                subfolder = "geo_extrapolation"
            else:
                subfolder = "geo_interpolation"
            for config in configs:
                exp_id = "{}_hparam_{}_{}".format(country, subfolder, config)
                script_fn = os.path.join(output_path, "{}.sh".format(exp_id))
                with open(script_fn, "w") as f:
                    print(
                        HPARAM_NORMAL_SBATCH_PREFACE.format(
                            exp_id, output_path, exp_id, output_path, exp_id
                        ),
                        file=f,
                    )
                    base_cmd = f"python main_hparam.py main --config hparam/configs/{country}/{subfolder}/{config} --learnsavedir learn/results/{country}/{subfolder}"
                    print(base_cmd, file=f)
                    print("sleep 1", file=f)

    script_fn = os.path.join(output_path, "make_learnsavedir.sh")
    with open(script_fn, "w") as f:
        print(
            SBATCH_PREFACE.format(
                "make_learnsavedir",
                output_path,
                "make_learnsavedir",
                output_path,
                "make_learnsavedir",
            ),
            file=f,
        )
        if not os.path.exists(f"learn/results"):
            print(f"mkdir learn/results", file=f)
        for country in COUNTRIES:
            if not os.path.exists(f"{output_path}/learn/results/{country}"):
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

                        if year == 2017:
                            print(
                                f"mkdir learn/results/{country}/{subfolder}/year={year}_d=20",
                                file=f,
                            )
                            print(
                                f"mkdir learn/results/{country}/{subfolder}/year={year}_sample_size",
                                file=f,
                            )
                            print(
                                f"mkdir learn/results/{country}/{subfolder}/year={year}_geo_only",
                                file=f,
                            )

                        print("sleep 1", file=f)


def make_learnsavedir(output_path):
    geo_extrapolation = [True]
    years = [2017, 2021]

    script_fn = os.path.join(output_path, "make_learnsavedir.sh")
    with open(script_fn, "w") as f:
        print(
            SBATCH_PREFACE.format(
                "make_learnsavedir",
                output_path,
                "make_learnsavedir",
                output_path,
                "make_learnsavedir",
            ),
            file=f,
        )
        if not os.path.exists(f"learn/results"):
            print(f"mkdir learn/results", file=f)
        for country in COUNTRIES:
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

                    if year == 2017 and not os.path.exists(
                        f"learn/results/{country}/{subfolder}/year={year}_d=20"
                    ):
                        print(
                            f"mkdir learn/results/{country}/{subfolder}/year={year}_d=20",
                            file=f,
                        )
                        print("sleep 1", file=f)

                    if year == 2017 and not os.path.exists(
                        f"learn/results/{country}/{subfolder}/year={year}_sample_size"
                    ):

                        print(
                            f"mkdir learn/results/{country}/{subfolder}/year={year}_sample_size",
                            file=f,
                        )
                        print("sleep 1", file=f)
                    if year == 2017 and not os.path.exists(
                        f"learn/results/{country}/{subfolder}/year={year}_geo_only"
                    ):
                        print(
                            f"mkdir learn/results/{country}/{subfolder}/year={year}_geo_only",
                            file=f,
                        )
                        print("sleep 1", file=f)


def main():
    parser = argparse.ArgumentParser(description="Generate SBATCH scripts.")
    parser.add_argument(
        "--output-path",
        help="Directory where generated SBATCH scripts will be written.",
    )
    args = parser.parse_args()
    output_path = args.output_path
    hparam_output_path = os.path.join(output_path, "hparam")
    learn_output_path = os.path.join(output_path, "learn")

    os.makedirs(hparam_output_path, exist_ok=True)
    os.makedirs(learn_output_path, exist_ok=True)
    # generate_hparam_run(hparam_output_path)
    #generate_satellite_hparam_run(hparam_output_path)

    generate_learn_run_2017_povertyline(learn_output_path)
    generate_learn_run_2021_povertyline(learn_output_path)
    # generate_learn_run_2017_restricted_features_povertyline(learn_output_path)
    #generate_satellite_learn_run(learn_output_path)
    # generate_sample_size_run_2017_povertyline(learn_output_path)
    # generate_geographic_learn_run(learn_output_path)


if __name__ == "__main__":
    main()
