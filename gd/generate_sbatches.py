import itertools
import glob
import os


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

# constants for commands

OUTPUT_PATH = "/home/users/rsahoo/zfs/gsb/intermediate-yens/rsahoo/poverty/gd/scripts"
SAVE_PATH = "/home/users/rsahoo/zfs/gsb/intermediate-yens/rsahoo/poverty/gd/results/"


def generate_malawi_test():

    district = "chitipa"
    policy = "optimized"
    uncondtol = 0.10
    pools = ["karonga"]

    for pool in pools:
        exp_id = "{}_policy={}_uncondtol={}_pool={}".format(
            district, policy, uncondtol, pool
        )
        script_fn = os.path.join(OUTPUT_PATH, "{}.sh".format(exp_id))
        with open(script_fn, "w") as f:
            print(
                SBATCH_PREFACE.format(exp_id, OUTPUT_PATH, exp_id, OUTPUT_PATH, exp_id),
                file=f,
            )
            base_cmd = "python main.py main --policy {} --district {} --uncondtol {} --pool {} --save {}".format(
                policy, district, uncondtol, pool, SAVE_PATH
            )
            print(base_cmd, file=f)
            print("sleep 1", file=f)


def generate_malawi_northern_district_runs():

    uncondtols = [0.10, 0.05, 0.20]
    districts = ["chitipa", "karonga"]
    policies = ["saturation", "binary", "optimized", "oracle"]
    # policies = ["optimized", "oracle"]
    pools = ["rural"]

    for pool in pools:
        for uncondtol in uncondtols:
            for district in districts:
                for policy in policies:
                    exp_id = "{}_policy={}_uncondtol={}_pool={}".format(
                        district, policy, uncondtol, pool
                    )
                    script_fn = os.path.join(OUTPUT_PATH, "{}.sh".format(exp_id))
                    with open(script_fn, "w") as f:
                        print(
                            SBATCH_PREFACE.format(
                                exp_id, OUTPUT_PATH, exp_id, OUTPUT_PATH, exp_id
                            ),
                            file=f,
                        )
                        base_cmd = "python main.py main --policy {} --district {} --uncondtol {} --pool {} --save {}".format(
                            policy, district, uncondtol, pool, SAVE_PATH
                        )
                        print(base_cmd, file=f)
                        print("sleep 1", file=f)


def generate_malawi_central_district_runs():

    uncondtols = [0.1]
    districts = ["mchinji", "dowa", "kasungu"]
    policies = ["saturation", "binary", "optimized", "oracle"]
    # policies = ["optimized", "oracle"]
    pools = ["central"]

    for pool in pools:
        for uncondtol in uncondtols:
            for district in districts:
                for policy in policies:
                    exp_id = "{}_policy={}_uncondtol={}_pool={}".format(
                        district, policy, uncondtol, pool
                    )
                    script_fn = os.path.join(OUTPUT_PATH, "{}.sh".format(exp_id))
                    with open(script_fn, "w") as f:
                        print(
                            SBATCH_PREFACE.format(
                                exp_id, OUTPUT_PATH, exp_id, OUTPUT_PATH, exp_id
                            ),
                            file=f,
                        )
                        base_cmd = "python main.py main --policy {} --district {} --uncondtol {} --pool {} --save {}".format(
                            policy, district, uncondtol, pool, SAVE_PATH
                        )
                        print(base_cmd, file=f)
                        print("sleep 1", file=f)


def generate_malawi_runs():

    uncondtols = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.55]
    districts = ["all"]
    policies = ["conditional_optimized", "optimized"]
    num_features = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # policies = ["optimized", "oracle"]
    for num_feat in num_features:
        for uncondtol in uncondtols:
            for district in districts:
                for policy in policies:
                    exp_id = "{}_policy={}_uncondtol={}_numfeatures={}".format(
                        district, policy, uncondtol, num_feat
                    )
                    script_fn = os.path.join(OUTPUT_PATH, "{}.sh".format(exp_id))
                    with open(script_fn, "w") as f:
                        print(
                            SBATCH_PREFACE.format(
                                exp_id, OUTPUT_PATH, exp_id, OUTPUT_PATH, exp_id
                            ),
                            file=f,
                        )
                        base_cmd = "python main.py main --policy {} --numfeatures {} --district {} --uncondtol {} --save {}".format(
                            policy, num_feat, district, uncondtol, SAVE_PATH
                        )
                        print(base_cmd, file=f)
                        print("sleep 1", file=f)


generate_malawi_runs()
