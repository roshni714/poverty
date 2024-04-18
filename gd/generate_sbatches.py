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


def generate_malawi_runs():

    uncondtols = [0.10]
    districts = ["mchinji"]
    policies = ["oracle"]

    for uncondtol in uncondtols:
        for district in districts:
            for policy in policies:
                exp_id = "{}_policy={}_uncondtol={}".format(district, policy, uncondtol)
                script_fn = os.path.join(OUTPUT_PATH, "{}.sh".format(exp_id))
                with open(script_fn, "w") as f:
                    print(
                        SBATCH_PREFACE.format(
                            exp_id, OUTPUT_PATH, exp_id, OUTPUT_PATH, exp_id
                        ),
                        file=f,
                    )
                    base_cmd = "python main.py main --policy {} --district {} --uncondtol {} --save {}".format(
                        policy, district, uncondtol, SAVE_PATH
                    )
                    print(base_cmd, file=f)
                    print("sleep 1", file=f)


generate_malawi_runs()
