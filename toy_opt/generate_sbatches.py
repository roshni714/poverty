import itertools
import glob
import os


SBATCH_PREFACE = """#!/bin/bash
#SBATCH -t 12:00:00
#SBATCH -p normal
#SBATCH -c 1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name="{}.sh"
#SBATCH --error="{}/{}_err.log"
#SBATCH --output="{}/{}_out.log"\n
"""

# constants for commands

OUTPUT_PATH = "/home/users/rsahoo/poverty/toy_opt/scripts/"
SAVE_PATH = "/home/users/rsahoo/poverty/toy_opt/"


def generate_sim_runs():
    ds = [2, 5, 8, 10]

    for d in ds:
        exp_id = "sim_d={}".format(d)
        script_fn = os.path.join(OUTPUT_PATH, "{}.sh".format(exp_id))
        with open(script_fn, "w") as f:
            print(
                SBATCH_PREFACE.format(exp_id, OUTPUT_PATH, exp_id, OUTPUT_PATH, exp_id),
                file=f,
            )
            base_cmd = (
                "python main.py main --d {} --density_est_method glm_spline".format(d)
            )
            print(base_cmd, file=f)
            print("sleep 1", file=f)


def generate_uganda_runs():
    ds = [2, 5, 10]
    for d in ds:
        exp_id = "uganda_d={}".format(d)
        script_fn = os.path.join(OUTPUT_PATH, "{}.sh".format(exp_id))
        with open(script_fn, "w") as f:
            print(
                SBATCH_PREFACE.format(exp_id, OUTPUT_PATH, exp_id, OUTPUT_PATH, exp_id),
                file=f,
            )
            base_cmd = "python main_real.py main --d {} --density_est_method glm_spline".format(
                d
            )
            print(base_cmd, file=f)
            print("sleep 1", file=f)


generate_sim_runs()
# generate_uganda_runs()
