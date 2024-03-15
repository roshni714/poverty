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

OUTPUT_PATH = "/home/users/rsahoo/poverty/replication/scripts/"
SAVE_PATH = "/home/users/rsahoo/poverty/replication/"


def generate_sim_runs():
    budgets = [0.05, 0.075, 0.10, 0.15, 0.20]
    #    ds = [2]
    for budget in budgets:
        exp_id = "sim_budget={}".format(budget)
        script_fn = os.path.join(OUTPUT_PATH, "{}.sh".format(exp_id))
        with open(script_fn, "w") as f:
            print(
                SBATCH_PREFACE.format(exp_id, OUTPUT_PATH, exp_id, OUTPUT_PATH, exp_id),
                file=f,
            )
            base_cmd = "python main.py main --d 10 --density_est_method glm --budget {}".format(
                budget
            )
            print(base_cmd, file=f)
            print("sleep 1", file=f)


def generate_uganda_runs():
    #    ds = [5, 8, 12]
    #    budgets= [0.05, 0.075, 0.10, 0.15, 0.20]

    ds = [5]
    budgets = [0.05]

    for d in ds:
        for budget in budgets:
            exp_id = "uganda_d={}_budget={}".format(d, budget)
            script_fn = os.path.join(OUTPUT_PATH, "{}.sh".format(exp_id))

            with open(script_fn, "w") as f:
                print(
                    SBATCH_PREFACE.format(
                        exp_id, OUTPUT_PATH, exp_id, OUTPUT_PATH, exp_id
                    ),
                    file=f,
                )
                base_cmd = "python main_real.py main --d {} --density_est_method glm --country uganda --budget {}".format(
                    d, budget
                )
                print(base_cmd, file=f)
                print("sleep 1", file=f)


def generate_malawi_runs():
    ds = [0, 3, 12, 20]
    budgets = [0.05]

    for d in ds:
        for budget in budgets:
            exp_id = "malawi_d={}_budget={}".format(d, budget)
            script_fn = os.path.join(OUTPUT_PATH, "{}.sh".format(exp_id))
            with open(script_fn, "w") as f:
                print(
                    SBATCH_PREFACE.format(
                        exp_id, OUTPUT_PATH, exp_id, OUTPUT_PATH, exp_id
                    ),
                    file=f,
                )
                base_cmd = "python main_real.py main --d {} --density_est_method glm --country malawi --budget {}".format(
                    d, budget
                )
                print(base_cmd, file=f)
                print("sleep 1", file=f)


# generate_sim_runs()

# generate_1d_runs()
# generate_uganda_runs()
generate_malawi_runs()
