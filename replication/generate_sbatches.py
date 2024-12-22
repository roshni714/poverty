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

OUTPUT_PATH = (
    "/home/users/rsahoo/zfs/gsb/intermediate-yens/rsahoo/poverty/replication/scripts"
)
SAVE_PATH = (
    "/home/users/rsahoo/zfs/gsb/intermediate-yens/rsahoo/poverty/replication/results/"
)


def generate_malawi_n_class_runs():
    ds = [0, 3, 12, 20]
    for d in ds:
        exp_id = "malawi_nclass_d={}".format(d)
        script_fn = os.path.join(OUTPUT_PATH, "{}.sh".format(exp_id))
        with open(script_fn, "w") as f:
            print(
                SBATCH_PREFACE.format(exp_id, OUTPUT_PATH, exp_id, OUTPUT_PATH, exp_id),
                file=f,
            )
            base_cmd = "python fixed_transfer_amt.py main --d {} --country malawi --uncondtol 0.10 --nclass 2 3 5 8 10 15 20 --save {}".format(
                d, SAVE_PATH
            )
            print(base_cmd, file=f)
            print("sleep 1", file=f)


def generate_malawi_runs():
    ds = [3, 12, 16, 23]

    policytypes = ["conditional", "unconditional", "oracle"]
    quantile_methods = ["qr"]


    for d in ds:
        for policytype in policytypes:
            if policytype == "conditional":
                for quantile_method in quantile_methods:
                    exp_id = "malawi_d={}_policytype={}_method={}".format(
                        d, policytype, quantile_method
                    )
                    script_fn = os.path.join(OUTPUT_PATH, "{}.sh".format(exp_id))
                    with open(script_fn, "w") as f:
                        print(
                            SBATCH_PREFACE.format(
                                exp_id, OUTPUT_PATH, exp_id, OUTPUT_PATH, exp_id
                            ),
                            file=f,
                        )
                        base_cmd = "python main.py main --d {} --policytype {} --method {} --country malawi --condtol 0.05 0.10 0.15 0.20 0.25 0.40 0.50 0.60 0.70 --save {}".format(
                            d, policytype, quantile_method, SAVE_PATH
                        )
                        print(base_cmd, file=f)
                        print("sleep 1", file=f)

            elif policytype == "unconditional":
                exp_id = "malawi_d={}_policytype={}".format(d, policytype)
                script_fn = os.path.join(OUTPUT_PATH, "{}.sh".format(exp_id))
                with open(script_fn, "w") as f:
                    print(
                        SBATCH_PREFACE.format(
                            exp_id, OUTPUT_PATH, exp_id, OUTPUT_PATH, exp_id
                        ),
                        file=f,
                    )
                    base_cmd = "python main.py main --d {} --policytype {} --country malawi --uncondtol 0.05 0.10 0.15 0.20 0.25 0.40 0.50 0.60 0.70 --save {}".format(
                        d, policytype, SAVE_PATH
                    )
                    print(base_cmd, file=f)
                    print("sleep 1", file=f)

            elif policytype == "hybrid":
                exp_id = "malawi_d={}_policytype={}".format(d, policytype)
                script_fn = os.path.join(OUTPUT_PATH, "{}.sh".format(exp_id))
                with open(script_fn, "w") as f:
                    print(
                        SBATCH_PREFACE.format(
                            exp_id, OUTPUT_PATH, exp_id, OUTPUT_PATH, exp_id
                        ),
                        file=f,
                    )
                    base_cmd = "python main.py main --d {} --policytype {} --country malawi --uncondtol 0.05 0.10 0.15 0.20 0.25 0.40 0.50 0.60 0.70 --condtol 0.25 --save {}".format(
                        d, policytype, SAVE_PATH
                    )
                    print(base_cmd, file=f)
                    print("sleep 1", file=f)

            elif policytype == "oracle":
                exp_id = "malawi_d={}_policytype={}".format(d, policytype)
                script_fn = os.path.join(OUTPUT_PATH, "{}.sh".format(exp_id))
                with open(script_fn, "w") as f:
                    print(
                        SBATCH_PREFACE.format(
                            exp_id, OUTPUT_PATH, exp_id, OUTPUT_PATH, exp_id
                        ),
                        file=f,
                    )
                    base_cmd = "python main.py main --d {} --policytype {} --country malawi --uncondtol 0.05 0.10 0.15 0.20 0.25 0.40 0.50 0.60 0.70 --save {}".format(d, policytype, SAVE_PATH
                    )
                    print(base_cmd, file=f)
                    print("sleep 1", file=f)


# generate_sim_runs()

# generate_1d_runs()
# generate_uganda_runs()
generate_malawi_runs()
