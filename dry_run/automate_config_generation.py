import pandas as pd
import os
import yaml
import numpy as np
from copy import deepcopy
from learn.data_loader import load_datasets


def generate_gt_hparam_config(country, geo_extrapolation, device):

    if geo_extrapolation:
        subfolder = "geo_extrapolation"
    else:
        subfolder = "geo_interpolation"

    base_config = {
        "savedir": f"hparam/results/{country}/{subfolder}",
        "device": device,
        "data": {
            "geo_extrapolation": geo_extrapolation,
            "outcome": "consumption_per_capita_per_day",
            "weight": "headcount_adjusted_hh_wgt",
            "povertyline": 2.15,
            "year": 2017,
            "gt": {
                "trainpath": "data/{}/train.parquet".format(country),
                "summarypath": "data/{}/summary.parquet".format(country),
                "auxpath": "data/auxiliary_data/auxiliary_data_20260409.csv",
            },
        },
    }

    default_nn_config = {
        "n_regressors": [10, 20, 30],
        "neural_network": {
            "n_layers": [1, 2, 3],
            "n_hidden_units": [256, 1024, 2048],
            "lr": [0.001, 0.005, 0.01],
        },
    }

    binary_gap_config = deepcopy(base_config)
    binary_gap_config["binary_gap"] = deepcopy(default_nn_config)

    binary_rate_config = deepcopy(base_config)
    binary_rate_config["binary_rate"] = deepcopy(default_nn_config)
    continuous_gap_config = deepcopy(base_config)
    continuous_gap_config["continuous_gap"] = deepcopy(default_nn_config)

    continuous_rate_config = deepcopy(base_config)
    continuous_rate_config["continuous_rate"] = {
        "n_alpha": [50, 100, 200],  # if country != "IDN" else [50, 100],
        "density_estimation": {
            "n_features": [5, 10, 15, 20],
            "n_bins": [10, 50, 100, 200],
            "n_knots": [2, 6, 12],
            "degree": [2, 4, 6],
        },
    }

    train_data, _, _, _ = load_datasets(
        base_config["data"]["gt"]["trainpath"],
        base_config["data"]["gt"]["trainpath"],
        base_config["data"]["gt"]["summarypath"],
        base_config["data"]["gt"]["auxpath"],
        geo_extrapolation=True,
        outcome=base_config["data"]["outcome"],
        weight=base_config["data"]["weight"],
        year=base_config["data"]["year"],
    )

    modern_pmt_config = deepcopy(base_config)
    modern_pmt_config["modern_pmt"] = deepcopy(default_nn_config)
    del modern_pmt_config["modern_pmt"]["n_regressors"]

    pmt_config = deepcopy(base_config)
    pmt_config["pmt"] = {
        "lasso": {"alpha": [0, 0.01, 0.1, 1.0]},
    }

    welfare_config = deepcopy(base_config)
    welfare_config["welfare"] = deepcopy(default_nn_config)

    pmt_gap_config = deepcopy(pmt_config)
    pmt_gap_config["pmt_gap"] = pmt_gap_config.pop("pmt")
    del pmt_gap_config["pmt_gap"]["transfer_value"]

    oracle_config = {
        "oracle_gap": {},
        "data": {
            "outcome": "consumption_per_capita_per_day",
            "weight": "headcount_adjusted_hh_wgt",
            "geo_extrapolation": geo_extrapolation,
        },
        "savedir": f"learn/results/{country}/{subfolder}",
    }

    ubi_config = {
        "ubi": {},
        "data": {
            "outcome": "consumption_per_capita_per_day",
            "weight": "headcount_adjusted_hh_wgt",
            "geo_extrapolation": geo_extrapolation,
        },
        "savedir": f"learn/results/{country}/{subfolder}",
    }

    if not os.path.exists(f"hparam/configs/{country}/{subfolder}"):
        os.makedirs(f"hparam/configs/{country}/{subfolder}")

    if not os.path.exists(f"hparam/results/{country}/{subfolder}"):
        os.makedirs(f"hparam/results/{country}/{subfolder}")

    names = [
        "gt_continuous_rate",
        "gt_binary_rate",
        "gt_binary_gap",
        "gt_continuous_gap",
        "gt_modern_pmt",
        "gt_pmt",
        "gt_welfare",
        "gt_pmt_gap",
    ]
    configs = [
        continuous_rate_config,
        binary_rate_config,
        binary_gap_config,
        continuous_gap_config,
        modern_pmt_config,
        pmt_config,
        welfare_config,
        pmt_gap_config,
    ]

    for i, name in enumerate(names):
        config = configs[i]
        with open(f"hparam/configs/{country}/{subfolder}/{name}.yaml", "w") as file:
            yaml.dump(config, file, default_flow_style=False)

    with open(f"hparam/results/{country}/{subfolder}/oracle_gap.yaml", "w") as file:
        yaml.dump(oracle_config, file, default_flow_style=False)

    with open(f"hparam/results/{country}/{subfolder}/ubi.yaml", "w") as file:
        yaml.dump(ubi_config, file, default_flow_style=False)


countries = [
    "BDI",
    "BGD",
    "BEN",
    "BFA",
    "BGD",
    "CAF",
    "CIV",
    "COL",
    "COD",
    "GHA",
    "ETH",
    "GNB",
    "IDN",
    "IND",
    "KEN",
    "LBR",
    "MDG",
    "MEX",
    "MWI",
    "MLI",
    "NER",
    "NGA",
    "PAK",
    "RWA",
    "SDN",
    "SEN",
    "TZA",
    "ZAF",
    "TGO",
    "UGA",
    "TLS",
    "ZWE",
    "YEM",
    "ZAF",
    "TGO_alpha_earth",
    "TGO_alpha_earth_and_survey",
]
geo_extrapolation = [True]
for country in countries:
    for geo in geo_extrapolation:
        generate_gt_hparam_config(country, geo, "cpu")
    # generate_default_hparam_config(country)
