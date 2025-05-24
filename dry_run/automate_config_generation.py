import pandas as pd
import os
import yaml


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
            "gt": {
                "trainpath": "data/{}/train.parquet".format(country),
                "summarypath": "data/{}/summary.parquet".format(country),
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

    binary_gap_config = base_config.copy()
    binary_gap_config["binary_gap"] = default_nn_config.copy()

    binary_rate_config = base_config.copy()
    binary_rate_config["binary_rate"] = default_nn_config.copy()

    continuous_gap_config = base_config.copy()
    continuous_gap_config["continuous_gap"] = default_nn_config.copy()

    continuous_rate_config = base_config.copy()
    continuous_rate_config["continuous_rate"] = {
        "n_alpha": [50, 100, 200],
        "density_estimation": {
            "n_features": [5, 10, 15, 20],
            "n_bins": [10, 50, 100, 200],
            "n_knots": [2, 6, 12],
            "degree": [2, 4, 6],
        },
    }

    modern_pmt_config = base_config.copy()
    modern_pmt_config["modern_pmt"] = default_nn_config.copy()
    del modern_pmt_config["modern_pmt"]["n_regressors"]

    pmt_config = {
        "pmt": {"transfer_value": 2.15},
        "data": {
            "outcome": "consumption_per_capita_per_day",
            "weight": "headcount_adjusted_hh_wgt",
            "geo_extrapolation": geo_extrapolation,
        },
        "savedir": f"learn/results/{country}/{subfolder}",
    }

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
    ]
    configs = [
        continuous_rate_config,
        binary_rate_config,
        binary_gap_config,
        continuous_gap_config,
        modern_pmt_config,
    ]

    for i, name in enumerate(names):
        config = configs[i]
        with open(f"hparam/configs/{country}/{subfolder}/{name}.yaml", "w") as file:
            yaml.dump(config, file, default_flow_style=False)

    with open(f"hparam/results/{country}/{subfolder}/pmt.yaml", "w") as file:
        yaml.dump(pmt_config, file, default_flow_style=False)

    with open(f"hparam/results/{country}/{subfolder}/oracle_gap.yaml", "w") as file:
        yaml.dump(oracle_config, file, default_flow_style=False)

    with open(f"hparam/results/{country}/{subfolder}/ubi.yaml", "w") as file:
        yaml.dump(ubi_config, file, default_flow_style=False)


countries = ["niger", "guinea_bissau", "south_africa", "cote_divoire", "burkina_faso"]
geo_extrapolation = [True]
for country in countries:
    for geo in geo_extrapolation:
        generate_gt_hparam_config(country, geo, "cpu")
    # generate_default_hparam_config(country)
