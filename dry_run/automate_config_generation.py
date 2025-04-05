import pandas as pd
import os
import yaml


def generate_gt_hparam_config(country, device):
    train_data = pd.read_parquet("data/{}/train.parquet".format(country))
    n_train = len(train_data)

    base_config = {
        "savedir": f"hparam/results/{country}",
        "device": device,
        "data": {
            "ntrain": n_train,
            "outcome": "consumption_per_capita_per_day",
            "weight": "hh_wgt",
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

    pmt_config = {
        "pmt": {"transfer_value": 2.15},
        "data": {"outcome": "consumption_per_capita_per_day", "weight": "hh_wgt"},
        "savedir": f"learn/results/{country}",
    }

    oracle_config = {
        "oracle_gap": {},
        "data": {"outcome": "consumption_per_capita_per_day", "weight": "hh_wgt"},
        "savedir": f"learn/results/{country}",
    }

    if not os.path.exists(f"hparam/configs/{country}"):
        os.makedirs(f"hparam/configs/{country}")

    if not os.path.exists(f"hparam/results/{country}"):
        os.makedirs(f"hparam/results/{country}")

    names = [
        "gt_continuous_rate",
        "gt_binary_rate",
        "gt_binary_gap",
        "gt_continuous_gap",
    ]
    configs = [
        continuous_rate_config,
        binary_rate_config,
        binary_gap_config,
        continuous_gap_config,
    ]

    for i, name in enumerate(names):
        config = configs[i]
        with open(f"hparam/configs/{country}/{name}.yaml", "w") as file:
            yaml.dump(config, file, default_flow_style=False)

    with open(f"hparam/results/{country}/pmt.yaml", "w") as file:
        yaml.dump(pmt_config, file, default_flow_style=False)

    with open(f"hparam/results/{country}/oracle_gap.yaml", "w") as file:
        yaml.dump(oracle_config, file, default_flow_style=False)


def generate_default_hparam_config(country):
    base_config = {
        "data": {"outcome": "consumption_per_capita_per_day", "weight": "hh_wgt"},
        "savedir": f"learn/results/{country}",
    }

    default_nn_config = {
        "n_regressors": 20,
        "neural_network": {"n_layers": 1, "n_hidden_units": 256, "lr": 0.005},
    }
    binary_gap_config = base_config.copy()
    binary_gap_config["binary_gap"] = default_nn_config.copy()
    continuous_gap_config = base_config.copy()
    continuous_gap_config["continuous_gap"] = default_nn_config.copy()
    binary_rate_config = base_config.copy()
    binary_rate_config["binary_rate"] = default_nn_config.copy()

    continuous_rate_config = base_config.copy()
    continuous_rate_config["continuous_rate"] = {
        "n_alpha": 200,
        "density_estimation": {
            "n_features": 10,
            "n_bins": 100,
            "n_knots": 6,
            "degree": 4,
        },
    }

    names = [
        "default_binary_gap",
        "default_binary_rate",
        "default_continuous_gap",
        "default_continuous_rate",
    ]
    configs = [
        binary_gap_config,
        binary_rate_config,
        continuous_gap_config,
        continuous_rate_config,
    ]

    if not os.path.exists(f"hparam/results/{country}"):
        os.makedirs(f"hparam/results/{country}")

    for i, name in enumerate(names):
        config = configs[i]
        with open(f"hparam/results/{country}/{name}.yaml", "w") as file:
            yaml.dump(config, file, default_flow_style=False)


countries = ["uganda", "malawi", "togo", "ethiopia", "nigeria"]
for country in countries:
    generate_gt_hparam_config(country, "cuda")
    # generate_default_hparam_config(country)
