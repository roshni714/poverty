import pandas as pd
import os
import yaml


def generate_hparam_config(country, device):
    train_data = pd.read_parquet("data/{}/train.parquet".format(country))
    test_data = pd.read_parquet("data/{}/test.parquet".format(country))
    n_train = len(train_data)
    n_test = len(test_data)
    n_val = len(train_data) * 3

    base_config = {
        "savedir": "hparam/results/{country}",
        "device": device,
        "data": {
            "ntrain": n_train,
            "ntest": n_test,
            "nval": n_val,
            "outcome": "consumption_per_capita_per_day",
            "weight": "hh_wgt",
            "gt": {
                "trainpath": "data/{country}/train.parquet",
                "summarypath": "data/{country}/summary.parquet",
            },
        },
    }

    binary_gap_config = base_config.copy()
    binary_gap_config["binary_gap"] = {
        "n_regressors": [10, 20, 30],
        "neural_network": {
            "n_layers": [1, 2, 3],
            "n_hidden_units": [256, 1024, 2048],
            "lr": [0.001, 0.005, 0.01],
        },
    }

    binary_rate_config = base_config.copy()
    binary_rate_config["binary_rate"] = {
        "n_regressors": [10, 20, 30],
        "neural_network": {
            "n_layers": [1, 2, 3],
            "n_hidden_units": [256, 1024, 2048],
            "lr": [0.001, 0.005, 0.01],
        },
    }

    continuous_gap_config = base_config.copy()
    continuous_gap_config["continuous_gap"] = {
        "n_regressors": [10, 20, 30],
        "neural_network": {
            "n_layers": [1, 2, 3],
            "n_hidden_units": [256, 1024, 2048],
            "lr": [0.001, 0.005, 0.01],
        },
    }
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

    if not os.path.exists(f"hparam/configs/{country}"):
        os.makedirs(f"hparam/configs/{country}")

    with open(f"hparam/configs/{country}/gt_continuous_rate.yaml", "w") as file:
        yaml.dump(continuous_rate_config, file, default_flow_style=False)

    with open(f"hparam/configs/{country}/gt_binary_rate.yaml", "w") as file:
        yaml.dump(binary_rate_config, file, default_flow_style=False)

    with open(f"hparam/configs/{country}/gt_continuous_gap.yaml", "w") as file:
        yaml.dump(continuous_gap_config, file, default_flow_style=False)

    with open(f"hparam/configs/{country}/gt_binary_gap.yaml", "w") as file:
        yaml.dump(binary_gap_config, file, default_flow_style=False)


generate_hparam_config("togo", "cuda")
generate_hparam_config("ethiopia", "cuda")
