import pandas as pd


def _load_asset_indicators():
    df = pd.read_stata("data/asset_indicators.dta")
    df = df.fillna(0)

    features = [
        "HHID",
        "num_appliance",
        "num_bicycle",
        "num_boat",
        "num_building",
        "num_computer",
        "num_electronics",
        "num_furniture",
        "num_generator",
        "num_house",
        "num_internet",
        "num_jewelry",
        "num_land",
        "num_mobile",
        "num_motorcycle",
        "num_radio",
        "num_solar_panel",
        "num_tv",
        "num_vehicle",
    ]

    _load_asset_indicators()
