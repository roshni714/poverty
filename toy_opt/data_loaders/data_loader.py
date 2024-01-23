import pandas as pd
import numpy as np


def _load_uganda_merged():
    df = pd.read_stata(
        "/home/users/rsahoo/poverty/toy_opt/data_loaders/data/clean/uganda_merged.dta"
    )
    df["HHID"] = df["HHID"].astype(int)
    df = df.set_index("HHID")
    other = pd.read_stata(
        "/home/users/rsahoo/poverty/toy_opt/data_loaders/data/clean/asset_indicators.dta"
    )
    other["HHID"] = other["HHID"].astype(int)
    other = other.set_index("HHID")
    total_df = df.join(other)

    total_df = total_df[total_df["welfare"].notna()]

    y = np.clip(total_df["welfare"], a_min=0.0, a_max=600000)
    r = total_df["wgt10"]
    total_df["urban"] = total_df["urban"].cat.codes
    features = [
        # entrpreneurial status
        "urban",
        "hh_income",
        "num_kids",
        "num_adults",
        "dist_market",
        "dist_borderpost",
        # type of stove
        # electricity access
        # type of stove
        # type of dwelling
        # number of rooms
        # average annual precipitation
        # soil quality
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
    X = total_df[features]
    one_hot_marital = pd.get_dummies(df[["maritalstat_head"]])
    X = X.join(one_hot_marital)
    X = X.fillna(X.mean())
    return X.to_numpy(), y.to_numpy(), r.to_numpy(), features


def load_uganda():
    return _load_uganda_merged()
