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
    total_df = total_df.reset_index()

    y = total_df["welfare"]
    r = total_df["wgt10"]
    total_df["urban"] = total_df["urban"].cat.codes
    features = [
        # entrpreneurial status
        "urban",
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
        #        "num_building",
        "num_computer",
        "num_electronics",
        "num_furniture",
        "num_generator",
        "num_house",
        "num_internet",
        "num_jewelry",
        "num_land",
        "num_mobile",
        #        "num_motorcycle",
        "num_radio",
        #        "num_solar_panel",
        "num_tv",
        "num_vehicle",
    ]
    X = total_df[features]
    one_hot_marital = pd.get_dummies(total_df[["maritalstat_head"]])
    one_hot_region = pd.get_dummies(total_df[["region"]])
    X = X.join(one_hot_marital)
    X = one_hot_region.join(X)
    X = X.fillna(X.mean())

    return X.to_numpy(), y.to_numpy(), r.to_numpy(), list(X.columns)


def load_dataset(dataset):
    if dataset == "uganda":
        return _load_uganda_merged()
    elif dataset == "ethiopia":
        return _load_ethiopia_merged()
    elif dataset == "malawi":
        return _load_malawi_merged()


def _load_ethiopia_merged():
    df = pd.read_csv(
        "/home/users/rsahoo/poverty/toy_opt/data_loaders/data/ethiopia_merged.csv"
    )

    df["HHID"] = df["HHID"].astype(int)

    total_df = total_df[total_df["welfare"].notna()]
    total_df = total_df.reset_index()

    y = total_df["welfare"]
    r = total_df["wgt10"]
    total_df["urban"] = total_df["urban"].cat.codes
    features = [
        # entrpreneurial status
        "urban",
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
        #        "num_building",
        "num_computer",
        "num_electronics",
        "num_furniture",
        "num_generator",
        "num_house",
        "num_internet",
        "num_jewelry",
        "num_land",
        "num_mobile",
        #        "num_motorcycle",
        "num_radio",
        #        "num_solar_panel",
        "num_tv",
        "num_vehicle",
    ]
    X = total_df[features]
    one_hot_marital = pd.get_dummies(total_df[["maritalstat_head"]])
    one_hot_region = pd.get_dummies(total_df[["region"]])
    X = X.join(one_hot_marital)
    X = one_hot_region.join(X)
    X = X.fillna(X.mean())

    return X.to_numpy(), y.to_numpy(), r.to_numpy(), list(X.columns)


def _load_malawi_merged():
    df = pd.read_csv(
        "/home/users/rsahoo/poverty/toy_opt/data_loaders/data/malawi_merged.csv"
    )

    """
['case_id', 'ea_id', 'reside', 'hh_wgt', 'hh_a01', 'hh_a02', 'yn_ac',
       'yn_bed', 'yn_beer_drum', 'yn_bicycle', 'yn_car', 'yn_cd_player',
       'yn_chair', 'yn_clock', 'yn_coffee_table', 'yn_computer', 'yn_cupboard',
       'yn_desk', 'yn_dish', 'yn_elec_stove', 'yn_fan', 'yn_generator',
       'yn_iron', 'yn_ker_stove', 'yn_lantern', 'yn_lorry', 'yn_mini_bus',
       'yn_mortar', 'yn_motorcycle', 'yn_radio', 'yn_refrigerator',
       'yn_sewing', 'yn_sofa', 'yn_solar', 'yn_table', 'yn_television',
       'yn_vcr', 'yn_wash_machine', 'latitude', 'longitude', 'dist_admarc',
       'dist_borderpost', 'precipitation', 'soil_quality', 'id_code',
       'sex_head', 'age_head', 'maritalstat_head', 'num_kids', 'num_adults',
       'hh_size', 'laborstat_head', 'dwelling_type', 'wall_type', 'roof_type',
       'floor_type', 'num_rooms', 'cooking_fuel', 'electricity',
       'water_source', 'toilet_type', 'yn_mobile', 'rexpagg']
    """
    df.dropna(axis=0, subset="rexpagg", inplace=True)
    df = df.reset_index()

    y = df["rexpagg"]
    adult_equiv = df["num_kids"] * 0.5 + df["num_adults"]
    adult_equiv = adult_equiv.fillna(adult_equiv.mean())
    y /= adult_equiv
    y /= 12
    #    y *= 0.0066

    r = df["hh_wgt"]
    df["laborstat_head"] = df["laborstat_head"].astype("category").cat.codes
    features = [
        "num_kids",
        "num_adults",
        "dist_admarc",
        "dist_borderpost",
        "yn_generator",
        "yn_elec_stove",
        "yn_ker_stove",
        "yn_radio",
        "yn_cd_player",
        "yn_bed",
        "latitude",
        "longitude",
        "laborstat_head",
        "precipitation",
        #        "soil_quality",
    ]
    X = df[features]
    one_hot_marital = pd.get_dummies(df[["maritalstat_head"]])
    one_hot_dwelling_type = pd.get_dummies(df[["dwelling_type"]])
    one_hot_wall_type = pd.get_dummies(df[["wall_type"]])

    X = one_hot_wall_type.join(X)
    X = one_hot_dwelling_type.join(X)
    X = X.join(one_hot_marital)
    X = X.fillna(X.mean())
    return X.to_numpy(), y.to_numpy(), r.to_numpy(), list(X.columns)
