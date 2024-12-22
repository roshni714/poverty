import pandas as pd
import numpy as np
from data_utils import split_data

PATH_TO_DATA = (
    "/zfs/gsb/intermediate-yens/rsahoo/poverty/data/malawi_cleaned_2019.parquet"
)
CONVERSION_FACTORS = {"malawi": 0.003361735405}

CENTRAL_POOLED_DISTRICTS = [
    "kasungu",
    "mchinji",
    "dowa",
    "nkhotakota",
    "ntchisi",
    "lilongwe",
    "salima",
    "dedza",
    "ntcheu",
]

INDIVIDUAL_POOLED_DISTRICTS = ["kasungu"]


NORTHERN_POOLED_DISTRICTS = [
    "chitipa",
    "karonga",
    "mzimba",
    "rumphi",
    "nkhatabay",
    "lizoma",
]

CATEGORICAL_FEATURES = [
    "district",
    "hh_f12",
    "hh_f40",
    "area",
    "hh_f11",
    "hh_g09",
    "hh_f26_2",
    "hh_f41_2",
    "hh_m00",
    "hh_f06",
    "hh_f43",
    "hh_f41",
    "hh_f07",
    "hh_f12",
    "hh_f19",
    "hh_t10",
    "ag_s01",
    "hh_x07",
]


def get_rural_minus_district_dataset(district, covariates):
    df = pd.read_parquet(PATH_TO_DATA)
    district = district.title()

    # Drop rows of dataframe that missing outcome values
    df.dropna(axis=0, subset="rexpaggpc", inplace=True)
    df = df.reset_index()

    # Convert outcome to consumption per capita per day in terms of 2017 USD
    #    1. Use Shruthi's conversion factor's to convert to 2017 USD
    #    2. Convert household consumption to consumption per capita (using adult equivalence scale).
    #       If adult equiv is NaN, then impute the mean.
    #    3. Convert consumption to consumption per day
    # Note that datasets may differ in whether they report yearly or monthly consumption,
    # or their adult equivalence scale.
    df["outcome"] = df["rexpaggpc"].copy()
    df["outcome"] *= CONVERSION_FACTORS["malawi"]
    df["outcome"] /= 365
    y = df["outcome"]
    y = y[df["urban"].isin(["RURAL"]) & ~df.district.isin([district])]
    # Get survey weights
    r = df["hh_wgt"]
    r = r[df["urban"].isin(["RURAL"]) & ~df.district.isin([district])]

    if covariates is not None:
        X_cat = [
            pd.get_dummies(
                df[[cat_feat]],
                dummy_na=(
                    df[[cat_feat]].isna().sum().item() > 0.15 * len(df[cat_feat])
                ),
            ).astype(float)
            for cat_feat in covariates
            if cat_feat in CATEGORICAL_FEATURES
        ]
        if len(X_cat) > 1:
            X = X_cat[0].join(X_cat[1:])
        else:
            X = pd.DataFrame([])

        other_features = [
            feat for feat in covariates if feat not in CATEGORICAL_FEATURES
        ]

        for col in other_features:
            if df[col].isna().sum() > 0.15 * len(df):
                df[f"{col}_nan"] = df[col].isna().astype(float)
                other_features.append(f"{col}_nan")
                df[col] = df[col].fillna(0.0)
            else:
                df[col] = df[col].fillna(df[col].mean())
        X_other = df[sorted(other_features)]

        X = X_other.join(X)
        X = X[df["urban"].isin(["RURAL"]) & ~df.district.isin([district])]

    else:
        df = df[df["urban"].isin(["RURAL"]) & ~df.district.isin([district])]
        X = df[[]]

    return X.to_numpy(), y.to_numpy(), r.to_numpy(), X.columns


def get_district_dataset(districts, covariates):

    if districts == ["all"]:
        return get_full_dataset(covariates)

    districts = [district.title() for district in districts]

    df = pd.read_parquet(PATH_TO_DATA)

    # Drop rows of dataframe that missing outcome values
    df.dropna(axis=0, subset="rexpaggpc", inplace=True)
    df = df.reset_index()

    # Convert outcome to consumption per capita per day in terms of 2017 USD
    #    1. Use Shruthi's conversion factor's to convert to 2017 USD
    #    2. Convert household consumption to consumption per capita (using adult equivalence scale).
    #       If adult equiv is NaN, then impute the mean.
    #    3. Convert consumption to consumption per day
    # Note that datasets may differ in whether they report yearly or monthly consumption,
    # or their adult equivalence scale.
    df["outcome"] = df["rexpaggpc"].copy()
    df["outcome"] *= CONVERSION_FACTORS["malawi"]
    df["outcome"] /= 365
    y = df["outcome"]
    y = y[df["district"].isin(districts)]
    # Get survey weights
    r = df["hh_wgt"]
    r = r[df["district"].isin(districts)]

    if covariates is not None:
        X_cat = [
            pd.get_dummies(
                df[[cat_feat]],
                dummy_na=(
                    df[[cat_feat]].isna().sum().item() > 0.15 * len(df[cat_feat])
                ),
            ).astype(float)
            for cat_feat in covariates
            if cat_feat in CATEGORICAL_FEATURES
        ]

        if len(X_cat) > 0:
            X = X_cat[0].join(X_cat[1:])
        else:
            X = pd.DataFrame([])

        other_features = [
            feat for feat in covariates if feat not in CATEGORICAL_FEATURES
        ]

        for col in other_features:
            if df[col].isna().sum() > 0.15 * len(df):
                df[f"{col}_nan"] = df[col].isna().astype(float)
                other_features.append(f"{col}_nan")
                df[col] = df[col].fillna(0.0)
            else:
                df[col] = df[col].fillna(df[col].mean())
        X_other = df[sorted(other_features)]

        X = X_other.join(X)
        X = X[df["district"].isin(districts)]
    else:
        df = df[df["district"].isin(districts)]
        X = df[[]]

    # Filter data down to district level

    return X.to_numpy(), y.to_numpy(), r.to_numpy(), X.columns


def get_full_dataset(covariates):
    df = pd.read_parquet(PATH_TO_DATA)

    # Drop rows of dataframe that missing outcome values
    df.dropna(axis=0, subset="rexpaggpc", inplace=True)
    df = df.reset_index()

    # Convert outcome to consumption per capita per day in terms of 2017 USD
    #    1. Use Shruthi's conversion factor's to convert to 2017 USD
    #    2. Convert household consumption to consumption per capita (using adult equivalence scale).
    #       If adult equiv is NaN, then impute the mean.
    #    3. Convert consumption to consumption per day
    # Note that datasets may differ in whether they report yearly or monthly consumption,
    # or their adult equivalence scale.
    df["outcome"] = df["rexpaggpc"].copy()
    df["outcome"] *= CONVERSION_FACTORS["malawi"]
    df["outcome"] /= 365
    y = df["outcome"]
    # Get survey weights
    r = df["hh_wgt"]

    if covariates is not None:
        X_cat = [
            pd.get_dummies(
                df[[cat_feat]],
                dummy_na=(
                    df[[cat_feat]].isna().sum().item() > 0.15 * len(df[cat_feat])
                ),
            ).astype(float)
            for cat_feat in covariates
            if cat_feat in CATEGORICAL_FEATURES
        ]
        if len(X_cat) > 0:
            X = X_cat[0].join(X_cat[1:])
        else:
            X = pd.DataFrame([])

        other_features = [
            feat for feat in covariates if feat not in CATEGORICAL_FEATURES
        ]

        for col in other_features:
            if df[col].isna().sum() > 0.15 * len(df):
                df[f"{col}_nan"] = df[col].isna().astype(float)
                other_features.append(f"{col}_nan")
                df[col] = df[col].fillna(0.0)
            else:
                df[col] = df[col].fillna(df[col].mean())
        X_other = df[sorted(other_features)]

        X = X_other.join(X)
    else:
        X = df[[]]

    return X.to_numpy(), y.to_numpy(), r.to_numpy(), X.columns


def get_pooled_dataset(district, pool, covariates):
    if pool == "central":
        pooled = [d for d in CENTRAL_POOLED_DISTRICTS if district != d]
        X1, y1, r1, features1 = get_district_dataset(pooled, covariates=covariates)
    elif pool == "north":
        pooled = [d for d in NORTHERN_POOLED_DISTRICTS if district != d]
        X1, y1, r1, features1 = get_district_dataset(pooled, covariates=covariates)
    elif pool == "rural":
        X1, y1, r1, features1 = get_rural_minus_district_dataset(
            district, covariates=covariates
        )
    else:
        X1, y1, r1, features1 = get_district_dataset([pool], covariates=covariates)
    return X1, y1, r1, features1


def get_datasets(district, pool, covariates):
    if district == "all":
        X, y, r, features1 = get_full_dataset(covariates=covariates)
        fold1, fold2 = split_data(X=X, y=y, r=r, p=0.6)
    else:
        X2, y2, r2, features1 = get_district_dataset([district], covariates=covariates)
        X1, y1, r1, features2 = get_pooled_dataset(
            district, pool, covariates=covariates
        )
        fold1 = (X1, y1, r1)
        fold2 = (X2, y2, r2)
        assert list(features1) == list(features2)

        print("Conditional Density Dataset Size:", len(y1))
        print("Conditional Density Dataset Size:", len(y2))

    print(features1)

    return fold1, fold2, features1
