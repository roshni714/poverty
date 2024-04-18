import pandas as pd
import numpy as np

PATH_TO_DATA = "/zfs/gsb/intermediate-yens/rsahoo/poverty/data/malawi_merged_2019.csv"
CONVERSION_FACTORS = {"malawi": 0.003361735405}


def get_dataset(district, covariates):

    df = pd.read_csv(PATH_TO_DATA)

    # Drop rows of dataframe that missing outcome values
    df.dropna(axis=0, subset="rexpagg", inplace=True)
    df = df.reset_index()

    # Filter data down to district level
    df = df[df.district == district.capitalize()]

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

    if covariates is not None:
        categorical_features = (
            covariates  # TODO: For now only assuming categorical features
        )
        X_cat = [
            pd.get_dummies(
                df[[cat_feat]],
                dummy_na=(
                    df[[cat_feat]].isna().sum().item() > 0.15 * len(df[cat_feat])
                ),
            ).astype(float)
            for cat_feat in categorical_features
        ]
        X = X_cat
    else:
        X = df[[]]

    # Get survey weights
    r = df["hh_wgt"]
    return X.to_numpy(), y.to_numpy(), r.to_numpy(), X.columns
