import pandas as pd
import wgan
import numpy as np
import dill


def load_data_for_wgan(path, summary_path):
    """
    Load data for WGAN training.

    Args:
        path (str): Path to the data file.

    Returns:
        data_for_wgan (pd.DataFrame): Data for WGAN training.
        data_wrapper (wgan.DataWrapper): DataWrapper object for WGAN training.
    """
    data = pd.read_parquet(path).reset_index(drop=True)
    summary = pd.read_parquet(summary_path)

    # Randomly select 50% of the data for training the WGAN
    rng = np.random.default_rng(145745893)
    train_rows = rng.choice(len(data), int(len(data) * 0.5), replace=False)
    data_for_wgan = data.iloc[train_rows].copy().reset_index(drop=True)

    # Identify which columns are continuous vs. categorical for the WGAN wrapper.

    numeric_columns = summary[summary["type"] == "numeric"]["covariate"].tolist()
    partial_categorical_columns = set(
        summary[summary["type"] == "categorical"]["covariate"]
    )
    enforced_categorical_columns = {c for c in data.columns if c.endswith("_nan")}
    categorical_columns = list(
        partial_categorical_columns.union(enforced_categorical_columns)
    )

    assert len(numeric_columns) + len(categorical_columns) == len(data_for_wgan.columns)

    categorical_mapping = {}
    for col in categorical_columns:
        categorical_mapping[col] = dict(
            zip(
                data_for_wgan[col].astype("category").cat.codes,
                data_for_wgan[col],
            )
        )
        data_for_wgan[col] = data_for_wgan[col].astype("category").cat.codes

    data_wrapper = wgan.DataWrapper(
        data_for_wgan,
        continuous_vars=numeric_columns,
        categorical_vars=categorical_columns,
    )
    return data_for_wgan, data_wrapper, categorical_mapping


def train_wgan(
    data_for_wgan,
    data_wrapper,
    device,
    batch_size=512,
    max_epochs=2000,
    lr=1e-3,
    dropout=0.1,
):
    """
    Train a WGAN model.

    Args:
        data_for_wgan (pd.DataFrame): Data for WGAN training.
        data_wrapper (wgan.DataWrapper): DataWrapper object for WGAN training.
        device (str): Device to train the WGAN model on.
        max_epochs (int): Maximum number of epochs for training.
        lr (float): Learning rate for training.
        dropout (float): Dropout rate for the generator.

    Returns:
        generator (wgan.Generator): Trained WGAN generator.
    """
    spec = wgan.Specifications(
        data_wrapper,
        batch_size=batch_size,
        max_epochs=max_epochs,
        critic_lr=lr,
        generator_lr=lr,
        critic_d_hidden=[256, 256, 256],
        generator_d_hidden=[256, 256, 256],
        generator_dropout=dropout,
        print_every=100,
        device=device,
    )
    generator = wgan.Generator(spec)
    critic = wgan.Critic(spec)
    x, context = data_wrapper.preprocess(data_for_wgan)
    wgan.train(generator, critic, x, context, spec)
    generator.to("cpu")
    return generator


def get_mean_std_error(df1, df2):
    """
    Compare two dataframes.

    Args:
        df1 (pd.DataFrame): First DataFrame.
        df2 (pd.DataFrame): Second DataFrame.

    Returns:
        mean_rmse (float): Mean RMSE between the two DataFrames.
        std_rmse (float): Standard deviation RMSE between the two DataFrames.
    """

    pos_vars = ["hh_wgt", "consumption_per_capita_per_day", "popdensity", "yearly_rent"]
    df1_new = df1.copy()
    df2_new = df2.copy()
    for var in pos_vars:
        df1_new[var] = np.log(df1_new["log_" + var] + 1e-5)
        df2_new[var] = np.log(df2_new["log_" + var] + 1e-5)
        df1_new = df1_new.drop(columns=[var])
        df2_new = df2_new.drop(columns=[var])

    mean1 = df1_new[df1_new.columns].mean()
    mean2 = df2_new[df1_new.columns].mean()

    std1 = df1_new[df1_new.columns].std()
    std2 = df2_new[df1_new.columns].std()

    mean_rmae = np.mean(np.abs((mean1 - mean2) / mean1.clip(1.0, None)))
    std_rmae = np.mean(np.abs((std1 - std2) / std1.clip(1.0, None)))

    return mean_rmae, std_rmae


# some of this preprocessing code should eventually be deprecated because
# it should be handled by prior data preprocessing code

# compute outcome conversion factor
# a = 340.2 / 430.05  # Malawi CPI in 2017 USD / Malawi CPI in 2019 USD
# b = 241.98  # Malawi Kwacha to USD exchange rate in 2017
# adulteq = data["adulteq"]
# can alternatively implement this as data["num_adults"] + alpha * data["num_children"]
# where alpha is in (0, 1).
# conversion_factor = (a / b) * (1 / 365) * (1 / adulteq)
# data["consumption_per_capita_per_day"] = data["rexpagg"] #* conversion_factor
# data["consumption_per_capita_per_day"] = np.clip(
#     data["consumption_per_capita_per_day"], 0, truncation_upper_value
# )

# we include hh_wgt and consumption_per_capita_per_day so that
# we can synthetically generate samples from the joint distribution (X, Y, R)
# durable_verifiable_covariates = list(
#    pd.read_csv("data/durable_verifiable_covariates.csv")["Covariates"]
# )

# data = data[
#    durable_verifiable_covariates + ["consumption_per_capita_per_day", "hh_wgt"]
# ]

# Log transform continuous variables that must be positive
# features_to_be_log_transformed = summary[(summary["minimum"] >= 0.) & (summary["type"] == 'numeric')]["covariate"].tolist()
# small_maximum = []
# for feature in features_to_be_log_transformed:
#     if data[feature].max() < 100:
#         small_maximum.append(feature)
# features_to_be_log_transformed = list(set(features_to_be_log_transformed) - set(small_maximum))
# log_cols = []
# new_rows = []
# for feature in features_to_be_log_transformed:
#     log_cols.append(pd.Series(np.log(np.clip(data[feature], 1e-5, None)), name="log_" + feature))
#     new_rows.append({'covariate': "log_" + feature,
#                                   'missing_count': summary[summary['covariate']==feature]['missing_count'].values[0].item(),
#                                    "median": np.log(summary[summary['covariate']==feature]['median'].values[0]),
#                                    'mean': np.mean(np.log(data[feature].clip(1e-5, None))),
#                                    'std': np.std(np.log(data[feature].clip(1e-5, None))),
#                                    'missing_fraction': summary[summary['covariate']==feature]['missing_fraction'].values[0].item(),
#                                    'description': 'Log transformation ' + summary[summary['covariate']==feature]['description'].values[0],
#                                    'module': summary[summary['covariate']==feature]['module'].values[0],
#                                    'type': summary[summary['covariate']==feature]['type'].values[0],
#                                    "minimum": np.log(summary[summary['covariate']==feature]['minimum'].values[0] + 1e-5),
#                                    'columns': summary[summary['covariate']==feature]['columns'].values[0]})
# data.drop(columns=features_to_be_log_transformed, inplace=True)
# data = pd.concat([data] + log_cols, axis=1)
# new_row_df = pd.DataFrame(new_rows)
# summary.drop(summary[summary['covariate'].isin(features_to_be_log_transformed)].index, inplace=True)
# summary = pd.concat([summary, new_row_df], axis=0).reset_index(drop=True)
