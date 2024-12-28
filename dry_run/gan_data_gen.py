import pandas as pd
import wgan
import numpy as np
import dill


def load_data_for_wgan(path):
    """
    Load data for WGAN training.

    Args:
        path (str): Path to the data file.

    Returns:
        data_for_wgan (pd.DataFrame): Data for WGAN training.
        data_wrapper (wgan.DataWrapper): DataWrapper object for WGAN training.
    """
    data = pd.read_parquet(path)

    # some of this preprocessing code should eventually be deprecated because
    # it should be handled by prior data preprocessing code

    # compute outcome conversion factor
    a = 340.2 / 430.05  # Malawi CPI in 2017 USD / Malawi CPI in 2019 USD
    b = 241.98  # Malawi Kwacha to USD exchange rate in 2017
    adulteq = data["adulteq"]
    # can alternatively implement this as data["num_adults"] + alpha * data["num_children"]
    # where alpha is in (0, 1).
    conversion_factor = (a / b) * (1 / 365) * (1 / adulteq)
    data["consumption_per_capita_per_day"] = data["rexpagg"] * conversion_factor
    # data["consumption_per_capita_per_day"] = np.clip(
    #     data["consumption_per_capita_per_day"], 0, truncation_upper_value
    # )

    # we include hh_wgt and consumption_per_capita_per_day so that
    # we can synthetically generate samples from the joint distribution (X, Y, R)
    durable_verifiable_covariates = list(
        pd.read_csv("data/durable_verifiable_covariates.csv")["Covariates"]
    )

    data = data[
        durable_verifiable_covariates + ["consumption_per_capita_per_day", "hh_wgt"]
    ]

    # Log transform continuous variables that must be positive
    positive_features = [
        "consumption_per_capita_per_day",
        "hh_wgt",
        "popdensity",
        "yearly_rent",
    ]
    for feature in positive_features:
        data["log_" + feature] = np.log(np.clip(data[feature], 1e-5, None))
    data.drop(columns=positive_features, inplace=True)

    # Randomly select 50% of the data for training the WGAN
    rng = np.random.default_rng(145745893)
    train_rows = rng.choice(len(data), int(len(data) * 0.5), replace=False)
    data_for_wgan = data.iloc[train_rows].copy().reset_index(drop=True)

    # Identify which columns are continuous vs. categorical for the WGAN wrapper.

    numeric_columns = set(data_for_wgan.select_dtypes(include=[np.number]).columns)

    # Treat integer-valued columns as categorical for synthetic data generation
    # (not necessary to do this for learning)
    integer_columns = set(
        [
            col
            for col in numeric_columns
            if np.all(data[col].apply(lambda x: int(x) == x))
        ]
    )

    non_numeric_columns = set(
        data_for_wgan.select_dtypes(exclude=[np.number, np.datetime64]).columns
    )

    enforced_categorical = {c for c in numeric_columns if c.endswith("_nan")}
    numeric_columns = list(numeric_columns - enforced_categorical - integer_columns)
    actual_categorical = list(non_numeric_columns | enforced_categorical)
    wgan_categorical = list(
        non_numeric_columns | enforced_categorical | integer_columns
    )

    categorical_mapping = {}
    for col in actual_categorical:
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
        categorical_vars=wgan_categorical,
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


def generate_synthetic_data(
    generator, data_wrapper, categorical_mapping, nsamples, seed
):
    """
    Generate synthetic data using a trained WGAN generator.

    Args:
        generator (wgan.Generator): Trained WGAN generator.
        data_wrapper (wgan.DataWrapper): DataWrapper object for WGAN training.
        categorical_mapping (list): List of dictionaries containing mappings for categorical variables.
        nsamples (int): Number of samples to generate.
        seed (int): Random seed for synthetic data generation.

    Returns:
        synthetic_df (pd.DataFrame): Synthetic data generated by the WGAN generator.
    """
    rng = np.random.default_rng(seed)
    rand_ints = rng.normal(0, 2**20, nsamples)
    synthetic_df = data_wrapper.apply_generator(generator, pd.DataFrame(rand_ints))
    vars = (
        data_wrapper.variables["continuous"]
        + data_wrapper.variables["context"]
        + data_wrapper.variables["categorical"]
    )
    synthetic_df.drop(columns=[0], inplace=True)
    for feature in vars:
        if feature.startswith("log_"):
            new_feature_name = feature[4:]
            synthetic_df[new_feature_name] = np.exp(synthetic_df[feature])
            synthetic_df.drop(columns=[feature], inplace=True)
    for col in categorical_mapping:
        synthetic_df[col] = synthetic_df[col].map(categorical_mapping[col])
    return synthetic_df


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
    mean1 = df1[df1.columns].mean()
    mean2 = df2[df1.columns].mean()

    std1 = df1[df1.columns].std()
    std2 = df2[df1.columns].std()

    mean_rmae = np.mean(np.abs((mean1 - mean2) / mean1.clip(1.0, None)))
    std_rmae = np.mean(np.abs((std1 - std2) / std1.clip(1.0, None)))

    return mean_rmae, std_rmae
