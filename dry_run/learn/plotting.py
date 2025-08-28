import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from learn.aggregation import (
    METHODS,
    AggregatePovertyResults,
    CountryMethodPovertyResults,
    COUNTRY_AUX_DATA_CSV,
    SECONDARY_AUX_DATA_CSV,
    preprocess_wpc_data
)
from learn.predictive_quality import get_out_of_sample_rmse

POVERTY_RATE_TARGET = 5.0


def get_country_name(country):
    if country == "cote_divoire":
        return "Côte d'Ivoire"
    if country == "congo_dr":
        return "Democratic Republic of the Congo"
    if country == "south_africa":
        return "South Africa"
    if country == "south_sudan":
        return "South Sudan"
    else:
        return "-".join([word.capitalize() for word in country.split("_")])


def get_data_dimension(country):
    train_data = pd.read_parquet("data/{}/train.parquet".format(country))
    n_train = len(train_data)
    test_data = pd.read_parquet("data/{}/test.parquet".format(country))
    n_test = len(test_data)
    n = n_train + n_test
    d = len(train_data.columns)
    # remove hh_wgt, household_size_adjusted_hh_wgt, and consumption_per_capita_per_day
    d -= 3
    return n, d


def make_plot_for_country(
    country, method_list, geo_extrapolation, povertyline, year, save_as, ubi_off=False
):

    methods = METHODS.copy()
    results = []
    for i, method in enumerate(method_list):
        results.append(
            CountryMethodPovertyResults(
                country, method, geo_extrapolation, povertyline=povertyline, year=year
            )
        )

    oracle_results = CountryMethodPovertyResults(
        country, "oracle_gap", geo_extrapolation, povertyline=povertyline, year=year
    )
    fontsize = 30
    # Plot policy_cost_per_capita vs post_transfer_poverty_rate
    fig, ax = plt.subplots(1, 2, figsize=(24, 8))
    if not ubi_off:
        ax[1].plot(
            np.linspace(0.0, results[i].initial_gap),
            np.ones(50) * povertyline * results[0].conversion_factor,
            linestyle="--",
            color=METHODS["ubi"]["color"],
            label="UBI ${}".format(povertyline),
        )
        ax[0].plot(
            np.linspace(0.0, results[i].initial_rate),
            np.ones(50) * povertyline * results[0].conversion_factor,
            linestyle="--",
            color=METHODS["ubi"]["color"],
            label="UBI ${}".format(povertyline),
        )
        print("ubi ${}".format(povertyline), povertyline * results[0].conversion_factor)

    for i, method in enumerate(method_list):
        dic = methods[method]
        df = results[i]._load_data(method)

        rates = [oracle_results.initial_rate] + list(
            df["post_transfer_poverty_rate"] * 100
        )
        gaps = [oracle_results.initial_gap] + list(
            df["post_transfer_poverty_gap"] * 100 / povertyline
        )
        costs = [0.0] + list(
            df["policy_cost_per_capita"] * results[0].conversion_factor
        )

        ax[0].plot(
            rates,
            costs,
            marker="o",
            label=dic["name"],
            color=dic["color"],
            linestyle=dic["linestyle"],
            linewidth=3,
        )

        ax[1].plot(
            gaps,
            costs,
            marker="o",
            label=dic["name"],
            color=dic["color"],
            linestyle=dic["linestyle"],
            linewidth=3,
        )
    ax[0].set_xlabel(
        "Post-Transfer Poverty Rate (%)",
        fontsize=fontsize,
    )
    ax[1].set_xlabel(
        "Post-Transfer Poverty Gap Index (%)",
        fontsize=fontsize,
    )

    for i in range(2):
        ax[i].set_ylabel("Policy Cost ($ Billion Per Year)", fontsize=fontsize)
        ax[i].grid(True)
        ax[i].tick_params(axis="x", labelsize=fontsize * 0.75)
        ax[i].tick_params(axis="y", labelsize=fontsize * 0.75)

    ax[1].legend(fontsize=fontsize * 0.75)  # , #bbox_to_anchor=(1.05, 0.5)
    # fig.tight_layout(rect=[0, 0, 0.85, 1])
    plt.tight_layout()
    plt.savefig("{}.pdf".format(save_as), bbox_inches="tight")
    plt.close()


def aggregate_plot(
    countries, method_list, geo_extrapolation, povertyline, year, save_as
):
    # Plot policy_cost_per_capita vs post_transfer_poverty_rate
    fig, ax = plt.subplots(1, 2, figsize=(24, 8))
    fontsize = 30

    results = []

    for method in method_list:
        results.append(
            AggregatePovertyResults(
                countries=countries,
                method=method,
                geo_extrapolation=geo_extrapolation,
                povertyline=povertyline,
                year=year,
            )
        )

    oracle_results = AggregatePovertyResults(
        countries, "oracle_gap", geo_extrapolation, povertyline=povertyline, year=year
    )
    initial_gap, initial_rate = oracle_results.get_initial_aggregate_gap_and_rate()

    ubi_cost = results[0].get_aggregate_ubi_cost()

    ax[0].plot(
        np.linspace(0.0, initial_rate),
        np.ones(50) * ubi_cost,
        linestyle="--",
        color=METHODS["ubi"]["color"],
        label="UBI ${}".format(povertyline),
    )
    ax[1].plot(
        np.linspace(0.0, initial_gap),
        np.ones(50) * ubi_cost,
        linestyle="--",
        color=METHODS["ubi"]["color"],
        label="UBI ${}".format(povertyline),
    )

    for i, method in enumerate(method_list):
        gap_domain = results[i].aggregate_interpolator_gap_domain
        rate_domain = results[i].aggregate_interpolator_rate_domain
        rate_interpolator = results[i].aggregate_interpolator_rate_to_cost
        gap_interpolator = results[i].aggregate_interpolator_gap_to_cost
        ax[0].plot(
            np.linspace(rate_domain[0], rate_domain[1], 200),
            rate_interpolator(np.linspace(rate_domain[0], rate_domain[1], 200)),
            label=METHODS[method]["name"],
            color=METHODS[method]["color"],
            linestyle=METHODS[method]["linestyle"],
        )
        ax[1].plot(
            np.linspace(gap_domain[0], gap_domain[1], 200),
            gap_interpolator(np.linspace(gap_domain[0], gap_domain[1], 200)),
            label=METHODS[method]["name"],
            color=METHODS[method]["color"],
            linestyle=METHODS[method]["linestyle"],
        )

    ax[0].set_xlabel(
        "Post-Transfer Poverty Rate\n(%)",
        fontsize=fontsize,
    )
    ax[1].set_xlabel(
        "Post-Transfer Poverty Gap Index\n(%)",
        fontsize=fontsize,
    )

    for i in range(2):
        ax[i].set_ylabel("Policy Cost ($ Billion Per Year)", fontsize=fontsize)
        ax[i].grid(True)
        ax[i].tick_params(axis="x", labelsize=fontsize * 0.75)
        ax[i].tick_params(axis="y", labelsize=fontsize * 0.75)

    ax[1].legend(fontsize=fontsize * 0.75)  # , #bbox_to_anchor=(1.05, 0.5)
    # fig.tight_layout(rect=[0, 0, 0.85, 1])

    plt.tight_layout()
    plt.savefig("{}.pdf".format(save_as), dpi=300, bbox_inches="tight")
    plt.close()


def convert_nominal_2023_to_nominal_survey_year(amt, country):
    df = pd.read_csv(COUNTRY_AUX_DATA_CSV)
    second_df = pd.read_csv("learn/inflation_adjustment.csv")
    survey_year = int(df[df["country"] == country]["survey_year"].values[0])
    print(survey_year)
    inflation_adjustment = second_df[second_df.survey_year == survey_year][
        "inflation_adjustment_to_2023"
    ].values[0]
    amt = amt * (1 / inflation_adjustment)
    return amt


def plot_bar_chart_policy_amt_as_percent_of_gdp(
    countries, geo_extrapolation, povertyline, year, save_as
):
    results = [
        CountryMethodPovertyResults(
            country,
            "continuous_gap",
            geo_extrapolation,
            povertyline=povertyline,
            year=year,
        )
        for country in countries
    ]

    amts = []  # nominal 2023 USD amts
    for i in range(len(countries)):
        amt = results[i].rate_to_cost_interpolator(POVERTY_RATE_TARGET).item()
        amts.append(amt)

    amts_survey_year = [
        convert_nominal_2023_to_nominal_survey_year(amt, country)
        for amt, country in zip(amts, countries)
    ]

    df = pd.read_csv(COUNTRY_AUX_DATA_CSV)
    gdp = (
        df[df["country"].isin(countries)][["country", "GDP_survey_year"]]
        .set_index("country")
        .to_dict()["GDP_survey_year"]
    )
    gdp = {country: gdp[country] for country in countries}
    amts_as_percent_of_gdp = np.array(
        [amt * 100 / gdp[country] for amt, country in zip(amts_survey_year, countries)]
    )

    govt_revenue_percentage = (
        df[df["country"].isin(countries)][
            ["country", "government_revenue_percentage_survey_year"]
        ]
        .set_index("country")
        .to_dict()["government_revenue_percentage_survey_year"]
    )
    govt_revenue = {
        country: govt_revenue_percentage[country] * gdp[country] / 100
        for country in countries
    }
    amts_as_percent_of_revenue = np.array(
        [
            amt * 100 / govt_revenue[country]
            for amt, country in zip(amts_survey_year, countries)
        ]
    )

    xlabels = np.array([get_country_name(c) for c in countries])
    sort_index = np.argsort(amts_as_percent_of_gdp)[::-1]

    fig, axes = plt.subplots(2, 1, figsize=(30, 8 * 2))
    fontsize = 30
    # Bar plot for amts_as_percent_of_gdp
    axes[0].bar(xlabels[sort_index], amts_as_percent_of_gdp[sort_index], zorder=3)
    axes[0].set_xlabel("Country", fontsize=fontsize)
    axes[0].set_ylabel("% of GDP", fontsize=fontsize)
    # axes[0].set_title("Policy Cost as Percentage of Country GDP", fontsize=fontsize)
    axes[0].set_xticklabels(xlabels[sort_index], rotation=90, fontsize=fontsize)
    axes[0].set_yticklabels(axes[0].get_yticks(), fontsize=fontsize)

    sort_index2 = np.argsort(amts_as_percent_of_revenue)[::-1]
    axes[0].grid(axis="y", zorder=0)
    axes[1].grid(axis="y", zorder=0)
    # Bar plot for amts_as_percent_of_revenue
    axes[1].bar(xlabels[sort_index2], amts_as_percent_of_revenue[sort_index2], zorder=3)
    axes[1].set_xlabel("Country", fontsize=fontsize)
    axes[1].set_ylabel("% of Gov't Revenue", fontsize=fontsize)
    axes[1].set_xticklabels(xlabels[sort_index2], rotation=90, fontsize=fontsize)
    axes[1].set_yticklabels(axes[1].get_yticks(), fontsize=fontsize)
    # axes[1].set_title("Policy Cost as Percentage of Country Govt Revenue", fontsize=fontsize)

    plt.tight_layout()
    plt.savefig("{}.pdf".format(save_as), bbox_inches="tight")


def get_table_policy_cost_gdp(countries, povertyline, year, save_as):
    df = pd.read_csv(COUNTRY_AUX_DATA_CSV)

    results = [
        CountryMethodPovertyResults(
            country,
            "continuous_gap",
            geo_extrapolation=True,
            povertyline=povertyline,
            year=year,
        )
        for country in countries
    ]

    res = []
    for result in results:
        amt = result.rate_to_cost_interpolator(POVERTY_RATE_TARGET).item()
        res.append({"country": result.country, "policy_cost": amt})

    df2 = pd.DataFrame(res)
    df = df2.merge(df, on="country", how="left")
    df.sort_values(by=["country"], inplace=True)
    df["Policy Cost / GDP"] = df["policy_cost"] / df["GDP_survey_year"]
    df["government_revenue_survey_year"] = (
        df["government_revenue_percentage_survey_year"] * df["GDP_survey_year"] / 100
    )
    df["Policy Cost / Gov't Revenue"] = (
        df["policy_cost"] / df["government_revenue_survey_year"]
    )
    df["survey_year"] = df["survey_year"].astype(int)
    df.rename(
        columns={
            "policy_cost": "Policy Cost",
            "GDP_survey_year": "GDP",
            "survey_year": "Reference Year",
            "country": "Country",
            "government_revenue_survey_year": "Gov't Revenue",
        },
        inplace=True,
    )
    df = df.sort_values(by=["Country"])
    new_df = df[
        [
            "Country",
            "Reference Year",
            "Policy Cost",
            "GDP",
            "Gov't Revenue",
            "Policy Cost / GDP",
            "Policy Cost / Gov't Revenue",
        ]
    ]

    new_df.to_latex(
        save_as + ".tex",
        index=False,
        float_format="%.2f",
        escape=False,
        formatters={"Country": get_country_name},
    )


def get_table_oecd(policy_cost, save_as):
    df = pd.read_csv("learn/auxiliary_data.csv")
    df["GDP"] = df["OECD_nominal_GDP_2023_billions"]
    df["Gov't Revenue"] = df["GDP"] * df["OECD_govt_revenue_percentage_GDP_2023"] / 100

    df["Policy Cost"] = policy_cost
    df["Policy Cost (\\% of GDP)"] = df["Policy Cost"] * 100 / df["GDP"]
    df["Policy Cost (\\% of Gov't Revenue)"] = (
        df["Policy Cost"] * 100 / df["Gov't Revenue"]
    )

    new_df = df[
        [
            "GDP",
            "Gov't Revenue",
            "Policy Cost",
            "Policy Cost (\\% of GDP)",
            "Policy Cost (\\% of Gov't Revenue)",
        ]
    ]

    if save_as:
        new_df.to_latex(
            save_as + ".tex",
            index=False,
            float_format="%.2f",
            escape=False,
        )
    return new_df


def get_table_oecd_plus_china(policy_cost, save_as):
    df = pd.read_csv("learn/auxiliary_data.csv")
    df["GDP"] = df["OECD_nominal_GDP_2023_billions"] + (
        df["China_nominal_GDP_2023_billions"]
    )
    df["Gov't Revenue"] = (
        df["OECD_nominal_GDP_2023_billions"]
        * df["OECD_govt_revenue_percentage_GDP_2023"]
        / 100
    ) + (
        df["China_nominal_GDP_2023_billions"]
        * df["China_govt_revenue_percentage_GDP_2023"]
        / 100
    )

    df["Policy Cost"] = policy_cost
    df["Policy Cost (\\% of GDP)"] = df["Policy Cost"] * 100 / df["GDP"]
    df["Policy Cost (\\% of Gov't Revenue)"] = (
        df["Policy Cost"] * 100 / df["Gov't Revenue"]
    )

    new_df = df[
        [
            "GDP",
            "Gov't Revenue",
            "Policy Cost",
            "Policy Cost (\\% of GDP)",
            "Policy Cost (\\% of Gov't Revenue)",
        ]
    ]
    if save_as:
        new_df.to_latex(
            save_as + ".tex", index=False, float_format="%.2f", escape=False
        )
    return new_df


def get_table_survey_info(countries, save_as):
    df = pd.read_csv(COUNTRY_AUX_DATA_CSV)
    df = df[df["country"].isin(countries)]

    sample_sizes = []
    covariate_dimensions = []

    for country in countries:
        n, d = get_data_dimension(country)
        sample_sizes.append(n)
        covariate_dimensions.append(d)

    new_df = pd.DataFrame(
        {
            "country": countries,
            "sample_size": sample_sizes,
            "covariate_dimension": covariate_dimensions,
        }
    )

    df = df.merge(new_df, on="country", how="left")

    columns = [
        "country",
        "survey_name",
        "survey_year",
        "sample_size",
        "covariate_dimension",
        "survey_poverty_rate",
        "wb_poverty_rate_survey_year",
    ]
    df["survey_year"] = df["survey_year"].astype(int)
    df = df[columns]
    df["survey_poverty_rate"] *= 100
    df["wb_poverty_rate_survey_year"] *= 100
    df.sort_values(by=["country"], inplace=True)
    df.rename(
        columns={
            "country": "Country",
            "sample_size": "Sample Size",
            "covariate_dimension": "Covariate Dimension",
            "survey_poverty_rate": "Survey Poverty Rate",
            "wb_poverty_rate_survey_year": "WB Poverty Rate (Survey Year)",
            "survey_name": "Survey Name",
            "survey_year": "Survey Year",
        },
        inplace=True,
    )
    df.to_latex(
        save_as + ".tex",
        index=False,
        float_format="%.2f",
        escape=False,
        formatters={"Country": get_country_name},
    )


def get_table_wpc(countries, save_as):
    df = preprocess_wpc_data(countries)
    df.rename(
        columns={
            "country": "Country",
            "wpc_poverty_rate": "Poverty Rate",
            "wpc_share_world_poor": "Share of World's Poor",
        },
        inplace=True,
    )
    df.to_latex(
        save_as + ".tex",
        index=False,
        float_format="%.2f",
        escape=False,
        formatters={"Country": get_country_name},
    )


def get_table_out_of_sample_rmse(countries, save_as):
    res = []
    for country in countries:
        r2 = get_out_of_sample_rmse(country)
        res.append({"country": country, "out_of_sample_rmse": r2})

    df = pd.DataFrame(res)
    df.sort_values(by=["country"])
    df.rename(
        columns={
            "out_of_sample_rmse": "Evaluation Set RMSE",
            "country": "Country",
        },
        inplace=True,
    )
    df.sort_values(by=["Country"], inplace=True)
    df.to_latex(
        save_as + ".tex",
        index=False,
        float_format="%.2f",
        escape=False,
        formatters={"Country": get_country_name},
    )


def get_table_diff_between_ubi_and_targeting(countries, povertyline, year, save_as):
    cont_gap_results = [
        CountryMethodPovertyResults(
            country,
            "continuous_gap",
            geo_extrapolation=True,
            povertyline=povertyline,
            year=year,
        )
        for country in countries
    ]
    ubi_results = [
        CountryMethodPovertyResults(
            country, "ubi", geo_extrapolation=True, povertyline=povertyline, year=year
        )
        for country in countries
    ]

    res = []
    for i, country in enumerate(countries):
        ubi_cost = ubi_results[i].rate_to_cost_interpolator(POVERTY_RATE_TARGET).item()
        targeting_cost = (
            cont_gap_results[i].rate_to_cost_interpolator(POVERTY_RATE_TARGET).item()
        )
        res.append(
            {
                "country": country,
                "continuous_gap_cost": targeting_cost,
                "ubi_cost": ubi_cost,
                "difference_between_ubi_and_targeting": ubi_cost - targeting_cost,
            }
        )

    df = pd.DataFrame(res)
    df.rename(
        columns={
            "difference_between_ubi_and_targeting": "Cost Difference Between UBI and Targeting",
            "continuous_gap_cost": "Targeting Cost",
            "ubi_cost": "UBI Cost",
            "country": "Country",
        },
        inplace=True,
    )
    df = df[
        [
            "Country",
            "UBI Cost",
            "Targeting Cost",
            "Cost Difference Between UBI and Targeting",
        ]
    ]
    df.to_latex(
        save_as + ".tex",
        index=False,
        float_format="%.2f",
        escape=False,
        formatters={"Country": get_country_name},
    )


def plot_bar_chart_ubi_ratio(countries, povertyline, year, save_as):
    cont_gap_results = [
        CountryMethodPovertyResults(
            country,
            method="continuous_gap",
            geo_extrapolation=True,
            povertyline=povertyline,
            year=year,
        )
        for country in countries
    ]
    ubi_results = [
        CountryMethodPovertyResults(
            country,
            method="ubi",
            geo_extrapolation=True,
            povertyline=povertyline,
            year=year,
        )
        for country in countries
    ]

    res = []
    for i, country in enumerate(countries):
        ubi_cost = ubi_results[i].rate_to_cost_interpolator(POVERTY_RATE_TARGET)
        targeting_cost = cont_gap_results[i].rate_to_cost_interpolator(
            POVERTY_RATE_TARGET
        )
        res.append(
            {
                "country": country,
                "ratio_of_ubi_and_targeting": ubi_cost / targeting_cost,
            }
        )

    df = pd.DataFrame(res)
    df.sort_values(by=["ratio_of_ubi_and_targeting"], ascending=False, inplace=True)
    fontsize = 30
    plt.figure(figsize=(30, 8))
    plt.bar(
        [get_country_name(country) for country in df["country"]],
        df["ratio_of_ubi_and_targeting"],
        zorder=3,
    )
    plt.grid(axis="y", zorder=0)

    plt.xlabel("Country", fontsize=fontsize)
    plt.ylabel("Cost Ratio", fontsize=fontsize)
    # plt.suptitle("Cost Ratio between UBI (Variable) and Gap Targeting (Continuous) vs. Country", fontsize=fontsize)
    plt.xticks(rotation=90, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.savefig("{}.pdf".format(save_as), bbox_inches="tight")


def plot_bar_chart_oracle_ratio(countries, povertyline, year, save_as):
    cont_gap_results = [
        CountryMethodPovertyResults(
            country,
            method="continuous_gap",
            geo_extrapolation=True,
            povertyline=povertyline,
            year=year,
        )
        for country in countries
    ]
    oracle_results = [
        CountryMethodPovertyResults(
            country,
            method="oracle_gap",
            geo_extrapolation=True,
            povertyline=povertyline,
            year=year,
        )
        for country in countries
    ]

    res = []
    for i, country in enumerate(countries):
        oracle_cost = oracle_results[i].rate_to_cost_interpolator(POVERTY_RATE_TARGET)
        targeting_cost = cont_gap_results[i].rate_to_cost_interpolator(
            POVERTY_RATE_TARGET
        )
        res.append(
            {
                "country": country,
                "ratio_of_oracle_and_targeting": targeting_cost / oracle_cost,
            }
        )

    df = pd.DataFrame(res)
    df.sort_values(by=["ratio_of_oracle_and_targeting"], ascending=False, inplace=True)
    fontsize = 30
    plt.figure(figsize=(30, 8))
    plt.bar(
        [get_country_name(country) for country in df["country"]],
        df["ratio_of_oracle_and_targeting"],
        zorder=3,
    )
    plt.grid(axis="y", zorder=0)
    plt.xlabel("Country", fontsize=fontsize)
    plt.ylabel("Cost Ratio", fontsize=fontsize)
    # plt.suptitle("Cost Ratio between UBI (Variable) and Gap Targeting (Continuous) vs. Country", fontsize=fontsize)
    plt.xticks(rotation=90, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.savefig("{}.pdf".format(save_as), bbox_inches="tight")


def get_extrapolation(countries, povertyline, year, save_as=None):
    in_sample_countries = countries
    df = pd.read_csv(COUNTRY_AUX_DATA_CSV)
    oracle_results = [
        CountryMethodPovertyResults(
            country,
            "oracle_gap",
            geo_extrapolation=True,
            povertyline=povertyline,
            year=year,
        )
        for country in in_sample_countries
    ]
    cont_gap_results = [
        CountryMethodPovertyResults(
            country,
            "continuous_gap",
            geo_extrapolation=True,
            povertyline=povertyline,
            year=year,
        )
        for country in in_sample_countries
    ]

    in_sample_costs = []
    in_sample_country_ratios = []
    for i, country in enumerate(in_sample_countries):
        in_sample_country_ratios.append(
            cont_gap_results[i].rate_to_cost_interpolator(POVERTY_RATE_TARGET)
            / oracle_results[i].rate_to_cost_interpolator(POVERTY_RATE_TARGET)
        )
        in_sample_costs.append(
            cont_gap_results[i].rate_to_cost_interpolator(POVERTY_RATE_TARGET)
        )
        print(country, "in-sample cost:", in_sample_costs[-1])

    X = []
    for i, country in enumerate(in_sample_countries):
        X.append(
            [
                oracle_results[i].initial_gap,
                oracle_results[i].initial_rate,
            ]
        )
    X_test = []
    out_of_sample_countries = df["country"].unique().tolist()
    for country in in_sample_countries:
        if country in out_of_sample_countries:
            out_of_sample_countries.remove(country)

    dropped_countries = []
    cols = [
        "oracle_gap",
        "oracle_rate",
        "PPP_conversion_factor_{}".format(year),
        "market_exchange_rate_{}".format(year),
        "total_population_2023",
    ]
    for country in out_of_sample_countries:
        if df[df["country"] == country][cols].isna().sum().sum() > 0:
            dropped_countries.append(country)
        elif df[df["country"] == country]["oracle_gap"].item() < 1:
            dropped_countries.append(country)

    for country in dropped_countries:
        out_of_sample_countries.remove(country)

    for country in out_of_sample_countries:
        X_test.append(
            [
                df[df["country"] == country]["oracle_gap"].item(),
                df[df["country"] == country]["oracle_rate"].item(),
            ]
        )

    X = np.array(X).reshape(len(X), 2)
    y = np.array(in_sample_country_ratios).reshape(len(X), 1)
    model = LinearRegression(fit_intercept=True)
    model.fit(X, y)

    X_test = np.array(X_test)
    pred_ratio = model.predict(X_test)

    costs = []
    secondary_aux_data = pd.read_csv(SECONDARY_AUX_DATA_CSV)
    inflation_adjustment = secondary_aux_data[
        "conversion_factor_nominal_USD_{}_to_2023".format(year)
    ].values[0]

    for i, country in enumerate(out_of_sample_countries):
        oracle_gap_index = df[df["country"] == country]["oracle_gap"].values[0].item()
        ppp_exchange_rate = (
            df[df["country"] == country]["PPP_conversion_factor_{}".format(year)]
            .values[0]
            .item()
        )
        market_exchange_rate = (
            df[df["country"] == country]["market_exchange_rate_{}".format(year)]
            .values[0]
            .item()
        )
        population = (
            df[df["country"] == country]["total_population_2023"].values[0].item()
        )
        oracle_gap = (
            oracle_gap_index
            / 100
            * povertyline
            * 365
            * inflation_adjustment
            * (ppp_exchange_rate / market_exchange_rate)
            * population
            / 1000000000
        )
        cost_for_country = pred_ratio[i] * oracle_gap
        costs.append(
            {
                "Country": country,
                "Poverty Gap Index": oracle_gap_index,
                "Predicted Feasible/Oracle Ratio": pred_ratio[i].item(),
                "Extrapolated Policy Cost": float(cost_for_country),
            }
        )
    costs = pd.DataFrame(costs)
    costs = costs.sort_values(by=["Country"])
    if save_as is not None:
        costs.to_latex(
            save_as + ".tex",
            index=False,
            float_format="%.2f",
            escape=False,
            formatters={"Country": get_country_name},
        )
    total_out_of_sample_cost = costs["Extrapolated Policy Cost"].sum()
    total_in_sample_cost = sum(in_sample_costs)
    total_cost = total_out_of_sample_cost + total_in_sample_cost
    print("Total In-Sample Cost: ", total_in_sample_cost)
    print("Total Out-of-Sample Cost: ", total_out_of_sample_cost)
    print("Total Cost: ", total_cost)
    print("Excluded Countries:", dropped_countries)
    return total_cost, total_in_sample_cost, total_out_of_sample_cost, dropped_countries


def make_string_country_list(l):
    if len(l) == 0:
        return ""
    elif len(l) == 1:
        return get_country_name(l[0])
    else:
        return (
            ", ".join([get_country_name(c) for c in l[:-1]])
            + ", and "
            + get_country_name(l[-1])
        )


def make_macro_file(countries, povertyline, year, save_as):
    countries = sorted(countries)
    all_countries_string = make_string_country_list(countries)

    wpc_data = preprocess_wpc_data(countries)
    total_world_poor = wpc_data["wpc_share_world_poor"].sum()
    malawi_world_poor = wpc_data[wpc_data["country"] == "malawi"][
        "wpc_share_world_poor"
    ].values[0]

    df = pd.read_csv(COUNTRY_AUX_DATA_CSV)
    population_total = df[df.country.isin(countries)][
        "total_population_survey_year"
    ].sum()

    oracle_results = [
        CountryMethodPovertyResults(
            country,
            "oracle_gap",
            geo_extrapolation=True,
            povertyline=povertyline,
            year=year,
        )
        for country in countries
    ]
    weights = (
        np.array(
            [
                df[df["country"] == country]["total_population_survey_year"].values[0]
                for country in countries
            ]
        )
        / population_total
    )
    pov_rates = [oracle_results[i].initial_rate for i, country in enumerate(countries)]
    initial_poverty_rate = sum(
        pov_rates[i] * weights[i] for i, country in enumerate(countries)
    )
    min_pov_rate = min(pov_rates)
    max_pov_rate = max(pov_rates)
    arg_min_pov_rate = np.argmin(pov_rates)
    arg_max_pov_rate = np.argmax(pov_rates)
    min_country = get_country_name(countries[arg_min_pov_rate])
    max_country = get_country_name(countries[arg_max_pov_rate])
    initial_poverty_gap = sum(
        [
            oracle_results[i].initial_gap * weights[i]
            for i, country in enumerate(countries)
        ]
    )

    # df["ODA / GDP"] = df["ODA"] / df["GDP_billions_survey_year"]
    # sample_oda = df[df["country"].isin(countries)]["ODA / GDP"].mean() * 100

    cont_gap_results = [
        CountryMethodPovertyResults(
            country,
            "continuous_gap",
            geo_extrapolation=True,
            povertyline=povertyline,
            year=year,
        )
        for country in countries
    ]

    policy_costs = np.array(
        [
            cont_gap_results[i].rate_to_cost_interpolator(POVERTY_RATE_TARGET)
            for i, country in enumerate(countries)
        ]
    ).flatten()
    gdp = np.array(
        [
            df["GDP_survey_year"][df["country"] == country].values[0]
            for country in countries
        ]
    ).flatten()
    sample_policy_cost = np.mean(policy_costs / gdp) * 100

    methods = ["continuous_gap", "binary_gap", "oracle_gap", "pmt", "ubi"]
    cost = {}
    for method in methods:
        method_results = AggregatePovertyResults(
            countries=countries,
            method=method,
            geo_extrapolation=True,
            povertyline=povertyline,
            year=year,
        )
        if method == "ubi":
            cost[method + "_variable"] = (
                method_results.aggregate_interpolator_rate_to_cost(
                    POVERTY_RATE_TARGET
                ).item()
            )
        else:
            cost[method] = method_results.aggregate_interpolator_rate_to_cost(
                POVERTY_RATE_TARGET
            ).item()

    cost["ubi"] = povertyline * sum(
        [oracle_results[i].conversion_factor for i in range(len(countries))]
    )
    print("HEADLINE COST", cost["continuous_gap"])

    malawi_costs = {}
    for method in methods:
        malawi_results = CountryMethodPovertyResults(
            "malawi", method, geo_extrapolation=True, povertyline=povertyline, year=year
        )
        if method == "ubi":
            malawi_costs[method + "_variable"] = (
                malawi_results.rate_to_cost_interpolator(POVERTY_RATE_TARGET).item()
            )
        else:
            malawi_costs[method] = malawi_results.rate_to_cost_interpolator(
                POVERTY_RATE_TARGET
            ).item()

    malawi_costs["ubi"] = povertyline * malawi_results.conversion_factor

    ratios = []
    for i, country in enumerate(countries):
        ratios.append(
            cont_gap_results[i].rate_to_cost_interpolator(POVERTY_RATE_TARGET).item()
            / oracle_results[i].rate_to_cost_interpolator(POVERTY_RATE_TARGET).item()
        )
    min_ratio = min(ratios)
    max_ratio = max(ratios)

    # total_cost, _, out_of_sample_cost, dropped_countries = get_extrapolation(
    #     countries, save_as=None
    # )
    # oecd_df = get_table_oecd(total_cost, save_as=None)
    # oecd_gdp_percent = oecd_df["Policy Cost (\\% of GDP)"].values[0].item()
    # oecd_revenue_percent = (
    #     oecd_df["Policy Cost (\\% of Gov't Revenue)"].values[0].item()
    # )

    # oecd_plus_china_df = get_table_oecd_plus_china(total_cost, save_as=None)
    # oecd_plus_china_gdp_percent = (
    #     oecd_plus_china_df["Policy Cost (\\% of GDP)"].values[0].item()
    # )

    # oecd_plus_china_revenue_percent = (
    #     oecd_plus_china_df["Policy Cost (\\% of GDP)"].values[0].item()
    # )

    # dropped_countries_string = make_string_country_list(sorted(dropped_countries))

    malawi_n, malawi_d = get_data_dimension("malawi")
    data_dimension = [get_data_dimension(country)[1] for country in countries]
    min_d = min(data_dimension)
    max_d = max(data_dimension)

    with open(save_as + ".tex", "w") as f:
        f.write("\\newcommand{\\sampleNumCountries}" + f"{{{len(countries)}}}\n")
        f.write("\\newcommand{\\sampleCountries}" + f"{{{all_countries_string}}}\n")
        f.write(
            "\\newcommand{\\sampleShareWorldsPoorExact}" + f"{{{total_world_poor}}}\n"
        )
        f.write(
            "\\newcommand{\\sampleShareWorldsPoor}" + f"{{{total_world_poor:.0f}}}\n"
        )
        f.write("\\newcommand{\\sampleGap}" + f"{{{initial_poverty_gap:.0f}}}\n")
        f.write("\\newcommand{\\sampleRate}" + f"{{{initial_poverty_rate:.0f}}}\n")
        f.write("\\newcommand{\\sampleMinRate}" + f"{{{min_pov_rate:.0f}}}\n")
        f.write("\\newcommand{\\sampleMaxRate}" + f"{{{max_pov_rate:.0f}}}\n")
        f.write("\\newcommand{\\sampleMinRateCountry}" + f"{{{min_country}}}\n")
        f.write("\\newcommand{\\sampleMaxRateCountry}" + f"{{{max_country}}}\n")
        f.write(
            "\\newcommand{\\samplePolicyCostPercent}"
            + f"{{{sample_policy_cost:.0f}}}\n"
        )

        f.write(
            "\\newcommand{\\headlineUBI}" + "{{{}}}\n".format(round(cost["ubi"], 1))
        )
        f.write(
            "\\newcommand{\\headlinePMT}" + "{{{}}}\n".format(round(cost["pmt"], 1))
        )
        f.write(
            "\\newcommand{\\headlineGap}"
            + "{{{}}}\n".format(round(cost["continuous_gap"], 1))
        )
        f.write(
            "\\newcommand{\\headlineOracle}"
            + "{{{}}}\n".format(round(cost["oracle_gap"], 1))
        )
        f.write(
            "\\newcommand{\\headlineBinaryGap}"
            + "{{{}}}\n".format(round(cost["binary_gap"], 1))
        )
        f.write(
            "\\newcommand{\\headlineUBIVariable}"
            + "{{{}}}\n".format(round(cost["ubi_variable"], 1))
        )

        f.write(
            "\\newcommand{\\headlineGapUBIPercent}"
            + "{{{}}}\n".format(round((cost["continuous_gap"] / cost["ubi"]) * 100))
        )
        f.write(
            "\\newcommand{\\headlineGapUBIVariablePercent}"
            + "{{{}}}\n".format(
                round((cost["continuous_gap"] / cost["ubi_variable"]) * 100)
            )
        )
        f.write(
            "\\newcommand{\\headlineGapPMTPercent}"
            + "{{{}}}\n".format(round((cost["continuous_gap"] / cost["pmt"]) * 100))
        )
        f.write(
            "\\newcommand{\\headlineGapOracleRatio}"
            + "{{{}}}\n".format(round((cost["continuous_gap"] / cost["oracle_gap"]), 2))
        )
        f.write(
            "\\newcommand{\\headlineBinaryContPercentIncrease}"
            + "{{{}}}\n".format(
                (
                    round(
                        (cost["binary_gap"] - cost["continuous_gap"])
                        * 100
                        / cost["continuous_gap"]
                    )
                ),
                0,
            )
        )

        # f.write(
        #     "\\newcommand{\\extrapolationCost}"
        #     + "{{{}}}\n".format(round(total_cost, 1))
        # )
        # f.write(
        #     "\\newcommand{\\extrapolationOECDGDPPercent}"
        #     + "{{{}}}\n".format(round(oecd_gdp_percent, 2))
        # )
        # f.write(
        #     "\\newcommand{\\extrapolationOECDGovtRevPercent}"
        #     + "{{{}}}\n".format(round(oecd_revenue_percent, 2))
        # )
        # f.write(
        #     "\\newcommand{\\extrapolationOECDPlusChinaGDPPercent}"
        #     + "{{{}}}\n".format(round(oecd_plus_china_gdp_percent, 2))
        # )
        # f.write(
        #     "\\newcommand{\\extrapolationOECDPlusChinaGovtRevPercent}"
        #     + "{{{}}}\n".format(round(oecd_plus_china_revenue_percent, 2))
        # )
        # f.write(
        #     "\\newcommand{\\extrapolationOutOfSampleCost}"
        #     + "{{{}}}\n".format(round(out_of_sample_cost, 0))
        # )
        # f.write(
        #     "\\newcommand{\\extrapolationDroppedCountries}"
        #     + "{{{}}}\n".format(dropped_countries_string)
        # )

        f.write(
            "\\newcommand{\\malawiShareWorldsPoor}"
            + "{{{}}}\n".format(malawi_world_poor)
        )
        f.write(
            "\\newcommand{\\malawiUBIVariableAmount}"
            + "{{{}}}\n".format(
                round(
                    malawi_costs["ubi_variable"] / malawi_results.conversion_factor, 2
                )
            )
        )
        f.write(
            "\\newcommand{\\malawiGapOracleRatio}"
            + "{{{}}}\n".format(
                round(malawi_costs["continuous_gap"] / malawi_costs["oracle_gap"], 1)
            )
        )
        f.write(
            "\\newcommand{\\malawiGapUBIPercent}"
            + "{{{}}}\n".format(
                round((malawi_costs["continuous_gap"] * 100 / malawi_costs["ubi"])), 1
            )
        )
        f.write(
            "\\newcommand{\\malawiGapUBIVariablePercent}"
            + "{{{}}}\n".format(
                round(
                    (
                        malawi_costs["continuous_gap"]
                        * 100
                        / malawi_costs["ubi_variable"]
                    )
                ),
                1,
            )
        )
        f.write(
            "\\newcommand{\\malawiGapPMTPercent}"
            + "{{{}}}\n".format(
                round(malawi_costs["continuous_gap"] * 100 / malawi_costs["pmt"], 1)
            )
        )
        f.write(
            "\\newcommand{\\malawiBinaryContPercentIncrease}"
            + "{{{}}}\n".format(
                round(
                    (malawi_costs["binary_gap"] - malawi_costs["continuous_gap"])
                    * 100
                    / malawi_costs["continuous_gap"],
                    0,
                )
            )
        )

        f.write(
            "\\newcommand{\\oracleFeasibleRatioMin}"
            + "{{{}}}\n".format(round(min_ratio, 1))
        )
        f.write(
            "\\newcommand{\\oracleFeasibleRatioMax}"
            + "{{{}}}\n".format(round(max_ratio, 1))
        )
        f.write(
            "\\newcommand{\\malawiCovariateDimension}" + "{{{}}}\n".format(malawi_d)
        )
        f.write("\\newcommand{\\malawiSampleSize}" + "{{{}}}\n".format(malawi_n))
        f.write("\\newcommand{\\minDimension}" + "{{{}}}\n".format(min_d))
        f.write("\\newcommand{\\maxDimension}" + "{{{}}}\n".format(max_d))
