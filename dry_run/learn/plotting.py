import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from learn.aggregation import (
    METHODS,
    AggregatePovertyResults,
    CountryMethodPovertyResults,
    preprocess_country_aux_data,
    SECONDARY_AUX_DATA_CSV,
    preprocess_wpc_data,
)


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
    country,
    method_list,
    geo_extrapolation,
    povertyline,
    year,
    save_as,
    ubi_on=True,
    poverty_gap_on=True,
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
    if ubi_on:
        ax[1].plot(
            np.linspace(0.0, results[i].initial_gap_index),
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
    if poverty_gap_on:
        poverty_gap = results[i].get_poverty_gap()
        ax[1].plot(
            np.linspace(0.0, results[i].initial_gap_index),
            np.ones(50) * poverty_gap,
            linestyle="--",
            color=METHODS["oracle_gap"]["color"],
            label="UBI ${}".format(povertyline),
        )
        ax[0].plot(
            np.linspace(0.0, results[i].initial_rate),
            np.ones(50) * poverty_gap,
            linestyle="--",
            color=METHODS["oracle_gap"]["color"],
            label="UBI ${}".format(povertyline),
        )

    for i, method in enumerate(method_list):
        dic = methods[method]
        df = results[i]._load_data(method)

        rates = [oracle_results.initial_rate] + list(
            df["post_transfer_poverty_rate"] * 100
        )
        gaps = [oracle_results.initial_gap_index] + list(
            df["post_transfer_poverty_gap"] * 100 / povertyline
        )
        costs = [0.0] + list(
            df["policy_cost_per_capita"] * results[0].conversion_factor
        )

        if method != "oracle_gap":
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
    countries,
    method_list,
    geo_extrapolation,
    povertyline,
    year,
    save_as,
    ubi_on=True,
    poverty_gap_on=True,
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
    initial_gap_index, initial_rate = (
        oracle_results.get_initial_aggregate_gap_index_and_rate()
    )

    if ubi_on:
        ubi_cost = results[0].get_aggregate_ubi_cost()
        ax[0].plot(
            np.linspace(0.0, initial_rate),
            np.ones(50) * ubi_cost,
            linestyle="--",
            color=METHODS["ubi"]["color"],
            label="UBI ${}".format(povertyline),
        )
        ax[1].plot(
            np.linspace(0.0, initial_gap_index),
            np.ones(50) * ubi_cost,
            linestyle="--",
            color=METHODS["ubi"]["color"],
            label="UBI ${}".format(povertyline),
        )
    if poverty_gap_on:
        aggregate_poverty_gap = results[0].get_aggregate_poverty_gap()
        ax[0].plot(
            np.linspace(0.0, initial_rate),
            np.ones(50) * aggregate_poverty_gap,
            linestyle="--",
            color=METHODS["oracle_gap"]["color"],
            label="Aggregate Poverty Gap",
        )

        ax[1].plot(
            np.linspace(0.0, initial_gap_index),
            np.ones(50) * aggregate_poverty_gap,
            linestyle="--",
            color=METHODS["oracle_gap"]["color"],
            label="Aggregate Poverty Gap",
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
    df = preprocess_country_aux_data()
    second_df = pd.read_csv("learn/inflation_adjustment.csv")
    survey_year = int(df[df["country"] == country]["survey_year"].values[0])
    inflation_adjustment = second_df[second_df.survey_year == survey_year][
        "inflation_adjustment_to_2023"
    ].values[0]
    amt = amt * (1 / inflation_adjustment)
    return amt


def plot_bar_chart_policy_amt_as_percent_of_gdp(
    countries, geo_extrapolation, povertyline, year, globalPovertyRate, save_as
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
    main_national_target = get_national_poverty_rate_target(globalPovertyRate)
    for i in range(len(countries)):
        amt = results[i].rate_to_cost_interpolator(main_national_target).item()
        amts.append(amt)

    amts_survey_year = [
        convert_nominal_2023_to_nominal_survey_year(amt, country)
        for amt, country in zip(amts, countries)
    ]

    df = preprocess_country_aux_data()
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


def get_national_poverty_rate_target(global_poverty_rate_target):
    df = preprocess_wpc_data()

    def global_poverty_rate(national_ceiling):
        national_poverty_rates = (
            np.minimum(df["wpc_poverty_rate"], national_ceiling).to_numpy() / 100
        )
        global_poverty_rate = (
            national_poverty_rates * df["total_population"]
        ).sum() / df["total_population"].sum()
        return global_poverty_rate.item() * 100

    ceilings = np.linspace(0, df["wpc_poverty_rate"].max(), 100)
    global_poverty_rates = [global_poverty_rate(c) for c in ceilings]
    national_poverty_rate_target = interp1d(global_poverty_rates, ceilings)(
        global_poverty_rate_target
    )
    return national_poverty_rate_target


def get_global_poverty_rate_target(national_poverty_rate_target):
    df = preprocess_wpc_data()

    def global_poverty_rate(national_ceiling):
        national_poverty_rates = (
            np.minimum(df["wpc_poverty_rate"], national_ceiling).to_numpy() / 100
        )
        global_poverty_rate = (
            national_poverty_rates * df["total_population"]
        ).sum() / df["total_population"].sum()
        return global_poverty_rate.item() * 100

    ceilings = np.linspace(0, df["wpc_poverty_rate"].max(), 100)
    global_poverty_rates = [global_poverty_rate(c) for c in ceilings]
    global_poverty_rate_target = interp1d(ceilings, global_poverty_rates)(
        national_poverty_rate_target
    )
    return global_poverty_rate_target


def get_table_policy_cost_gdp(countries, povertyline, year, globalPovertyRate, save_as):
    df = preprocess_country_aux_data()

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
    national_target = get_national_poverty_rate_target(globalPovertyRate)
    for result in results:
        amt = result.rate_to_cost_interpolator(national_target).item()
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


def get_macros_relative_cost(policy_cost):
    df = pd.read_csv(SECONDARY_AUX_DATA_CSV)
    percentage_oecd_gdp = 100 * policy_cost / df["OECD_GDP"].item()
    percentage_oecd_plus_china_gdp = (
        100 * policy_cost / (df["OECD_GDP"] + df["China_GDP"]).item()
    )
    percentage_oecd_govt_revenue = (
        100
        * policy_cost
        / (df["OECD_GDP"].item() * df["OECD_govt_revenue_percentage"] / 100).item()
    )
    percentage_oecd_plus_china_govt_revenue = (
        100
        * policy_cost
        / (
            df["OECD_GDP"].item() * df["OECD_govt_revenue_percentage"] / 100
            + df["China_GDP"].item() * df["China_govt_revenue_percentage"] / 100
        ).item()
    )
    percentage_global_gdp = 100 * policy_cost / df["Global_GDP"].item()
    return (
        percentage_oecd_gdp,
        percentage_oecd_plus_china_gdp,
        percentage_oecd_govt_revenue,
        percentage_oecd_plus_china_govt_revenue,
        percentage_global_gdp,
    )


def get_table_survey_info(countries, save_as):
    df = preprocess_country_aux_data()
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
    df = df[["country", "wpc_poverty_rate", "wpc_share_world_poor"]]
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


def plot_bar_chart_ubi_ratio(countries, povertyline, year, globalPovertyRate, save_as):
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
    main_national_poverty_rate_target = get_national_poverty_rate_target(
        globalPovertyRate
    )
    for i, country in enumerate(countries):
        ubi_cost = ubi_results[i].rate_to_cost_interpolator(
            main_national_poverty_rate_target
        )
        targeting_cost = cont_gap_results[i].rate_to_cost_interpolator(
            main_national_poverty_rate_target
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


def plot_bar_chart_oracle_ratio(
    countries, povertyline, year, globalPovertyRate, save_as
):
    national_poverty_rate_target = get_national_poverty_rate_target(globalPovertyRate)
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
        oracle_cost = oracle_results[i].rate_to_cost_interpolator(
            national_poverty_rate_target
        )
        targeting_cost = cont_gap_results[i].rate_to_cost_interpolator(
            national_poverty_rate_target
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


def get_extrapolation(countries, povertyline, year, globalPovertyRate, save_as=None):
    in_sample_countries = countries
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
    main_national_target = get_national_poverty_rate_target(globalPovertyRate)
    for i, country in enumerate(in_sample_countries):
        in_sample_country_ratios.append(
            cont_gap_results[i].rate_to_cost_interpolator(main_national_target)
            / oracle_results[i].rate_to_cost_interpolator(main_national_target)
        )
        in_sample_costs.append(
            cont_gap_results[i].rate_to_cost_interpolator(main_national_target)
        )
        print(country, "in-sample cost:", in_sample_costs[-1])

    X = []
    for i, country in enumerate(in_sample_countries):
        X.append(
            [
                oracle_results[i].initial_rate,
            ]
        )
    X_test = []
    df = preprocess_country_aux_data()
    out_of_sample_countries = df["country"].unique().tolist()
    for country in in_sample_countries:
        if country in out_of_sample_countries:
            out_of_sample_countries.remove(country)
    population_df = pd.read_csv("learn/population.csv")
    df = df.merge(
        population_df[["country_code", "total_population"]],
        on="country_code",
        how="left",
    )

    dropped_countries = []
    for country in out_of_sample_countries:
        if np.isnan(df[df["country"] == country]["wb_poverty_rate_most_recent"].item()):
            print(country, "poverty rate missing")
            dropped_countries.append(country)
        elif np.isnan(
            df[df["country"] == country]["wb_poverty_gap_index_most_recent"].item()
        ):
            print(country, "poverty gap missing")
            dropped_countries.append(country)
        elif np.isnan(df[df["country"] == country]["total_population"].item()):
            print(country, "population missing")
            dropped_countries.append(country)
        elif np.isnan(
            df[df["country"] == country]["PPP_conversion_factor_{}".format(year)].item()
        ):
            print(country, "PPP missing")
            dropped_countries.append(country)
        elif np.isnan(
            df[df["country"] == country]["market_exchange_rate_{}".format(year)].item()
        ):
            print(country, "market exchange rate missing")
            dropped_countries.append(country)

    for country in dropped_countries:
        out_of_sample_countries.remove(country)

    for country in out_of_sample_countries:
        X_test.append(
            [
                df[df["country"] == country]["wb_poverty_rate_most_recent"].item()
                * 100,
            ]
        )

    X = np.array(X).reshape(len(X), 1)
    y = np.array(in_sample_country_ratios).reshape(len(X), 1)
    model = LinearRegression(fit_intercept=True)
    model.fit(X, y)

    X_test = np.array(X_test).reshape(len(X_test), 1)
    pred_ratio = np.maximum(model.predict(X_test), 1)
    costs = []

    if year == 2021:
        inflation_adjustment = 1.14
    elif year == 2017:
        inflation_adjustment = 1.26

    for i, country in enumerate(out_of_sample_countries):
        most_recent_year = int(
            df[df["country"] == country]["wb_poverty_gap_index_most_recent_year"]
            .values[0]
            .item()
        )
        oracle_gap_index = (
            df[df["country"] == country]["wb_poverty_gap_index_most_recent"]
            .values[0]
            .item()
        )
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
        population = df[df["country"] == country]["total_population"].values[0].item()
        oracle_gap = (
            oracle_gap_index
            * povertyline
            * 365
            * inflation_adjustment
            * (ppp_exchange_rate / market_exchange_rate)
            * population
            / (10**9)
        )
        cost_for_country = pred_ratio[i] * oracle_gap
        costs.append(
            {
                "Country": country,
                "Poverty Gap Index Year": most_recent_year,
                "Poverty Gap Index": oracle_gap_index * 100,
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
    print("Dropped countries:", dropped_countries)
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


def get_macros_share_world_poor(countries):
    wpc_data = preprocess_wpc_data(countries)
    total_world_poor = wpc_data["wpc_share_world_poor"].sum()
    malawi_world_poor = wpc_data[wpc_data["country"] == "malawi"][
        "wpc_share_world_poor"
    ].values[0]
    return total_world_poor, malawi_world_poor


def get_macros_survey_info(countries):
    df = preprocess_country_aux_data()

    weights = np.array(
        [
            df[df["country"] == country]["total_population_survey_year"].values[0]
            for country in countries
        ]
    )
    weights = weights / weights.sum()

    pov_rates = np.array(
        [
            df[df["country"] == country]["survey_poverty_rate"].values[0] * 100
            for country in countries
        ]
    )
    pov_gaps = np.array(
        [
            df[df["country"] == country]["survey_poverty_gap_index"].values[0] * 100
            for country in countries
        ]
    )
    initial_pov_rate = np.sum(weights * pov_rates)
    initial_pov_gap = np.sum(weights * pov_gaps)

    min_pov_rate = min(pov_rates)
    max_pov_rate = max(pov_rates)
    arg_min_pov_rate = np.argmin(pov_rates)
    arg_max_pov_rate = np.argmax(pov_rates)
    min_country = get_country_name(countries[arg_min_pov_rate])
    max_country = get_country_name(countries[arg_max_pov_rate])

    return (
        initial_pov_rate,
        initial_pov_gap,
        min_pov_rate,
        max_pov_rate,
        min_country,
        max_country,
    )


def get_headline_numbers(countries, povertyline, year, national_poverty_rate_target):
    agg_results = []
    methods = ["continuous_gap", "binary_gap", "oracle_gap", "pmt", "ubi"]
    for method in methods:
        agg_results.append(
            AggregatePovertyResults(
                countries=countries,
                method=method,
                geo_extrapolation=True,
                povertyline=povertyline,
                year=year,
            )
        )

    cost = {}
    for i, method in enumerate(methods):
        if method == "ubi":
            cost[method + "_variable"] = (
                agg_results[i]
                .aggregate_interpolator_rate_to_cost(national_poverty_rate_target)
                .item()
            )
        elif method == "oracle_gap":
            cost[method] = (
                agg_results[i].aggregate_interpolator_rate_to_cost(0).item()
            )  # want oracle to correpond to global poverty gap
        else:
            cost[method] = (
                agg_results[i]
                .aggregate_interpolator_rate_to_cost(national_poverty_rate_target)
                .item()
            )

        cost["ubi"] = povertyline * sum(
            [
                agg_results[i].country_results[countries[j]].conversion_factor
                for j in range(len(countries))
            ]
        )
    return cost, agg_results


def get_macros_oracle_ratios(countries, povertyline, year, globalPovertyRate):
    ratios = []
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
    main_national_rate_target = get_national_poverty_rate_target(globalPovertyRate)
    for i, country in enumerate(countries):
        ratios.append(
            cont_gap_results[i]
            .rate_to_cost_interpolator(main_national_rate_target)
            .item()
            / oracle_results[i]
            .rate_to_cost_interpolator(main_national_rate_target)
            .item()
        )
    min_ratio = min(ratios)
    max_ratio = max(ratios)
    return min_ratio, max_ratio


def make_macro_file(
    countries, povertyline, year, nationalPovertyRate, globalPovertyRate, save_as
):
    countries = sorted(countries)
    all_countries_string = make_string_country_list(countries)

    # SHARE WORLD'S POOR METRICS
    total_world_poor, malawi_world_poor = get_macros_share_world_poor(countries)

    # POVERTY RATES AND GAPS OF THE SAMPLE
    (
        initial_pov_rate,
        initial_pov_gap,
        min_pov_rate,
        max_pov_rate,
        min_country,
        max_country,
    ) = get_macros_survey_info(countries)

    # GET HEADLINE NUMBERS FOR GLOBAL POVERTY RATE TARGET
    national_poverty_rate_for_global = get_national_poverty_rate_target(
        globalPovertyRate
    )
    global_cost, _ = get_headline_numbers(
        countries, povertyline, year, national_poverty_rate_for_global
    )

    # GET HEADLINE NUMBERS FOR NATIONAL POVERTY RATE TARGET
    national_cost, _ = get_headline_numbers(
        countries, povertyline, year, nationalPovertyRate
    )
    global_poverty_rate_for_national = get_global_poverty_rate_target(
        nationalPovertyRate
    )

    # GET MALAWI HEADLINE NUMBERS
    malawi_cost, agg_results = get_headline_numbers(
        ["malawi"], povertyline, year, nationalPovertyRate
    )
    conversion_factor_malawi = (
        agg_results[0].country_results["malawi"]._get_conversion_factor()
    )

    malawi_variable_amt = malawi_cost["ubi_variable"] / conversion_factor_malawi

    # GET ORACLE RATIOS
    min_ratio, max_ratio = get_macros_oracle_ratios(
        countries, povertyline, year, globalPovertyRate
    )

    # GET EXTRAPOLATION
    extrapolated_cost, _, total_cost_out_of_sample_costs, dropped_countries = (
        get_extrapolation(
            countries,
            povertyline,
            year,
            globalPovertyRate,
            save_as="exhibits/year={}/tables/appendix-table-4-extrapolation".format(
                year
            ),
        )
    )
    (
        percentage_oecd_gdp,
        percentage_oecd_plus_china_gdp,
        percentage_oecd_govt_revenue,
        percentage_oecd_plus_china_govt_revenue,
        percentage_global_gdp,
    ) = get_macros_relative_cost(extrapolated_cost)

    dropped_countries_string = make_string_country_list(dropped_countries)

    malawi_n, malawi_d = get_data_dimension("malawi")
    data_dimension = [get_data_dimension(country)[1] for country in countries]
    min_d = min(data_dimension)
    max_d = max(data_dimension)

    with open(save_as + ".tex", "w") as f:
        f.write("\\newcommand{\\sampleNumCountries}" + f"{{{len(countries)}}}\n")
        f.write("\\newcommand{\\sampleCountries}" + f"{{{all_countries_string}}}\n")
        f.write(
            "\\newcommand{\\sampleShareWorldsPoor}" + f"{{{total_world_poor:.0f}}}\n"
        )
        f.write("\\newcommand{\\sampleGap}" + f"{{{initial_pov_gap:.0f}}}\n")
        f.write("\\newcommand{\\sampleRate}" + f"{{{initial_pov_rate:.0f}}}\n")
        f.write("\\newcommand{\\sampleMinRate}" + f"{{{min_pov_rate:.0f}}}\n")
        f.write("\\newcommand{\\sampleMaxRate}" + f"{{{max_pov_rate:.0f}}}\n")
        f.write("\\newcommand{\\sampleMinRateCountry}" + f"{{{min_country}}}\n")
        f.write("\\newcommand{\\sampleMaxRateCountry}" + f"{{{max_country}}}\n")
        f.write("\\newcommand{\\nationalTarget}" + f"{{{int(nationalPovertyRate)}}}\n")
        f.write("\\newcommand{\\globalTarget}" + f"{{{int(globalPovertyRate)}}}\n")
        f.write(
            "\\newcommand{\\nationalPovertyRateForGlobalTarget}"
            + f"{{{national_poverty_rate_for_global:.1f}}}\n"
        )
        f.write(
            "\\newcommand{\\globalPovertyRateForNationalTarget}"
            + f"{{{global_poverty_rate_for_national:.1f}}}\n"
        )

        f.write(
            "\\newcommand{\\headlineUBINationalTarget}"
            + "{{{}}}\n".format(round(national_cost["ubi"], 1))
        )
        f.write(
            "\\newcommand{\\headlinePMTNationalTarget}"
            + "{{{}}}\n".format(round(national_cost["pmt"], 1))
        )
        f.write(
            "\\newcommand{\\headlineGapNationalTarget}"
            + "{{{}}}\n".format(round(national_cost["continuous_gap"], 1))
        )
        f.write(
            "\\newcommand{\\headlineOracleNationalTarget}"
            + "{{{}}}\n".format(round(national_cost["oracle_gap"], 1))
        )
        f.write(
            "\\newcommand{\\headlineBinaryGapNationalTarget}"
            + "{{{}}}\n".format(round(national_cost["binary_gap"], 1))
        )
        f.write(
            "\\newcommand{\\headlineUBIVariableNationalTarget}"
            + "{{{}}}\n".format(round(national_cost["ubi_variable"], 1))
        )

        f.write(
            "\\newcommand{\\headlineGapUBIPercentNationalTarget}"
            + "{{{}}}\n".format(
                round((national_cost["continuous_gap"] / national_cost["ubi"]) * 100)
            )
        )
        f.write(
            "\\newcommand{\\headlineGapUBIVariablePercentNationalTarget}"
            + "{{{}}}\n".format(
                round(
                    (national_cost["continuous_gap"] / national_cost["ubi_variable"])
                    * 100
                )
            )
        )
        f.write(
            "\\newcommand{\\headlineGapPMTPercentNationalTarget}"
            + "{{{}}}\n".format(
                round((national_cost["continuous_gap"] / national_cost["pmt"]) * 100)
            )
        )
        f.write(
            "\\newcommand{\\headlineGapOracleRatioNationalTarget}"
            + "{{{}}}\n".format(
                round(
                    (national_cost["continuous_gap"] / national_cost["oracle_gap"]), 1
                )
            )
        )
        f.write(
            "\\newcommand{\\headlineBinaryContPercentIncreaseNationalTarget}"
            + "{{{}}}\n".format(
                (
                    round(
                        (national_cost["binary_gap"] - national_cost["continuous_gap"])
                        * 100
                        / national_cost["continuous_gap"]
                    )
                ),
                0,
            )
        )

        f.write(
            "\\newcommand{\\headlineUBIGlobalTarget}"
            + "{{{}}}\n".format(round(global_cost["ubi"], 1))
        )
        f.write(
            "\\newcommand{\\headlinePMTGlobalTarget}"
            + "{{{}}}\n".format(round(global_cost["pmt"], 1))
        )
        f.write(
            "\\newcommand{\\headlineGapGlobalTarget}"
            + "{{{}}}\n".format(round(global_cost["continuous_gap"], 1))
        )
        f.write(
            "\\newcommand{\\headlineOracleGlobalTarget}"
            + "{{{}}}\n".format(round(global_cost["oracle_gap"], 1))
        )
        f.write(
            "\\newcommand{\\headlineBinaryGapGlobalTarget}"
            + "{{{}}}\n".format(round(global_cost["binary_gap"], 1))
        )
        f.write(
            "\\newcommand{\\headlineUBIVariableGlobalTarget}"
            + "{{{}}}\n".format(round(global_cost["ubi_variable"], 1))
        )

        f.write(
            "\\newcommand{\\headlineGapUBIPercentGlobalTarget}"
            + "{{{}}}\n".format(
                round((global_cost["continuous_gap"] / global_cost["ubi"]) * 100)
            )
        )
        f.write(
            "\\newcommand{\\headlineGapUBIVariablePercentGlobalTarget}"
            + "{{{}}}\n".format(
                round(
                    (global_cost["continuous_gap"] / global_cost["ubi_variable"]) * 100
                )
            )
        )
        f.write(
            "\\newcommand{\\headlineGapPMTPercentGlobalTarget}"
            + "{{{}}}\n".format(
                round((global_cost["continuous_gap"] / global_cost["pmt"]) * 100)
            )
        )
        f.write(
            "\\newcommand{\\headlineGapOracleRatioGlobalTarget}"
            + "{{{}}}\n".format(
                round((global_cost["continuous_gap"] / global_cost["oracle_gap"]), 2)
            )
        )
        f.write(
            "\\newcommand{\\headlineBinaryContPercentIncreaseGlobalTarget}"
            + "{{{}}}\n".format(
                (
                    round(
                        (global_cost["binary_gap"] - global_cost["continuous_gap"])
                        * 100
                        / global_cost["continuous_gap"]
                    )
                ),
                0,
            )
        )

        f.write(
            "\\newcommand{\\extrapolationCost}"
            + "{{{}}}\n".format(round(extrapolated_cost, 1))
        )
        f.write(
            "\\newcommand{\\extrapolationOECDGDPPercent}"
            + "{{{}}}\n".format(round(percentage_oecd_gdp, 2))
        )
        f.write(
            "\\newcommand{\\extrapolationOECDGovtRevPercent}"
            + "{{{}}}\n".format(round(percentage_oecd_govt_revenue, 2))
        )
        f.write(
            "\\newcommand{\\extrapolationOECDPlusChinaGDPPercent}"
            + "{{{}}}\n".format(round(percentage_oecd_plus_china_gdp, 2))
        )
        f.write(
            "\\newcommand{\\extrapolationOECDPlusChinaGovtRevPercent}"
            + "{{{}}}\n".format(round(percentage_oecd_plus_china_govt_revenue, 2))
        )
        f.write(
            "\\newcommand{\\extrapolationDroppedCountries}"
            + "{{{}}}\n".format(dropped_countries_string)
        )
        f.write(
            "\\newcommand{\\extrapolationOutOfSampleCost}"
            + "{{{}}}\n".format(round(total_cost_out_of_sample_costs, 0))
        )
        f.write(
            "\\newcommand{\\extrapolationGlobalGDP}"
            + "{{{}}}\n".format(round(percentage_global_gdp, 2))
        )

        f.write(
            "\\newcommand{\\malawiShareWorldsPoor}"
            + "{{{}}}\n".format(malawi_world_poor)
        )
        f.write(
            "\\newcommand{\\malawiUBIVariableAmount}"
            + "{{{}}}\n".format(round(malawi_variable_amt, 2))
        )
        f.write(
            "\\newcommand{\\malawiGapOracleRatio}"
            + "{{{}}}\n".format(
                round(malawi_cost["continuous_gap"] / malawi_cost["oracle_gap"], 1)
            )
        )
        f.write(
            "\\newcommand{\\malawiGapUBIPercent}"
            + "{{{}}}\n".format(
                round((malawi_cost["continuous_gap"] * 100 / malawi_cost["ubi"])), 1
            )
        )
        f.write(
            "\\newcommand{\\malawiGapUBIVariablePercent}"
            + "{{{}}}\n".format(
                round(
                    (malawi_cost["continuous_gap"] * 100 / malawi_cost["ubi_variable"])
                ),
                1,
            )
        )
        f.write(
            "\\newcommand{\\malawiGapPMTPercent}"
            + "{{{}}}\n".format(
                round(malawi_cost["continuous_gap"] * 100 / malawi_cost["pmt"], 1)
            )
        )
        f.write(
            "\\newcommand{\\malawiBinaryContPercentIncrease}"
            + "{{{}}}\n".format(
                round(
                    (malawi_cost["binary_gap"] - malawi_cost["continuous_gap"])
                    * 100
                    / malawi_cost["continuous_gap"],
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
