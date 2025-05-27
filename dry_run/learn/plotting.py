import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from learn.aggregation import (
    METHODS,
    get_conversion_factors,
    get_country_interpolators,
    get_aggregate_interpolators_population_weighted_poverty_measure_global_gap,
    get_aggregate_interpolators_population_weighted_poverty_measure,
    get_aggregate_interpolators_fraction,
    _load_data,
    get_initial_poverty_gaps_and_rates,
    get_initial_aggregate_gap_and_rate,
    get_aggregate_ubi_cost,
    prune_results,
)
from learn.predictive_quality import get_out_of_sample_rmse


def get_country_name(country):
    return " ".join([word.capitalize() for word in country.split("_")])


def make_plot_for_country(
    country, method_list, geo_extrapolation, save_as, ubi_off=False
):
    train_data = pd.read_parquet("data/{}/train.parquet".format(country))
    n_train = len(train_data)
    test_data = pd.read_parquet("data/{}/test.parquet".format(country))
    n_test = len(test_data)
    n = n_train + n_test
    d = len(train_data.columns)

    methods = METHODS.copy()
    conversion_factor = get_conversion_factors([country])[country]

    for method in method_list:
        dic = methods[method]
        df = _load_data(country, method, geo_extrapolation)
        dic["df"] = df
    fontsize = 20

    initial = get_initial_poverty_gaps_and_rates([country])

    # Plot policy_cost_per_capita vs post_transfer_poverty_rate
    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    if not ubi_off:
        ax[0].plot(
            np.linspace(0.0, initial[country]["gap"]),
            np.ones(50) * 2.15 * conversion_factor,
            linestyle=":",
            color=METHODS["ubi"]["color"],
            label="UBI $2.15",
        )
        ax[1].plot(
            np.linspace(0.0, initial[country]["rate"]),
            np.ones(50) * 2.15 * conversion_factor,
            linestyle=":",
            color=METHODS["ubi"]["color"],
            label="UBI $2.15",
        )
        print("ubi $2.15", 2.15 * conversion_factor)

    for method in method_list:
        dic = methods[method]
        df = dic["df"]

        rates = [initial[country]["rate"]] + list(
            df["post_transfer_poverty_rate"] * 100
        )
        gaps = [initial[country]["gap"]] + list(
            df["post_transfer_poverty_gap"] * 100 / 2.15
        )
        costs = [0.0] + list(df["policy_cost_per_capita"] * conversion_factor)

        ax[1].plot(
            rates,
            costs,
            marker="o",
            label=dic["name"],
            color=dic["color"],
            linestyle=dic["linestyle"],
        )

        ax[0].plot(
            gaps,
            costs,
            marker="o",
            label=dic["name"],
            color=dic["color"],
            linestyle=dic["linestyle"],
        )
    ax[1].set_xlabel(
        "{} Post-Transfer Poverty Rate (%)".format(get_country_name(country)),
        fontsize=fontsize,
    )
    ax[0].set_xlabel(
        "{} Post-Transfer Poverty Gap Index (%)".format(get_country_name(country)),
        fontsize=fontsize,
    )

    ax[1].set_title(
        "Policy Cost vs {} Post-Transfer Poverty Rate".format(
            get_country_name(country)
        ),
        fontsize=fontsize,
    )
    ax[0].set_title(
        "Policy Cost vs {} Post-Transfer Poverty Gap Index".format(
            get_country_name(country)
        ),
        fontsize=fontsize,
    )

    for i in range(2):
        ax[i].set_ylabel(
            "Policy Cost (Billions of Nominal 2023 USD)", fontsize=fontsize
        )
        ax[i].grid(True)
        ax[i].tick_params(axis="x", labelsize=15)
        ax[i].tick_params(axis="y", labelsize=15)

    plt.legend(loc="upper right", fontsize=fontsize * 0.75, bbox_to_anchor=(1.0, 0.8))
    plt.suptitle(
        "{} (n={}, d={}): Policy Cost vs. Post-Transfer Poverty Measure".format(
            get_country_name(country), n, d
        ),
        fontsize=fontsize + 2,
    )
    plt.tight_layout()
    plt.savefig("{}.pdf".format(save_as), bbox_inches="tight")


def aggregate_plot_x_axis_population_weighted_poverty_measure_global_gap(
    countries, method_list, geo_extrapolation, save_as
):
    # Plot policy_cost_per_capita vs post_transfer_poverty_rate
    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    fontsize = 20

    initial_popweighted_gap, initial_popweighted_rate = (
        get_initial_aggregate_gap_and_rate(countries)
    )
    ubi_cost = get_aggregate_ubi_cost(countries)
    ax[0].axvline(x=1, color="black", linestyle=":", label="1% Gap Target")

    ax[0].plot(
        np.linspace(0.0, initial_popweighted_gap),
        np.ones(50) * ubi_cost,
        linestyle=":",
        color=METHODS["ubi"]["color"],
        label="UBI $2.15",
    )
    ax[1].plot(
        np.linspace(0.0, initial_popweighted_rate),
        np.ones(50) * ubi_cost,
        linestyle=":",
        color=METHODS["ubi"]["color"],
        label="UBI $2.15",
    )
    for method in method_list:
        dic = (
            get_aggregate_interpolators_population_weighted_poverty_measure_global_gap(
                countries=countries, method=method, geo_extrapolation=geo_extrapolation
            )
        )
        gap_range = dic["gap"]["range"]
        gap_interpolator = dic["gap"]["interpolator"]
        rate_range = dic["rate"]["range"]
        rate_interpolator = dic["rate"]["interpolator"]
        ax[0].plot(
            np.linspace(gap_range[0], gap_range[1], 100),
            gap_interpolator(np.linspace(gap_range[0], gap_range[1], 100)),
            label=METHODS[method]["name"],
            color=METHODS[method]["color"],
            linestyle=METHODS[method]["linestyle"],
        )
        ax[1].plot(
            np.linspace(rate_range[0], rate_range[1], 100),
            rate_interpolator(np.linspace(rate_range[0], rate_range[1], 100)),
            label=METHODS[method]["name"],
            color=METHODS[method]["color"],
            linestyle=METHODS[method]["linestyle"],
        )

        ax[1].set_xlabel(
            "Population-Weighted Post-Transfer Poverty Rate\n(%)",
            fontsize=fontsize,
        )
        ax[0].set_xlabel(
            "Population-Weighted Post-Transfer Poverty Gap Index\n(%)",
            fontsize=fontsize,
        )
        ax[1].set_title(
            "Total Policy Cost vs Post-Transfer Poverty Rate", fontsize=fontsize
        )
        ax[0].set_title(
            "Total Policy Cost vs Post-Transfer Poverty Gap Index", fontsize=fontsize
        )
        for i in range(2):
            ax[i].set_ylabel(
                "Total Policy Cost \n(Billions of Nominal 2023 USD)", fontsize=fontsize
            )
            ax[i].grid(True)
            ax[i].tick_params(axis="x", labelsize=15)
            ax[i].tick_params(axis="y", labelsize=15)

        plt.suptitle(
            "Total Policy Cost vs. Poverty Measure Across Countries \n (Global Gap Optimization)",
            fontsize=fontsize,
        )
        ax[0].legend(
            loc="upper right", fontsize=fontsize * 0.75, bbox_to_anchor=(1.0, 0.8)
        )

    plt.tight_layout()
    plt.savefig("{}.pdf".format(save_as), dpi=300, bbox_inches="tight")


def aggregate_plot_x_axis_population_weighted_poverty_measure(
    countries, method_list, geo_extrapolation, save_as, ubi_off=False
):
    # Plot policy_cost_per_capita vs post_transfer_poverty_rate
    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    fontsize = 20

    initial_popweighted_gap, initial_popweighted_rate = (
        get_initial_aggregate_gap_and_rate(countries)
    )
    ubi_cost = get_aggregate_ubi_cost(countries)
    if not ubi_off:
        ax[0].plot(
            np.linspace(0.0, initial_popweighted_gap),
            np.ones(50) * ubi_cost,
            linestyle=":",
            color=METHODS["ubi"]["color"],
            label="UBI $2.15",
        )
        ax[1].plot(
            np.linspace(0.0, initial_popweighted_rate),
            np.ones(50) * ubi_cost,
            linestyle=":",
            color=METHODS["ubi"]["color"],
            label="UBI $2.15",
        )
    for method in method_list:
        dic = get_aggregate_interpolators_population_weighted_poverty_measure(
            countries=countries, method=method, geo_extrapolation=geo_extrapolation
        )
        gap_range = dic["gap"]["range"]
        gap_interpolator = dic["gap"]["interpolator"]
        rate_range = dic["rate"]["range"]
        rate_interpolator = dic["rate"]["interpolator"]
        ax[0].plot(
            np.linspace(gap_range[0], gap_range[1], 100),
            gap_interpolator(np.linspace(gap_range[0], gap_range[1], 100)),
            label=METHODS[method]["name"],
            color=METHODS[method]["color"],
            linestyle=METHODS[method]["linestyle"],
        )
        ax[1].plot(
            np.linspace(rate_range[0], rate_range[1], 100),
            rate_interpolator(np.linspace(rate_range[0], rate_range[1], 100)),
            label=METHODS[method]["name"],
            color=METHODS[method]["color"],
            linestyle=METHODS[method]["linestyle"],
        )

        ax[1].set_xlabel(
            "Population-Weighted Post-Transfer Poverty Rate\n(%)",
            fontsize=fontsize,
        )
        ax[0].set_xlabel(
            "Population-Weighted Post-Transfer Poverty Gap Index\n(%)",
            fontsize=fontsize,
        )
        ax[1].set_title(
            "Total Policy Cost vs Post-Transfer Poverty Rate \n (Global Rate Optimization)",
            fontsize=fontsize,
        )
        ax[0].set_title(
            "Total Policy Cost vs Post-Transfer Poverty Gap Index \n (Global Gap Optimization)",
            fontsize=fontsize,
        )
        for i in range(2):
            ax[i].set_ylabel(
                "Total Policy Cost \n(Billions of Nominal 2023 USD)", fontsize=fontsize
            )
            ax[i].grid(True)
            ax[i].tick_params(axis="x", labelsize=15)
            ax[i].tick_params(axis="y", labelsize=15)

        plt.suptitle("Gap and Rate Targeting Comparison", fontsize=fontsize)
        ax[1].legend(
            loc="upper right", fontsize=fontsize * 0.75, bbox_to_anchor=(1.0, 0.8)
        )

    plt.tight_layout()
    plt.savefig("{}.pdf".format(save_as), dpi=300, bbox_inches="tight")


def aggregate_plot_x_axis_fraction(countries, method_list, geo_extrapolation, save_as):
    # Plot policy_cost_per_capita vs post_transfer_poverty_rate
    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    fontsize = 20
    for method in method_list:
        dic = get_aggregate_interpolators_fraction(
            countries=countries, method=method, geo_extrapolation=geo_extrapolation
        )
        gap_range = dic["gap"]["range"]
        gap_interpolator = dic["gap"]["interpolator"]
        rate_range = dic["rate"]["range"]
        rate_interpolator = dic["rate"]["interpolator"]
        print(gap_range, rate_range)

        ax[0].plot(
            np.linspace(gap_range[0], gap_range[1], 100),
            gap_interpolator(np.linspace(gap_range[0], gap_range[1], 100)),
            label=METHODS[method]["name"],
            color=METHODS[method]["color"],
            linestyle=METHODS[method]["linestyle"],
        )
        ax[1].plot(
            np.linspace(rate_range[0], rate_range[1], 100),
            rate_interpolator(np.linspace(rate_range[0], rate_range[1], 100)),
            label=METHODS[method]["name"],
            color=METHODS[method]["color"],
            linestyle=METHODS[method]["linestyle"],
        )

        ax[1].set_xlabel(
            "Percent Reduction in Poverty Rate in All Countries (%)",
            fontsize=fontsize,
        )
        ax[0].set_xlabel(
            "Percent Reduction in Poverty Gap in All Countries (%)",
            fontsize=fontsize,
        )
        ax[1].set_title(
            "Policy Cost for x% Reduction in Poverty Rate in All Countries \n (Global Rate Optimization)",
            fontsize=fontsize,
        )
        ax[0].set_title(
            "Policy Cost for x% Reduction in Poverty Gap in All Countries \n (Global Gap Optimization)",
            fontsize=fontsize,
        )
        for i in range(2):
            ax[i].set_ylabel(
                "Total Policy Cost \n(Billions of Nominal 2023 USD)", fontsize=fontsize
            )
            ax[i].grid(True)
            ax[i].tick_params(axis="x", labelsize=15)
            ax[i].tick_params(axis="y", labelsize=15)

        plt.suptitle(
            "Policy Cost for x% Reduction in Poverty Measure in All Countries",
            fontsize=fontsize,
        )
        ax[1].legend(loc="upper left", fontsize=fontsize * 0.75)

    plt.tight_layout()
    plt.savefig("{}.pdf".format(save_as), dpi=300, bbox_inches="tight")


def aggregate_plot_geo_extrapolation(countries, save_as):
    # Plot policy_cost_per_capita vs post_transfer_poverty_rate
    # Plot policy_cost_per_capita vs post_transfer_poverty_rate
    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    fontsize = 20
    method = "continuous_gap"

    for geo_extrapolation in [True, False]:
        dic = (
            get_aggregate_interpolators_population_weighted_poverty_measure_global_gap(
                countries=countries, method=method, geo_extrapolation=geo_extrapolation
            )
        )
        gap_range = dic["gap"]["range"]
        gap_interpolator = dic["gap"]["interpolator"]
        rate_range = dic["rate"]["range"]
        rate_interpolator = dic["rate"]["interpolator"]
        print(gap_range, rate_range)

        if geo_extrapolation:
            name = METHODS[method]["name"] + "\n w/ Status-Quo Geographic Identifiers"
            color = METHODS[method]["color"]
        else:
            name = METHODS[method]["name"] + " \n w/ Finer Geographic Identifiers"
            color = "deepskyblue"

        ax[0].plot(
            np.linspace(gap_range[0], gap_range[1], 100),
            gap_interpolator(np.linspace(gap_range[0], gap_range[1], 100)),
            label=name,
            color=color,
            linestyle=METHODS[method]["linestyle"],
        )
        ax[1].plot(
            np.linspace(rate_range[0], rate_range[1], 100),
            rate_interpolator(np.linspace(rate_range[0], rate_range[1], 100)),
            label=name,
            color=color,
            linestyle=METHODS[method]["linestyle"],
        )

        ax[1].set_xlabel(
            "Population-Weighted Post-Transfer Poverty Rate\n(%)",
            fontsize=fontsize,
        )
        ax[0].set_xlabel(
            "Population-Weighted Post-Transfer Poverty Gap Index\n(%)",
            fontsize=fontsize,
        )
        ax[1].set_title(
            "Total Policy Cost vs Post-Transfer Poverty Rate \n (Global Gap Optimization)",
            fontsize=fontsize,
        )
        ax[0].set_title(
            "Total Policy Cost vs Post-Transfer Poverty Gap Index \n (Global Gap Optimization)",
            fontsize=fontsize,
        )
        for i in range(2):
            ax[i].set_ylabel(
                "Total Policy Cost \n(Billions of Nominal 2023 USD)", fontsize=fontsize
            )
            ax[i].grid(True)
            ax[i].tick_params(axis="x", labelsize=15)
            ax[i].tick_params(axis="y", labelsize=15)

        plt.suptitle(
            "Comparison of Policy Costs with Different Granularity of Geographic Identifiers",
            fontsize=fontsize,
        )
        ax[1].legend(loc="upper right", fontsize=fontsize * 0.75)

    plt.tight_layout()
    plt.savefig("{}.pdf".format(save_as), dpi=300, bbox_inches="tight")


def convert_nominal_2023_to_nominal_survey_year(amt, country):
    df1 = pd.read_csv("learn/inflation_adjustment.csv")
    df = pd.read_csv("learn/eop_conversion_factor.csv")
    survey_year = df[df["country"] == country]["survey_year"].values[0]
    inflation_adjustment = df1[df1["survey_year"] == survey_year][
        "inflation_adjustment_to_2023"
    ].values[0]
    amt = amt * (1 / inflation_adjustment)
    return amt


def plot_bar_chart_policy_amt_as_percent_of_gdp(countries, geo_extrapolation, save_as):
    dic = get_country_interpolators(countries, "continuous_gap", geo_extrapolation)

    amts = []  # nominal 2023 USD amts
    for country in countries:
        amt = dic[country]["gap_to_cost_interpolator"](1.0).item()
        amts.append(amt)
        print(country, amt)

    amts_survey_year = [
        convert_nominal_2023_to_nominal_survey_year(amt, country)
        for amt, country in zip(amts, countries)
    ]

    df = pd.read_csv("learn/eop_conversion_factor.csv")
    gdp = (
        df[df["country"].isin(countries)][["country", "GDP_billions_survey_year"]]
        .set_index("country")
        .to_dict()["GDP_billions_survey_year"]
    )
    gdp = {country: gdp[country] for country in countries}
    amts_as_percent_of_gdp = [
        amt * 100 / gdp[country] for amt, country in zip(amts_survey_year, countries)
    ]

    govt_revenue_percentage = (
        df[df["country"].isin(countries)][
            ["country", "govt_revenue_percentage_GDP_survey_year"]
        ]
        .set_index("country")
        .to_dict()["govt_revenue_percentage_GDP_survey_year"]
    )
    govt_revenue = {
        country: govt_revenue_percentage[country] * gdp[country] / 100
        for country in countries
    }
    amts_as_percent_of_revenue = [
        amt * 100 / govt_revenue[country]
        for amt, country in zip(amts_survey_year, countries)
    ]

    fig, axes = plt.subplots(1, 2, figsize=(len(countries) * 2, 6))
    fontsize = 15
    # Bar plot for amts_as_percent_of_gdp
    axes[0].bar([get_country_name(c) for c in countries], amts_as_percent_of_gdp)
    axes[0].set_xlabel("Country", fontsize=fontsize)
    axes[0].set_ylabel(
        "Percentage (%) of Country GDP \n in Survey Year", fontsize=fontsize
    )
    axes[0].set_title("Policy Cost Relative to Country GDP", fontsize=fontsize)

    # Bar plot for amts_as_percent_of_revenue
    axes[1].bar([get_country_name(c) for c in countries], amts_as_percent_of_revenue)
    axes[1].set_xlabel("Country", fontsize=fontsize)
    axes[1].set_ylabel(
        "Percentage (%) of Govt Revenue  \n in Survey Year", fontsize=fontsize
    )
    axes[1].set_title("Policy Cost Relative to Country Govt Revenue", fontsize=fontsize)

    plt.suptitle(
        "Policy Cost to Reach 1% Poverty Gap Target \n Relative to Country GDP and Govt Revenue",
        fontsize=fontsize,
    )

    plt.tight_layout()
    plt.savefig("{}.pdf".format(save_as), bbox_inches="tight")


def get_table_policy_cost_gdp_oda(countries, save_as):
    df = pd.read_csv("learn/eop_conversion_factor.csv")

    dic = get_country_interpolators(countries, "continuous_gap", True)

    res = []
    for country in countries:
        amt = dic[country]["gap_to_cost_interpolator"](1.0).item()
        res.append({"country": country, "policy_cost": amt})

    df2 = pd.DataFrame(res)
    df = df2.merge(df, on="country", how="left")
    df.sort_values(by=["country"], inplace=True)
    df["Policy Cost / GDP"] = df["policy_cost"] / df["GDP_billions_survey_year"]
    df["ODA"] /= 1000000  # ODA is reported in thousands so we convert to billions
    df["ODA / GDP"] = df["ODA"] / df["GDP_billions_survey_year"]
    df.rename(
        columns={
            "policy_cost": "Policy Cost",
            "GDP_billions_survey_year": "GDP",
            "survey_year": "Survey Year",
            "ODA": "Status-quo ODA",
            "country": "Country",
        },
        inplace=True,
    )
    df = df.sort_values(by=["Country"])
    new_df = df[
        [
            "Country",
            "Survey Year",
            "GDP",
            "Status-quo ODA",
            "Policy Cost",
            "Policy Cost / GDP",
            "ODA / GDP",
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


def get_table_diff_between_ubi_and_targeting(countries, save_as):
    dic1 = get_country_interpolators(countries, "continuous_gap", True)
    dic2 = get_country_interpolators(countries, "ubi", True)

    res = []
    for country in countries:
        ubi_cost = dic2[country]["gap_to_cost_interpolator"](1.0).item()
        targeting_cost = dic1[country]["gap_to_cost_interpolator"](1.0).item()
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


def plot_bar_chart_ubi_ratio(countries, save_as):
    dic1 = get_country_interpolators(countries, "continuous_gap", True)
    dic2 = get_country_interpolators(countries, "ubi", True)

    res = []
    for country in countries:
        ubi_cost = dic2[country]["gap_to_cost_interpolator"](1.0).item()
        targeting_cost = dic1[country]["gap_to_cost_interpolator"](1.0).item()
        res.append(
            {
                "country": country,
                "ratio_of_ubi_and_targeting": ubi_cost / targeting_cost,
            }
        )

    df = pd.DataFrame(res)
    df.sort_values(by=["ratio_of_ubi_and_targeting"], ascending=False, inplace=True)
    fontsize = 15
    plt.figure(figsize=(len(countries) * 2, 6))
    plt.bar(
        [get_country_name(country) for country in df["country"]],
        df["ratio_of_ubi_and_targeting"],
    )
    plt.xlabel("Country", fontsize=fontsize)
    plt.ylabel("Ratio", fontsize=fontsize)
    plt.suptitle("Ratio of UBI Cost to Targeting Cost vs. Country", fontsize=fontsize)
    plt.savefig("{}.pdf".format(save_as), bbox_inches="tight")


def get_extrapolation(countries, save_as=None):
    in_sample_countries = countries
    df = pd.read_csv("learn/eop_conversion_factor.csv")
    dic1 = get_country_interpolators(
        in_sample_countries, "oracle_gap", geo_extrapolation=True
    )
    dic2 = get_country_interpolators(
        in_sample_countries, "continuous_gap", geo_extrapolation=True
    )

    in_sample_costs = []
    in_sample_country_ratios = []
    for country in in_sample_countries:
        in_sample_country_ratios.append(
            dic2[country]["gap_to_cost_interpolator"](1.0)
            / dic1[country]["gap_to_cost_interpolator"](1.0)
        )
        in_sample_costs.append(dic2[country]["gap_to_cost_interpolator"](1.0))
        print(country, "in-sample cost:", in_sample_costs[-1])

    initial = get_initial_poverty_gaps_and_rates(countries)
    X = []
    for country in in_sample_countries:
        X.append(
            [
                initial[country]["gap"],
                initial[country]["rate"],
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
        "PPP_conversion_factor_2017",
        "market_exchange_rate_2017",
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
    inflation_df = pd.read_csv("learn/inflation_adjustment.csv")
    inflation_adjustment_factor = (
        inflation_df[inflation_df["survey_year"] == 2017][
            "inflation_adjustment_to_2023"
        ]
        .values[0]
        .item()
    )
    for i, country in enumerate(out_of_sample_countries):
        oracle_gap_index = df[df["country"] == country]["oracle_gap"].values[0].item()
        ppp_exchange_rate = (
            df[df["country"] == country]["PPP_conversion_factor_2017"].values[0].item()
        )
        market_exchange_rate = (
            df[df["country"] == country]["market_exchange_rate_2017"].values[0].item()
        )
        population = (
            df[df["country"] == country]["total_population_2023"].values[0].item()
        )
        oracle_gap = (
            oracle_gap_index
            / 100
            * 2.15
            * 365
            * inflation_adjustment_factor
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


def make_macro_file(countries, save_as):
    countries = sorted(countries)

    all_countries_string = make_string_country_list(countries)

    df = pd.read_csv("learn/share_world_poor.csv")
    total_world_poor = df[df.country.isin(countries)]["share_world_poor"].sum()
    malawi_world_poor = df[df.country == "malawi"]["share_world_poor"].values[0]

    df = pd.read_csv("learn/eop_conversion_factor.csv")
    population_total = df[df.country.isin(countries)][
        "total_population_survey_year"
    ].sum()
    initial = get_initial_poverty_gaps_and_rates(countries)
    weights = (
        np.array(
            [
                df[df["country"] == country]["total_population_survey_year"].values[0]
                for country in countries
            ]
        )
        / population_total
    )
    pov_rates = [initial[country]["rate"] for i, country in enumerate(countries)]
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
        [initial[country]["gap"] * weights[i] for i, country in enumerate(countries)]
    )

    df["ODA"] /= 1000000  # ODA is reported in thousands so we convert to billions
    df["ODA / GDP"] = df["ODA"] / df["GDP_billions_survey_year"]
    sample_oda = df[df["country"].isin(countries)]["ODA / GDP"].mean() * 100

    interpolators = get_country_interpolators(
        countries, "continuous_gap", geo_extrapolation=True
    )
    policy_costs = np.array(
        [
            interpolators[country]["gap_to_cost_interpolator"](1.0)
            for country in countries
        ]
    ).flatten()
    gdp = np.array(
        [
            df["GDP_billions_survey_year"][df["country"] == country].values[0]
            for country in countries
        ]
    ).flatten()
    sample_policy_cost = np.mean(policy_costs / gdp) * 100

    methods = ["continuous_gap", "binary_gap", "oracle_gap", "pmt", "ubi"]
    cost = {}
    for method in methods:
        interpolator = get_aggregate_interpolators_population_weighted_poverty_measure(
            countries, method, True
        )
        if method == "ubi":
            cost[method + "_variable"] = interpolator["gap"]["interpolator"](1.0).item()
        else:
            cost[method] = interpolator["gap"]["interpolator"](1.0).item()

    conversion_factors = get_conversion_factors(countries)
    cost["ubi"] = 2.15 * sum([conversion_factors[country] for country in countries])
    print("HEADLINE COST", cost["continuous_gap"])

    malawi_costs = {}
    for method in methods:
        malawi_interpolator = get_country_interpolators(
            ["malawi"], method, geo_extrapolation=True
        )
        if method == "ubi":
            malawi_costs[method + "_variable"] = malawi_interpolator["malawi"][
                "gap_to_cost_interpolator"
            ](1.0).item()
        else:
            malawi_costs[method] = malawi_interpolator["malawi"][
                "gap_to_cost_interpolator"
            ](1.0).item()

    malawi_costs["ubi"] = 2.15 * conversion_factors["malawi"]

    ratios = []
    for country in countries:
        gap_interpolator = get_country_interpolators(
            [country], "continuous_gap", geo_extrapolation=True
        )
        oracle_interpolator = get_country_interpolators(
            [country], "oracle_gap", geo_extrapolation=True
        )
        ratios.append(
            gap_interpolator[country]["gap_to_cost_interpolator"](1.0).item()
            / oracle_interpolator[country]["gap_to_cost_interpolator"](1.0).item()
        )
    min_ratio = min(ratios)
    max_ratio = max(ratios)

    total_cost, _, out_of_sample_cost, dropped_countries = get_extrapolation(
        countries, save_as=None
    )
    oecd_df = get_table_oecd(total_cost, save_as=None)
    oecd_gdp_percent = oecd_df["Policy Cost (\\% of GDP)"].values[0].item()
    oecd_revenue_percent = oecd_df["Policy Cost (\\% of Gov't Revenue)"].values[0].item() 
    
    oecd_plus_china_df = get_table_oecd_plus_china(total_cost, save_as=None)
    oecd_plus_china_gdp_percent = oecd_plus_china_df["Policy Cost (\\% of GDP)"].values[0].item() 
    
    oecd_plus_china_revenue_percent = oecd_plus_china_df["Policy Cost (\\% of GDP)"].values[0].item()
    
    dropped_countries_string = make_string_country_list(sorted(dropped_countries))

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
        f.write("\\newcommand{\\sampleODAPercent}" + f"{{{sample_oda:.0f}}}\n")
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
            + "{{{}}}\n".format(round((cost["continuous_gap"] / cost["ubi"]) * 100), 0)
        )
        f.write(
            "\\newcommand{\\headlineGapUBIVariablePercent}"
            + "{{{}}}\n".format(
                round((cost["continuous_gap"] / cost["ubi_variable"]) * 100), 0
            )
        )
        f.write(
            "\\newcommand{\\headlineGapOracleRatio}"
            + "{{{}}}\n".format(round((cost["continuous_gap"] / cost["oracle_gap"])), 1)
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

        f.write(
            "\\newcommand{\\extrapolationCost}"
            + "{{{}}}\n".format(round(total_cost, 0))
        )
        f.write(
            "\\newcommand{\\extrapolationOECDGDPPercent}"
            + "{{{}}}\n".format(round(oecd_gdp_percent, 2))
        )
        f.write(
            "\\newcommand{\\extrapolationOECDGovtRevPercent}"
            + "{{{}}}\n".format(round(oecd_revenue_percent, 2))
        )
        f.write(
            "\\newcommand{\\extrapolationOECDPlusChinaGDPPercent}"
            + "{{{}}}\n".format(round(oecd_plus_china_gdp_percent, 2))
        )
        f.write(
            "\\newcommand{\\extrapolationOECDPlusChinaGovtRevPercent}"
            + "{{{}}}\n".format(round(oecd_plus_china_revenue_percent, 2))
        )
        f.write(
            "\\newcommand{\\extrapolationOutOfSampleCost}"
            + "{{{}}}\n".format(round(out_of_sample_cost, 0))
        )
        f.write(
            "\\newcommand{\\extrapolationDroppedCountries}"
            + "{{{}}}\n".format(dropped_countries_string)
        )

        f.write(
            "\\newcommand{\\malawiShareWorldsPoor}"
            + "{{{}}}\n".format(malawi_world_poor)
        )
        f.write(
            "\\newcommand{\\malawiUBIVariableAmount}"
            + "{{{}}}\n".format(
                round(malawi_costs["ubi_variable"] / conversion_factors["malawi"], 2)
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


# def aggregate_plot_x_axis_wc_poverty_measure(countries, method_list, geo_extrapolation):
#     # Plot policy_cost_per_capita vs post_transfer_poverty_rate
#     fig, ax = plt.subplots(1, 2, figsize=(20, 8))
#     fontsize = 20
#     for method in method_list:
#         dic = get_aggregate_interpolators_wc_poverty_measure(
#             countries=countries, method=method, geo_extrapolation=geo_extrapolation
#         )
#         gap_range = dic["gap"]["range"]
#         gap_interpolator = dic["gap"]["interpolator"]
#         rate_range = dic["rate"]["range"]
#         rate_interpolator = dic["rate"]["interpolator"]
#         print(gap_range, rate_range)

#         ax[0].plot(
#             np.linspace(gap_range[0], gap_range[1], 100),
#             gap_interpolator(np.linspace(gap_range[0], gap_range[1], 100)),
#             label=METHODS[method]["name"],
#             color=METHODS[method]["color"],
#             linestyle=METHODS[method]["linestyle"],
#         )
#         ax[1].plot(
#             np.linspace(rate_range[0], rate_range[1], 100),
#             rate_interpolator(np.linspace(rate_range[0], rate_range[1], 100)),
#             label=METHODS[method]["name"],
#             color=METHODS[method]["color"],
#             linestyle=METHODS[method]["linestyle"],
#         )

#         ax[1].set_xlabel("Worst-Case Poverty Rate in a Country\n(%)", fontsize=fontsize)
#         ax[0].set_xlabel(
#             "Worst-Case Poverty Gap in a Country \n(Billions of Nominal 2023 USD)",
#             fontsize=fontsize,
#         )
#         ax[1].set_title(
#             "Total Policy Cost vs Post-Transfer Poverty Rate", fontsize=fontsize
#         )
#         ax[0].set_title(
#             "Total Policy Cost vs Post-Transfer Poverty Gap", fontsize=fontsize
#         )
#         for i in range(2):
#             ax[i].set_ylabel(
#                 "Total Policy Cost \n(Billions of Nominal 2023 USD)", fontsize=fontsize
#             )
#             ax[i].grid(True)
#             ax[i].tick_params(axis="x", labelsize=15)
#             ax[i].tick_params(axis="y", labelsize=15)

#         plt.suptitle(
#             "Total Policy Cost vs. Worst-Case Poverty Measure in a Country",
#             fontsize=fontsize,
#         )
#         ax[1].legend(loc="upper right", fontsize=fontsize * 0.75)

#     plt.tight_layout()
#     plt.savefig("figs/aggregate_wc.pdf", dpi=300, bbox_inches="tight")
