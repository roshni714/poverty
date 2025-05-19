import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
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
            linestyle="--",
            color=METHODS["ubi"]["color"],
            label="UBI $2.15",
        )
        ax[1].plot(
            np.linspace(0.0, initial[country]["rate"]),
            np.ones(50) * 2.15 * conversion_factor,
            linestyle="--",
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

        pruned_rates, pruned_costs = prune_results(
            np.array(rates), np.array(costs), val=0
        )

        ax[1].plot(
            pruned_rates,
            pruned_costs,
            marker="o",
            label=dic["name"],
            color=dic["color"],
            linestyle=dic["linestyle"],
        )

        pruned_gaps, pruned_costs = prune_results(
            np.array(gaps), np.array(costs), val=0
        )

        gap_interpolator = interp1d(
            pruned_rates, pruned_costs, kind="linear", fill_value="extrapolate"
        )

        print(method, gap_interpolator(5.0))

        ax[0].plot(
            pruned_gaps,
            pruned_costs,
            marker="o",
            label=dic["name"],
            color=dic["color"],
            linestyle=dic["linestyle"],
        )
    ax[1].set_xlabel(
        "{} Post-Transfer Poverty Rate (%)".format(country.capitalize()),
        fontsize=fontsize,
    )
    ax[0].set_xlabel(
        "{} Post-Transfer Poverty Gap Index (%)".format(country.capitalize()),
        fontsize=fontsize,
    )

    ax[1].set_title(
        "Policy Cost vs {} Post-Transfer Poverty Rate".format(country.capitalize()),
        fontsize=fontsize,
    )
    ax[0].set_title(
        "Policy Cost vs {} Post-Transfer Poverty Gap Index".format(
            country.capitalize()
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
            country.capitalize(), n, d
        ),
        fontsize=fontsize + 2,
    )
    plt.tight_layout()
    plt.savefig("figs/{}.pdf".format(save_as), bbox_inches="tight")


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
        linestyle="--",
        color=METHODS["ubi"]["color"],
        label="UBI $2.15",
    )
    ax[1].plot(
        np.linspace(0.0, initial_popweighted_rate),
        np.ones(50) * ubi_cost,
        linestyle="--",
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
    plt.savefig("figs/{}.pdf".format(save_as), dpi=300, bbox_inches="tight")


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
            linestyle="--",
            color=METHODS["ubi"]["color"],
            label="UBI $2.15",
        )
        ax[1].plot(
            np.linspace(0.0, initial_popweighted_rate),
            np.ones(50) * ubi_cost,
            linestyle="--",
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
    plt.savefig("figs/{}.pdf".format(save_as), dpi=300, bbox_inches="tight")


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
    plt.savefig("figs/{}.pdf".format(save_as), dpi=300, bbox_inches="tight")


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
    plt.savefig("figs/{}.pdf".format(save_as), dpi=300, bbox_inches="tight")


def convert_nominal_2023_to_nominal_survey_year(amt, country):
    inflation_adjustment = {2018: 0.83, 2019: 0.84}
    df = pd.read_csv("learn/currency_conversion.csv")
    survey_year = df[df["country"] == country]["survey_year"].values[0]
    amt = amt * inflation_adjustment[survey_year]
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

    df = pd.read_csv("learn/currency_conversion.csv")
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
        country: govt_revenue_percentage[country] * gdp[country]
        for country in countries
    }
    amts_as_percent_of_revenue = [
        amt * 100 / govt_revenue[country]
        for amt, country in zip(amts_survey_year, countries)
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Bar plot for amts_as_percent_of_gdp
    axes[0].bar([c.capitalize() for c in countries], amts_as_percent_of_gdp)
    axes[0].set_xlabel("Country")
    axes[0].set_ylabel("Percentage (%) of Country GDP \n in Survey Year")
    axes[0].set_title("Policy Cost Relative to Country GDP")

    # Bar plot for amts_as_percent_of_revenue
    axes[1].bar([c.capitalize() for c in countries], amts_as_percent_of_revenue)
    axes[1].set_xlabel("Country")
    axes[1].set_ylabel("Percentage (%) of Govt Revenue  \n in Survey Year")
    axes[1].set_title("Policy Cost Relative to Country Govt Revenue")

    plt.suptitle(
        "Policy Cost to Reach 1% Poverty Gap Target \n Relative to Country GDP and Govt Revenue"
    )

    plt.tight_layout()
    plt.savefig("figs/{}.pdf".format(save_as), bbox_inches="tight")


def get_table_policy_cost_gdp_oda(countries, save_as):
    df = pd.read_csv("learn/currency_conversion.csv")

    dic = get_country_interpolators(countries, "continuous_gap", True)

    res = []
    for country in countries:
        amt = dic[country]["gap_to_cost_interpolator"](1.0).item()
        res.append({"country": country, "policy_cost": amt})

    df2 = pd.DataFrame(res)
    df = df2.merge(df, on="country", how="left")
    df.sort_values(by=["country"], inplace=True)
    df["Policy Cost / GDP"] = df["policy_cost"] / df["GDP_billions_survey_year"]
    df["ODA / GDP"] = df["oda_billions_latest_year"] / df["GDP_billions_survey_year"]
    df.rename(
        columns={
            "policy_cost": "Policy Cost",
            "GDP_billions_survey_year": "GDP",
            "survey_year": "Survey Year",
            "oda_billions_latest_year": "Status-quo ODA",
            "country": "Country",
        },
        inplace=True,
    )
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
        formatters={"Country": str.capitalize},
    )


def get_table_oecd(countries, save_as):
    df = pd.read_csv("learn/auxiliary_data.csv")
    df["GDP"] = (
        df["OECD_nominal_GDP_per_capita_2023"] * df["OECD_population_2023"] / 1000000000
    )
    df["Gov't Revenue"] = (
        df["OECD_nominal_GDP_per_capita_2023"]
        * df["OECD_population_2023"]
        * df["OECD_govt_revenue_percentage_GDP_2023"]
        / 1000000000
    )

    policy_cost = 5.0  # TODO need to write extrapolation code
    df["Policy Cost"] = policy_cost
    df["Policy Cost / GDP"] = df["Policy Cost"] / df["GDP"]
    df["Policy Cost / Gov't Revenue"] = df["Policy Cost"] / df["Gov't Revenue"]

    new_df = df[
        [
            "GDP",
            "Gov't Revenue",
            "Policy Cost",
            "Policy Cost / GDP",
            "Policy Cost / Gov't Revenue",
        ]
    ]

    new_df.to_latex(
        save_as + ".tex",
        index=False,
        float_format="%.2f",
        escape=False,
    )


def get_table_oecd_plus_china(countries, save_as):
    df = pd.read_csv("learn/auxiliary_data.csv")
    df["GDP"] = (
        df["OECD_nominal_GDP_per_capita_2023"] * df["OECD_population_2023"] / 1000000000
    ) + (
        df["China_nominal_GDP_per_capita_2023"]
        * df["China_population_2023"]
        / 1000000000
    )
    df["Gov't Revenue"] = (
        df["OECD_nominal_GDP_per_capita_2023"]
        * df["OECD_population_2023"]
        * df["OECD_govt_revenue_percentage_GDP_2023"]
        / 1000000000
    ) + (
        df["China_nominal_GDP_per_capita_2023"]
        * df["China_population_2023"]
        * df["China_govt_revenue_percentage_GDP_2023"]
        / 1000000000
    )

    policy_cost = 5.0  # TODO need to write extrapolation code
    df["Policy Cost"] = policy_cost
    df["Policy Cost / GDP"] = df["Policy Cost"] / df["GDP"]
    df["Policy Cost / Gov't Revenue"] = df["Policy Cost"] / df["Gov't Revenue"]

    new_df = df[
        [
            "GDP",
            "Gov't Revenue",
            "Policy Cost",
            "Policy Cost / GDP",
            "Policy Cost / Gov't Revenue",
        ]
    ]

    new_df.to_latex(save_as + ".tex", index=False, float_format="%.2f", escape=False)


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
        formatters={"Country": str.capitalize},
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
                "ratio_between_ubi_and_targeting": ubi_cost / targeting_cost,
            }
        )

    df = pd.DataFrame(res)
    df.sort_values(
        by=["ratio_between_ubi_and_targeting"], ascending=False, inplace=True
    )
    plt.bar_chart(df["country"], df["ratio_between_ubi_and_targeting"])
    plt.xlabel("Country")
    plt.ylabel("Ratio")
    plt.suptitle("Ratio of UBI Cost to Targeting Cost vs. Country")
    plt.savefig("figs/{}.pdf".format(save_as), bbox_inches="tight")


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
