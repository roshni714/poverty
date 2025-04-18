import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from aggregation import (
    METHODS,
    get_conversion_factors,
    get_aggregate_interpolators_population_weighted_poverty_measure_global_gap,
    get_aggregate_interpolators_population_weighted_poverty_measure,
    get_aggregate_interpolators_fraction,
    _load_data,
    get_initial_poverty_gaps_and_rates,
    get_initial_aggregate_gap_and_rate,
    get_aggregate_conversion_factor,
)


def make_plot_for_country(country, method_list, geo_extrapolation, save_as):
    train_data = pd.read_parquet("../data/{}/train.parquet".format(country))
    n_train = len(train_data)
    test_data = pd.read_parquet("../data/{}/test.parquet".format(country))
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
            "Policy Cost (Billions of Nominal 2025 USD)", fontsize=fontsize
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
    aggregate_conversion_factor = get_aggregate_conversion_factor(countries)
    ax[0].plot(
        np.linspace(0.0, initial_popweighted_gap),
        np.ones(50) * 2.15 * aggregate_conversion_factor,
        linestyle="--",
        color=METHODS["ubi"]["color"],
        label="UBI $2.15",
    )
    ax[1].plot(
        np.linspace(0.0, initial_popweighted_rate),
        np.ones(50) * 2.15 * aggregate_conversion_factor,
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
                "Total Policy Cost \n(Billions of Nominal 2025 USD)", fontsize=fontsize
            )
            ax[i].grid(True)
            ax[i].tick_params(axis="x", labelsize=15)
            ax[i].tick_params(axis="y", labelsize=15)

        plt.suptitle(
            "Total Policy Cost vs. Poverty Measure Across Countries \n (Global Gap Optimization)",
            fontsize=fontsize,
        )
        ax[1].legend(
            loc="upper right", fontsize=fontsize * 0.75, bbox_to_anchor=(1.0, 0.8)
        )

    plt.tight_layout()
    plt.savefig("figs/{}.pdf".format(save_as), dpi=300, bbox_inches="tight")


def aggregate_plot_x_axis_population_weighted_poverty_measure(
    countries, method_list, geo_extrapolation, save_as
):
    # Plot policy_cost_per_capita vs post_transfer_poverty_rate
    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    fontsize = 20

    initial_popweighted_gap, initial_popweighted_rate = (
        get_initial_aggregate_gap_and_rate(countries)
    )
    aggregate_conversion_factor = get_aggregate_conversion_factor(countries)
    ax[0].plot(
        np.linspace(0.0, initial_popweighted_gap),
        np.ones(50) * 2.15 * aggregate_conversion_factor,
        linestyle="--",
        color=METHODS["ubi"]["color"],
        label="UBI $2.15",
    )
    ax[1].plot(
        np.linspace(0.0, initial_popweighted_rate),
        np.ones(50) * 2.15 * aggregate_conversion_factor,
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
                "Total Policy Cost \n(Billions of Nominal 2025 USD)", fontsize=fontsize
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
                "Total Policy Cost \n(Billions of Nominal 2025 USD)", fontsize=fontsize
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
        dic = get_aggregate_interpolators_population_weighted_poverty_measure_global_gap(
            countries=countries, method=method, geo_extrapolation=geo_extrapolation
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
            "Total Policy Cost vs Post-Transfer Poverty Rate \n (Global Gap Optimization)", fontsize=fontsize
        )
        ax[0].set_title(
            "Total Policy Cost vs Post-Transfer Poverty Gap Index \n (Global Gap Optimization)", fontsize=fontsize
        )
        for i in range(2):
            ax[i].set_ylabel(
                "Total Policy Cost \n(Billions of Nominal 2025 USD)", fontsize=fontsize
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
#             "Worst-Case Poverty Gap in a Country \n(Billions of Nominal 2025 USD)",
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
#                 "Total Policy Cost \n(Billions of Nominal 2025 USD)", fontsize=fontsize
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
