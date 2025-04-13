import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from aggregation import (
    METHODS,
    get_conversion_factors,
    get_aggregate_interpolators_wc_poverty_measure,
    get_aggregate_interpolators_population_weighted_poverty_measure,
    _load_data,
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

    pre_transfer_poverty_gap = max(df["post_transfer_poverty_gap"])
    # Plot policy_cost_per_capita vs post_transfer_poverty_rate
    fig, ax = plt.subplots(1, 2, figsize=(20, 8))

    for method in method_list:
        dic = methods[method]
        df = dic["df"]

        ax[1].plot(
            df["post_transfer_poverty_rate"] * 100,
            df["policy_cost_per_capita"] * conversion_factor,
            marker="o",
            label=dic["name"],
            color=dic["color"],
            linestyle=dic["linestyle"],
        )
        ax[0].plot(
            df["post_transfer_poverty_gap"] * conversion_factor,
            df["policy_cost_per_capita"] * conversion_factor,
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
        "{} Post-Transfer Poverty Gap (Billons of Nominal 2025 USD)".format(
            country.capitalize()
        ),
        fontsize=fontsize,
    )

    ax[1].set_title(
        "Policy Cost vs {} Post-Transfer Poverty Rate".format(country.capitalize()),
        fontsize=fontsize,
    )
    ax[0].set_title(
        "Policy Cost vs {} Post-Transfer Poverty Gap".format(country.capitalize()),
        fontsize=fontsize,
    )

    for i in range(2):
        ax[i].set_ylabel(
            "Policy Cost (Billions of Nominal 2025 USD)", fontsize=fontsize
        )
        ax[i].grid(True)
        ax[i].tick_params(axis="x", labelsize=15)
        ax[i].tick_params(axis="y", labelsize=15)

    plt.legend(loc="upper right", fontsize=fontsize * 0.75)
    plt.suptitle(
        "{} (n={}, d={}): Policy Cost vs. Post-Transfer Poverty Measure".format(
            country.capitalize(), n, d
        ),
        fontsize=fontsize + 2,
    )
    plt.tight_layout()
    plt.savefig("figs/{}.pdf".format(save_as), bbox_inches="tight")


def aggregate_plot_x_axis_wc_poverty_measure(countries, method_list, geo_extrapolation):
    # Plot policy_cost_per_capita vs post_transfer_poverty_rate
    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    fontsize = 20
    for method in method_list:
        dic = get_aggregate_interpolators_wc_poverty_measure(
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

        ax[1].set_xlabel("Worst-Case Poverty Rate in a Country\n(%)", fontsize=fontsize)
        ax[0].set_xlabel(
            "Worst-Case Poverty Gap in a Country \n(Billions of Nominal 2025 USD)",
            fontsize=fontsize,
        )
        ax[1].set_title(
            "Total Policy Cost vs Post-Transfer Poverty Rate", fontsize=fontsize
        )
        ax[0].set_title(
            "Total Policy Cost vs Post-Transfer Poverty Gap", fontsize=fontsize
        )
        for i in range(2):
            ax[i].set_ylabel(
                "Total Policy Cost \n(Billions of Nominal 2025 USD)", fontsize=fontsize
            )
            ax[i].grid(True)
            ax[i].tick_params(axis="x", labelsize=15)
            ax[i].tick_params(axis="y", labelsize=15)

        plt.suptitle(
            "Total Policy Cost vs. Worst-Case Poverty Measure in a Country",
            fontsize=fontsize,
        )
        ax[1].legend(loc="upper right", fontsize=fontsize * 0.75)

    plt.tight_layout()
    plt.savefig("figs/aggregate_wc.pdf", dpi=300, bbox_inches="tight")


def aggregate_plot_x_axis_population_weighted_poverty_measure(
    countries, method_list, geo_extrapolation
):
    # Plot policy_cost_per_capita vs post_transfer_poverty_rate
    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    fontsize = 20
    for method in method_list:
        dic = get_aggregate_interpolators_population_weighted_poverty_measure(
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
            "Average (Population-Weighted) Poverty Rate Across Countries\n(%)",
            fontsize=fontsize,
        )
        ax[0].set_xlabel(
            "Total Poverty Gap Across Countries \n(Billions of Nominal 2025 USD)",
            fontsize=fontsize,
        )
        ax[1].set_title(
            "Total Policy Cost vs Post-Transfer Poverty Rate", fontsize=fontsize
        )
        ax[0].set_title(
            "Total Policy Cost vs Post-Transfer Poverty Gap", fontsize=fontsize
        )
        for i in range(2):
            ax[i].set_ylabel(
                "Total Policy Cost \n(Billions of Nominal 2025 USD)", fontsize=fontsize
            )
            ax[i].grid(True)
            ax[i].tick_params(axis="x", labelsize=15)
            ax[i].tick_params(axis="y", labelsize=15)

        plt.suptitle(
            "Total Policy Cost vs. Poverty Measure Across Countries", fontsize=fontsize
        )
        ax[1].legend(loc="upper right", fontsize=fontsize * 0.75)

    plt.tight_layout()
    plt.savefig("figs/aggregate_population_weighted.pdf", dpi=300, bbox_inches="tight")


def aggregate_plot_geo_extrapolation(countries):
    # Plot policy_cost_per_capita vs post_transfer_poverty_rate
    # Plot policy_cost_per_capita vs post_transfer_poverty_rate
    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    fontsize = 20
    method = "continuous_gap"
    for geo_extrapolation in [True, False]:
        dic = get_aggregate_interpolators_population_weighted_poverty_measure(
            countries=countries, method=method, geo_extrapolation=geo_extrapolation
        )
        gap_range = dic["gap"]["range"]
        gap_interpolator = dic["gap"]["interpolator"]
        rate_range = dic["rate"]["range"]
        rate_interpolator = dic["rate"]["interpolator"]
        print(gap_range, rate_range)

        if geo_extrapolation:
            name = METHODS[method]["name"] + " (Geo Extrapolation)"
            color = "deepskyblue"
        else:
            name = METHODS[method]["name"] + " (Geo Interpolation)"
            color = METHODS[method]["color"]

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
            "Average (Population-Weighted) Poverty Rate Across Countries\n(%)",
            fontsize=fontsize,
        )
        ax[0].set_xlabel(
            "Total Poverty Gap Across Countries \n(Billions of Nominal 2025 USD)",
            fontsize=fontsize,
        )
        ax[1].set_title(
            "Total Policy Cost vs Post-Transfer Poverty Rate", fontsize=fontsize
        )
        ax[0].set_title(
            "Total Policy Cost vs Post-Transfer Poverty Gap", fontsize=fontsize
        )
        for i in range(2):
            ax[i].set_ylabel(
                "Total Policy Cost \n(Billions of Nominal 2025 USD)", fontsize=fontsize
            )
            ax[i].grid(True)
            ax[i].tick_params(axis="x", labelsize=15)
            ax[i].tick_params(axis="y", labelsize=15)

        plt.suptitle(
            "Total Policy Cost vs. Poverty Measure Across Countries", fontsize=fontsize
        )
        ax[1].legend(loc="upper right", fontsize=fontsize * 0.75)

    plt.tight_layout()
    plt.savefig("figs/aggregate_geo_extrapolation.pdf", dpi=300, bbox_inches="tight")
