import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from learn.aggregation import (
    AggregatePovertyResults,
    CountryMethodPovertyResults,
)
from learn.aux_data_prep import Metadata
from learn.formatting import METHODS
from extrapolation import get_national_poverty_rate_target, ExtrapolationResults
from learn.post_processing_utils import (
    convert_nominal_2023_to_nominal_survey_year,
    get_data_dimension,
    get_country_name,
    make_string_country_list,
)


def make_plot_for_country_presentation(
    country,
    method_list,
    metadata,
    save_as,
    ubi_on=True,
):

    methods = METHODS.copy()
    results = []
    for i, method in enumerate(method_list):
        results.append(
            CountryMethodPovertyResults(
                country,
                method,
                metadata,
            )
        )

    oracle_results = CountryMethodPovertyResults(
        country,
        "oracle_gap",
        metadata,
    )
    fontsize = 30
    # Plot policy_cost_per_capita vs post_transfer_poverty_rate
    fig, ax = plt.subplots(1, 2, figsize=(24, 8))

    ax[0].set_xlim(
        -oracle_results.initial_rate * 0.05, oracle_results.initial_rate * 1.05
    )
    ax[1].set_xlim(
        -oracle_results.initial_gap_index * 0.05,
        oracle_results.initial_gap_index * 1.05,
    )
    if ubi_on:
        ax[1].plot(
            np.linspace(0.0, oracle_results.initial_gap_index),
            np.ones(50) * metadata.povertyline * oracle_results.conversion_factor,
            linestyle="--",
            color=METHODS["ubi_standard"]["color"],
            label=METHODS["ubi_standard"]["name"]
            + " (${})".format(metadata.povertyline),
            linewidth=3,
        )
        ax[0].plot(
            np.linspace(0.0, oracle_results.initial_rate),
            np.ones(50) * metadata.povertyline * oracle_results.conversion_factor,
            linestyle="--",
            color=METHODS["ubi_standard"]["color"],
            label=METHODS["ubi_standard"]["name"]
            + " (${})".format(metadata.povertyline),
            linewidth=3,
        )
    ax[0].set_ylim(-0.1, metadata.povertyline * oracle_results.conversion_factor * 1.05)
    ax[1].set_ylim(-0.1, metadata.povertyline * oracle_results.conversion_factor * 1.05)
    ax[0].vlines(
        x=1.0,
        color="grey",
        linestyle="solid",
        linewidth=3,
        ymin=0,
        ymax=metadata.povertyline * oracle_results.conversion_factor * 1.05,
    )

    for i, method in enumerate(method_list):
        dic = methods[method]
        df = results[i]._load_data(method)

        rates = [oracle_results.initial_rate] + list(
            df["post_transfer_poverty_rate"] * 100
        )
        gaps = [oracle_results.initial_gap_index] + list(
            df["post_transfer_poverty_gap"] * 100 / metadata.povertyline
        )
        costs = [0.0] + list(
            df["policy_cost_per_capita"] * results[0].conversion_factor
        )

        f = interp1d(rates, costs)

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

        if method == "oracle_gap":
            ax[0].plot(
                rates,
                max(costs) * np.ones(len(rates)),
                color=dic["color"],
                linestyle="--",
                linewidth=3,
                label="Aggregate Poverty Gap",
            )
            ax[1].plot(
                gaps,
                max(costs) * np.ones(len(gaps)),
                color=dic["color"],
                linestyle="--",
                linewidth=3,
                label="Aggregate Poverty Gap",
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

    ax[0].legend(fontsize=fontsize * 0.75)  # , #bbox_to_anchor=(1.05, 0.5)
    # fig.tight_layout(rect=[0, 0, 0.85, 1])
    plt.tight_layout()
    plt.savefig("{}.pdf".format(save_as), bbox_inches="tight")
    plt.close()


def make_plot_for_country(
    country,
    method_list,
    metadata,
    save_as,
    ubi_on=True,
):

    methods = METHODS.copy()
    results = []
    for i, method in enumerate(method_list):
        results.append(CountryMethodPovertyResults(country, method, metadata))

    oracle_results = CountryMethodPovertyResults(country, "oracle_gap", metadata)
    fontsize = 30
    # Plot policy_cost_per_capita vs post_transfer_poverty_rate
    fig, ax = plt.subplots(1, 2, figsize=(24, 8))

    ax[0].set_xlim(
        -oracle_results.initial_rate * 0.05, oracle_results.initial_rate * 1.05
    )
    ax[1].set_xlim(
        -oracle_results.initial_gap_index * 0.05,
        oracle_results.initial_gap_index * 1.05,
    )
    if ubi_on:
        ax[1].plot(
            np.linspace(0.0, oracle_results.initial_gap_index),
            np.ones(50) * metadata.povertyline * oracle_results.conversion_factor,
            linestyle="--",
            color=METHODS["ubi_standard"]["color"],
            label=METHODS["ubi_standard"]["name"]
            + " (${})".format(metadata.povertyline),
            linewidth=3,
        )
        ax[0].plot(
            np.linspace(0.0, oracle_results.initial_rate),
            np.ones(50) * metadata.povertyline * oracle_results.conversion_factor,
            linestyle="--",
            color=METHODS["ubi_standard"]["color"],
            label=METHODS["ubi_standard"]["name"]
            + " (${})".format(metadata.povertyline),
            linewidth=3,
        )
    ax[0].set_ylim(-0.1, metadata.povertyline * oracle_results.conversion_factor * 1.05)
    ax[1].set_ylim(-0.1, metadata.povertyline * oracle_results.conversion_factor * 1.05)
    for i, method in enumerate(method_list):
        dic = methods[method]
        df = results[i]._load_data(method)

        rates = [oracle_results.initial_rate] + list(
            df["post_transfer_poverty_rate"] * 100
        )
        gaps = [oracle_results.initial_gap_index] + list(
            df["post_transfer_poverty_gap"] * 100 / metadata.povertyline
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
    metadata,
    save_as,
    ubi_on=True,
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
                metadata=metadata,
            )
        )

    oracle_results = AggregatePovertyResults(countries, "oracle_gap", metadata=metadata)
    initial_gap_index, initial_rate = (
        oracle_results.get_initial_aggregate_gap_index_and_rate()
    )

    if ubi_on:
        ubi_cost = results[0].get_aggregate_ubi_cost()
        ax[0].plot(
            np.linspace(0.0, initial_rate),
            np.ones(50) * ubi_cost,
            linestyle="--",
            color=METHODS["ubi_standard"]["color"],
            label=METHODS["ubi_standard"]["name"]
            + " (${})".format(metadata.povertyline),
            linewidth=3,
        )
        ax[1].plot(
            np.linspace(0.0, initial_gap_index),
            np.ones(50) * ubi_cost,
            linestyle="--",
            color=METHODS["ubi_standard"]["color"],
            label=METHODS["ubi_standard"]["name"]
            + " (${})".format(metadata.povertyline),
            linewidth=3,
        )

    for i, method in enumerate(method_list):
        color = METHODS[method]["color"]
        gap_domain = results[i].aggregate_interpolator_gap_domain
        rate_domain = results[i].aggregate_interpolator_rate_domain
        rate_interpolator = results[i].aggregate_interpolator_rate_to_cost
        gap_interpolator = results[i].aggregate_interpolator_gap_to_cost

        if method != "oracle_gap":
            ax[0].plot(
                np.linspace(rate_domain[0], rate_domain[1], 200),
                rate_interpolator(np.linspace(rate_domain[0], rate_domain[1], 200)),
                label=METHODS[method]["name"],
                color=color,
                linestyle=METHODS[method]["linestyle"],
                linewidth=3,
            )
        ax[1].plot(
            np.linspace(gap_domain[0], gap_domain[1], 200),
            gap_interpolator(np.linspace(gap_domain[0], gap_domain[1], 200)),
            label=METHODS[method]["name"],
            color=color,
            linestyle=METHODS[method]["linestyle"],
            linewidth=3,
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


def aggregate_plot_roshni_presentation(
    countries,
    show_method_list,
    metadata,
    save_as,
    ubi_on=True,
    vertical_arrow_rate=True,
    vertical_arrow_gap=True,
):
    # Plot policy_cost_per_capita vs post_transfer_poverty_rate
    fig, ax = plt.subplots(1, 2, figsize=(24, 8))
    fontsize = 30

    results = []
    alpha_val = 0.1

    method_list = [
        # "ubi",
        # "modern_pmt",
        # "pmt",
        # "binary_gap",
        "continuous_rate",
        "continuous_gap",
        # "oracle_gap",
    ]

    arrow_min_x = 1.0
    arrow_min_y = 0.0
    arrow_max_x = arrow_min_x

    for method in method_list:
        results.append(
            AggregatePovertyResults(
                countries=countries,
                method=method,
                metadata=metadata,
            )
        )

    oracle_results = AggregatePovertyResults(countries, "oracle_gap", metadata=metadata)
    initial_gap_index, initial_rate = (
        oracle_results.get_initial_aggregate_gap_index_and_rate()
    )

    ubi_cost = results[0].get_aggregate_ubi_cost()
    if ubi_on:
        alpha = 1.0
        if vertical_arrow_rate:
            ax[0].vlines(
                x=arrow_max_x,
                ymin=0,
                ymax=ubi_cost,
                color="grey",
                linestyle="solid",
                linewidth=3,
                zorder=0,
            )
            ax[0].scatter(
                [arrow_max_x],
                [ubi_cost],
                color=METHODS["ubi"]["color"],
                s=100,
                zorder=3,
            )
    else:
        alpha = alpha_val
    ax[0].plot(
        np.linspace(0.0, initial_rate),
        np.ones(50) * ubi_cost,
        linestyle="--",
        color=METHODS["ubi_standard"]["color"],
        label=METHODS["ubi_standard"]["name"] + " (${})".format(metadata.povertyline),
        linewidth=3,
        alpha=alpha,
        zorder=2,
    )
    ax[1].plot(
        np.linspace(0.0, initial_gap_index),
        np.ones(50) * ubi_cost,
        linestyle="--",
        color=METHODS["ubi_standard"]["color"],
        label=METHODS["ubi_standard"]["name"] + " (${})".format(metadata.povertyline),
        linewidth=3,
        alpha=alpha,
        zorder=2,
    )

    for i, method in enumerate(method_list):
        if method in show_method_list:
            point = results[i].aggregate_interpolator_rate_to_cost(arrow_max_x)
            gap_val = results[i].aggregate_interpolator_rate_to_gap(arrow_max_x)
            gap_cost = results[i].aggregate_interpolator_gap_to_cost(gap_val).item()

            if vertical_arrow_rate:
                ax[0].vlines(
                    x=arrow_max_x,
                    ymin=0,
                    ymax=point,
                    color="grey",
                    linestyle="solid",
                    linewidth=3,
                )
                if method != "oracle_gap":
                    ax[0].scatter(
                        [arrow_max_x],
                        [point],
                        color=METHODS[method]["color"],
                        s=100,
                        zorder=3,
                    )
            if vertical_arrow_gap and method == "continuous_gap":
                ax[1].vlines(
                    x=gap_val,
                    ymin=0,
                    ymax=gap_cost,
                    color="grey",
                    linestyle="solid",
                    linewidth=3,
                )
                # if "oracle_gap" in show_method_list:
                ax[1].scatter(
                    [gap_val],
                    [gap_cost],
                    color=METHODS[method]["color"],
                    s=100,
                    zorder=3,
                )
            if vertical_arrow_rate and method == "oracle_gap":
                gap_val = results[0].aggregate_interpolator_rate_to_gap(
                    arrow_max_x
                )  # hardcoded to get gap that feasible policy attains
                gap_cost = results[i].aggregate_interpolator_gap_to_cost(gap_val).item()
                ax[0].scatter(
                    [1.0],
                    [gap_cost],
                    color=METHODS[method]["color"],
                    s=100,
                    zorder=3,
                )

    for i, method in enumerate(method_list):
        if method in show_method_list:
            alpha = 1.0
        else:
            alpha = alpha_val

        gap_domain = results[i].aggregate_interpolator_gap_domain
        rate_domain = results[i].aggregate_interpolator_rate_domain
        rate_interpolator = results[i].aggregate_interpolator_rate_to_cost
        gap_interpolator = results[i].aggregate_interpolator_gap_to_cost

        if method != "oracle_gap":
            ax[0].plot(
                np.linspace(rate_domain[0], rate_domain[1], 200),
                rate_interpolator(np.linspace(rate_domain[0], rate_domain[1], 200)),
                label=METHODS[method]["name"],
                color=METHODS[method]["color"],
                linestyle=METHODS[method]["linestyle"],
                alpha=alpha,
                linewidth=3,
            )
            ax[1].plot(
                np.linspace(gap_domain[0], gap_domain[1], 200),
                gap_interpolator(np.linspace(gap_domain[0], gap_domain[1], 200)),
                label=METHODS[method]["name"],
                color=METHODS[method]["color"],
                linestyle=METHODS[method]["linestyle"],
                alpha=alpha,
                linewidth=3,
            )
        elif method == "oracle_gap":
            ax[0].plot(
                np.linspace(rate_domain[0], rate_domain[1], 200),
                rate_interpolator(0) * np.ones(200),
                label="Aggregate Poverty Gap",
                color=METHODS[method]["color"],
                linestyle="--",
                alpha=alpha,
                linewidth=3,
            )

            ax[1].plot(
                np.linspace(gap_domain[0], gap_domain[1], 200),
                gap_interpolator(0) * np.ones(200),
                label="Aggregate Poverty Gap",
                color=METHODS[method]["color"],
                linestyle="--",
                alpha=alpha,
                linewidth=3,
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

    ax[0].legend(fontsize=fontsize * 0.75)

    plt.tight_layout()
    plt.savefig("{}.pdf".format(save_as), dpi=300, bbox_inches="tight")
    plt.close()


def aggregate_plot_presentation(
    countries,
    show_method_list,
    metadata,
    save_as,
    ubi_on=True,
    vertical_arrow_rate=True,
    vertical_arrow_gap=True,
):
    # Plot policy_cost_per_capita vs post_transfer_poverty_rate
    fig, ax = plt.subplots(1, 2, figsize=(24, 8))
    fontsize = 30

    results = []
    alpha_val = 0.1

    method_list = [
        "ubi",
        # "modern_pmt",
        "pmt",
        "binary_gap",
        "continuous_gap",
        "oracle_gap",
    ]

    arrow_min_x = 1.0
    arrow_min_y = 0.0
    arrow_max_x = arrow_min_x

    for method in method_list:
        results.append(
            AggregatePovertyResults(
                countries=countries, method=method, metadata=metadata
            )
        )

    oracle_results = AggregatePovertyResults(countries, "oracle_gap", metadata=metadata)
    initial_gap_index, initial_rate = (
        oracle_results.get_initial_aggregate_gap_index_and_rate()
    )

    ubi_cost = results[0].get_aggregate_ubi_cost()
    if ubi_on:
        alpha = 1.0
        if vertical_arrow_rate:
            ax[0].vlines(
                x=arrow_max_x,
                ymin=0,
                ymax=ubi_cost,
                color="grey",
                linestyle="solid",
                linewidth=3,
                zorder=0,
            )
            ax[0].scatter(
                [arrow_max_x],
                [ubi_cost],
                color=METHODS["ubi"]["color"],
                s=100,
                zorder=3,
            )
    else:
        alpha = alpha_val
    ax[0].plot(
        np.linspace(0.0, initial_rate),
        np.ones(50) * ubi_cost,
        linestyle="--",
        color=METHODS["ubi_standard"]["color"],
        label=METHODS["ubi_standard"]["name"] + " (${})".format(metadata.povertyline),
        linewidth=3,
        alpha=alpha,
        zorder=2,
    )
    ax[1].plot(
        np.linspace(0.0, initial_gap_index),
        np.ones(50) * ubi_cost,
        linestyle="--",
        color=METHODS["ubi_standard"]["color"],
        label=METHODS["ubi_standard"]["name"] + " (${})".format(metadata.povertyline),
        linewidth=3,
        alpha=alpha,
        zorder=2,
    )

    for i, method in enumerate(method_list):
        if method in show_method_list:
            point = results[i].aggregate_interpolator_rate_to_cost(arrow_max_x)
            gap_val = results[i].aggregate_interpolator_rate_to_gap(arrow_max_x)
            gap_cost = results[i].aggregate_interpolator_gap_to_cost(gap_val).item()

            if vertical_arrow_rate:
                ax[0].vlines(
                    x=arrow_max_x,
                    ymin=0,
                    ymax=point,
                    color="grey",
                    linestyle="solid",
                    linewidth=3,
                )
                if method != "oracle_gap":
                    ax[0].scatter(
                        [arrow_max_x],
                        [point],
                        color=METHODS[method]["color"],
                        s=100,
                        zorder=3,
                    )
            if vertical_arrow_gap and method == "continuous_gap":
                ax[1].vlines(
                    x=gap_val,
                    ymin=0,
                    ymax=gap_cost,
                    color="grey",
                    linestyle="solid",
                    linewidth=3,
                )
                # if "oracle_gap" in show_method_list:
                ax[1].scatter(
                    [gap_val],
                    [gap_cost],
                    color=METHODS[method]["color"],
                    s=100,
                    zorder=3,
                )
            if vertical_arrow_gap and method == "oracle_gap":
                gap_val = results[0].aggregate_interpolator_rate_to_gap(
                    arrow_max_x
                )  # hardcoded to get gap that feasible policy attains
                gap_cost = results[i].aggregate_interpolator_gap_to_cost(gap_val).item()
                ax[1].scatter(
                    [gap_val],
                    [gap_cost],
                    color=METHODS[method]["color"],
                    s=100,
                    zorder=3,
                )

    for i, method in enumerate(method_list):
        if method in show_method_list:
            alpha = 1.0
        else:
            alpha = alpha_val

        gap_domain = results[i].aggregate_interpolator_gap_domain
        rate_domain = results[i].aggregate_interpolator_rate_domain
        rate_interpolator = results[i].aggregate_interpolator_rate_to_cost
        gap_interpolator = results[i].aggregate_interpolator_gap_to_cost

        if method != "oracle_gap":
            ax[0].plot(
                np.linspace(rate_domain[0], rate_domain[1], 200),
                rate_interpolator(np.linspace(rate_domain[0], rate_domain[1], 200)),
                label=METHODS[method]["name"],
                color=METHODS[method]["color"],
                linestyle=METHODS[method]["linestyle"],
                alpha=alpha,
                linewidth=3,
            )
        ax[1].plot(
            np.linspace(gap_domain[0], gap_domain[1], 200),
            gap_interpolator(np.linspace(gap_domain[0], gap_domain[1], 200)),
            label=METHODS[method]["name"],
            color=METHODS[method]["color"],
            linestyle=METHODS[method]["linestyle"],
            alpha=alpha,
            linewidth=3,
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

    ax[1].legend(fontsize=fontsize * 0.75)

    plt.tight_layout()
    plt.savefig("{}.pdf".format(save_as), dpi=300, bbox_inches="tight")
    plt.close()


def plot_bar_chart_policy_amt_as_percent_of_gdp(countries, metadata, save_as):
    results = [
        CountryMethodPovertyResults(country, "continuous_gap", metadata=metadata)
        for country in countries
    ]

    amts = []  # nominal 2023 USD amts
    for i in range(len(countries)):
        amt = results[i].rate_to_cost_interpolator(metadata.nationalPovertyRate).item()
        amts.append(amt)

    amts_survey_year = [
        convert_nominal_2023_to_nominal_survey_year(amt, country, metadata=metadata)
        for amt, country in zip(amts, countries)
    ]

    df = metadata.preprocess_country_aux_data()
    gdp = (
        df[df["country_code"].isin(countries)][["country_code", "GDP_survey_year"]]
        .set_index("country_code")
        .to_dict()["GDP_survey_year"]
    )
    gdp = {country: gdp[country] for country in countries}
    amts_as_percent_of_gdp = np.array(
        [amt * 100 / gdp[country] for amt, country in zip(amts_survey_year, countries)]
    )

    govt_revenue_percentage = (
        df[df["country_code"].isin(countries)][
            ["country_code", "government_revenue_percentage_survey_year"]
        ]
        .set_index("country_code")
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

    xlabels = np.array([get_country_name(c, metadata) for c in countries])
    sort_index = np.argsort(amts_as_percent_of_gdp)[::-1]

    fig, axes = plt.subplots(2, 1, figsize=(30, 8 * 2))
    fontsize = 30
    # Bar plot for amts_as_percent_of_gdp
    axes[0].bar(xlabels[sort_index], amts_as_percent_of_gdp[sort_index], zorder=3)
    axes[0].set_xlabel("Country", fontsize=fontsize)
    axes[0].set_ylabel("% of GDP", fontsize=fontsize)
    # axes[0].set_title("Policy Cost as Percentage of Country GDP", fontsize=fontsize)
    axes[0].set_xticklabels(xlabels[sort_index], rotation=90, fontsize=fontsize)
    axes[0].set_yticklabels(
        np.round(axes[0].get_yticks()).astype(int), fontsize=fontsize
    )

    sort_index2 = np.argsort(amts_as_percent_of_revenue)[::-1]
    axes[0].grid(axis="y", zorder=0)
    axes[1].grid(axis="y", zorder=0)
    # Bar plot for amts_as_percent_of_revenue
    axes[1].bar(xlabels[sort_index2], amts_as_percent_of_revenue[sort_index2], zorder=3)
    axes[1].set_xlabel("Country", fontsize=fontsize)
    axes[1].set_ylabel("% of Gov't Revenue", fontsize=fontsize)
    axes[1].set_xticklabels(xlabels[sort_index2], rotation=90, fontsize=fontsize)
    axes[1].set_yticklabels(
        np.round(axes[1].get_yticks()).astype(int), fontsize=fontsize
    )
    # axes[1].set_title("Policy Cost as Percentage of Country Govt Revenue", fontsize=fontsize)

    plt.tight_layout()
    plt.savefig("{}.pdf".format(save_as), bbox_inches="tight")


def get_global_poverty_rate_target(metadata):
    df = metadata.preprocess_wpc_data()
    df = df[df["year"] == 2023]
    national_poverty_rate_target = metadata.nationalPovertyRate / 100

    def global_poverty_rate(national_ceiling):
        national_poverty_rates = np.minimum(
            df["wpc_poverty_rate"], national_ceiling
        ).to_numpy()
        global_poverty_rate = (
            national_poverty_rates * df["total_population_2023"]
        ).sum() / df["total_population_2023"].sum()
        return global_poverty_rate.item() * 100

    ceilings = np.linspace(0, df["wpc_poverty_rate"].max(), 100)
    global_poverty_rates = [global_poverty_rate(c) for c in ceilings]
    global_poverty_rate_target = interp1d(ceilings, global_poverty_rates)(
        national_poverty_rate_target
    )
    return global_poverty_rate_target


def get_table_policy_cost_gdp(countries, metadata, save_as):
    df = metadata.preprocess_country_aux_data()

    results = [
        CountryMethodPovertyResults(country, "continuous_gap", metadata=metadata)
        for country in countries
    ]

    res = []
    nationalPovertyRate = get_national_poverty_rate_target(metadata)
    for result in results:
        amt = result.rate_to_cost_interpolator(nationalPovertyRate).item()
        res.append({"country_code": result.country, "policy_cost": amt})

    df2 = pd.DataFrame(res)
    df = df2.merge(df, on="country_code", how="left")
    df.sort_values(by=["country_code"], inplace=True)
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
            "country_code": "Country Code",
            "government_revenue_survey_year": "Gov't Revenue",
        },
        inplace=True,
    )
    df["Country"] = df["Country Code"].apply(lambda x: get_country_name(x, metadata))
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
    )


def get_macros_relative_cost(policy_cost, oracle_cost, metadata):
    df = metadata.preprocess_secondary_aux_data()

    percentage_oecd_gdp = (
        100 * policy_cost / df[df["indicator"] == "OECD_GDP_2023"]["value"].item()
    )
    percentage_oecd_plus_china_gdp = (
        100
        * policy_cost
        / (
            df[df["indicator"] == "OECD_GDP_2023"]["value"].item()
            + df[df["indicator"] == "china_GDP_2023"]["value"].item()
        )
    )
    percentage_oecd_govt_revenue = (
        100
        * policy_cost
        / (
            df[df["indicator"] == "OECD_GDP_2023"]["value"].item()
            * df[df["indicator"] == "OECD_govt_revenue_percentage_2023"]["value"].item()
            / 100
        )
    )
    percentage_oecd_plus_china_govt_revenue = (
        100
        * policy_cost
        / (
            df[df["indicator"] == "OECD_GDP_2023"]["value"].item()
            * df[df["indicator"] == "OECD_govt_revenue_percentage_2023"]["value"].item()
            / 100
            + df[df["indicator"] == "china_GDP_2023"]["value"].item()
            * df[df["indicator"] == "china_govt_revenue_percentage_2023"][
                "value"
            ].item()
            / 100
        )
    )
    percentage_feasible_global_gdp = (
        100 * policy_cost / df[df["indicator"] == "global_GDP_2023"]["value"].item()
    )
    percentage_oracle_global_gdp = (
        100 * oracle_cost / df[df["indicator"] == "global_GDP_2023"]["value"].item()
    )
    return (
        percentage_oecd_gdp,
        percentage_oecd_plus_china_gdp,
        percentage_oecd_govt_revenue,
        percentage_oecd_plus_china_govt_revenue,
        percentage_feasible_global_gdp,
        percentage_oracle_global_gdp,
    )


def get_table_survey_info(countries, metadata, save_as, slides=False):
    df = metadata.preprocess_country_aux_data()
    df = df[df["country_code"].isin(countries)]

    survey_names = {
        "Enquête Harmonisée sur le Conditions de Vie des Ménages (EHCVM) 2018-2019": "EHCVM",
        "Socioeconomic Panel Survey: 2009-2010": "Socioeconomic Panel Survey",
        "Fifth Integrated Household Survey 2019-2020": "Fifth Integrated Household Survey",
        "Living Standards Survey 2018-19": "Living Standards Survey",
        "Income and Expenditure Survey 2010-2011": "Income and Expenditure Survey",
        "High Frequency Survey 2015": "High Frequency Survey",
        "National Panel Survey 2020-21, Wave 5": "National Panel Survey, Wave 5",
        "Harmonized Survey on Households Living Standards 2018-2019": "HSHLS",
        "National Panel Survey 2019-2020": "National Panel Survey",
        "Kenya Continuous Household Survey 2021": "Continuous Household Survey",
        "Ethiopia - Socio-Economic Panel Survey 2021-2022": "Socio-Economic Panel Survey",
        "Household Consumption Expenditure Survey: 2022-23": "Household Consumption Expenditure Survey",
        "National Household Budget Survey (ENPH) 2016-2017": "ENPH",
    }

    def rename_survey(x):
        if x in survey_names:
            return survey_names[x]
        return x

    df["survey_name"] = df["survey_name"].apply(rename_survey)

    sample_sizes = []
    covariate_dimensions = []

    for country in countries:
        n, d = get_data_dimension(country)
        sample_sizes.append(n)
        covariate_dimensions.append(d)

    new_df = pd.DataFrame(
        {
            "country_code": countries,
            "country_name": [get_country_name(c, metadata) for c in countries],
            "sample_size": sample_sizes,
            "covariate_dimension": covariate_dimensions,
        }
    )

    df = df.merge(new_df, on="country_code", how="left")

    columns = [
        "country_name",
        "survey_name",
        "survey_year",
        "sample_size",
        "covariate_dimension",
        "survey_poverty_rate_povertyline_{}".format(metadata.year),
        "wb_poverty_rate_povertyline_{}_survey_year".format(metadata.year),
    ]
    df["survey_year"] = df["survey_year"].astype(int)
    df = df[columns]
    df["survey_poverty_rate_povertyline_{}".format(metadata.year)] *= 100
    df["wb_poverty_rate_povertyline_{}_survey_year".format(metadata.year)] *= 100
    df.sort_values(by=["country_name"], inplace=True)
    df.rename(
        columns={
            "country_name": "Country",
            "sample_size": "$n$",
            "covariate_dimension": "$d$",
            "survey_poverty_rate_povertyline_{}".format(
                metadata.year
            ): "Survey Poverty Rate",
            "wb_poverty_rate_povertyline_{}_survey_year".format(
                metadata.year
            ): "WB Poverty Rate",
            "survey_name": "Survey Name",
            "survey_year": "Survey Year",
        },
        inplace=True,
    )
    if save_as:
        df.to_latex(
            save_as + ".tex",
            index=False,
            float_format="%.2f",
            escape=False,
        )
    if slides:
        df.drop(columns=["WB Poverty Rate", "Survey Poverty Rate"], inplace=True)
        df.to_latex(
            save_as + "_slides.tex",
            index=False,
            float_format="%.2f",
            escape=False,
        )
    return df


def get_table_wpc(countries, metadata, save_as):
    df = metadata.preprocess_wpc_data(countries=countries)
    df = df[df["year"] == 2023]
    df = df[["country_code", "wpc_poverty_rate", "wpc_share_world_poor"]]
    df["wpc_poverty_rate"] *= 100
    df["wpc_share_world_poor"] *= 100
    df.rename(
        columns={
            "country_code": "Country Code",
            "wpc_poverty_rate": "Poverty Rate",
            "wpc_share_world_poor": "Share of World's Poor",
        },
        inplace=True,
    )
    df["Country"] = df["Country Code"].apply(lambda x: get_country_name(x, metadata))
    df = df.sort_values(by=["Country"])
    df.to_latex(
        save_as + ".tex",
        index=False,
        float_format="%.2f",
        escape=False,
    )


def plot_bar_chart_ubi_ratio(countries, metadata, save_as):
    cont_gap_results = [
        CountryMethodPovertyResults(country, method="continuous_gap", metadata=metadata)
        for country in countries
    ]
    # ubi_results = [
    #     CountryMethodPovertyResults(
    #         country,
    #         method="ubi",
    #         geo_extrapolation=True,
    #         povertyline=povertyline,
    #         year=year,
    #     )
    #     for country in countries
    # ]

    res = []
    for i, country in enumerate(countries):
        ubi_cost = cont_gap_results[i].get_ubi_cost()
        targeting_cost = cont_gap_results[i].rate_to_cost_interpolator(
            metadata.nationalPovertyRate
        )
        res.append(
            {
                "country_code": country,
                "ratio_of_ubi_and_targeting": ubi_cost / targeting_cost,
            }
        )

    df = pd.DataFrame(res)
    df.sort_values(by=["ratio_of_ubi_and_targeting"], ascending=False, inplace=True)
    fontsize = 30
    plt.figure(figsize=(30, 8))

    plt.bar(
        [get_country_name(c, metadata) for c in df["country_code"]],
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
    plt.close()

    plt.figure(figsize=(10, 8))
    initial_rates = np.array(
        [cont_gap_results[i].initial_rate for i in range(len(countries))]
    ).reshape(-1, 1)
    ubi_ratios = np.array(
        [res[i]["ratio_of_ubi_and_targeting"] for i in range(len(countries))]
    ).reshape(-1, 1)
    plt.scatter(
        initial_rates.flatten(), ubi_ratios.flatten(), marker="o", s=100, zorder=3
    )
    xs = np.linspace(0, max(initial_rates), 100).reshape(-1, 1)
    plt.xlabel("Poverty Rate (Survey)", fontsize=fontsize)
    plt.ylabel("Cost Ratio", fontsize=fontsize)
    plt.xticks(fontsize=fontsize * 0.75)
    plt.yticks(fontsize=fontsize * 0.75)
    plt.axhline(y=1, color="grey", linestyle="--", linewidth=2, label="Cost Ratio = 1")
    plt.legend(fontsize=fontsize * 0.75)
    plt.savefig("{}_scatter.pdf".format(save_as), bbox_inches="tight")


def plot_bar_chart_oracle_ratio(countries, metadata, save_as):
    cont_gap_results = [
        CountryMethodPovertyResults(country, method="continuous_gap", metadata=metadata)
        for country in countries
    ]
    oracle_costs = [
        cont_gap_results[i].get_poverty_gap() for i in range(len(countries))
    ]

    res = []
    for i, country in enumerate(countries):
        oracle_cost = oracle_costs[i]
        targeting_cost = cont_gap_results[i].rate_to_cost_interpolator(
            metadata.nationalPovertyRate
        )
        res.append(
            {
                "country_code": country,
                "ratio_of_oracle_and_targeting": targeting_cost / oracle_cost,
            }
        )

    df = pd.DataFrame(res)
    df.sort_values(by=["ratio_of_oracle_and_targeting"], ascending=False, inplace=True)
    fontsize = 30
    plt.figure(figsize=(30, 8))
    plt.bar(
        [get_country_name(c, metadata) for c in df["country_code"]],
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


def get_macros_share_world_poor(countries, metadata):
    wpc_data = metadata.preprocess_wpc_data(countries=countries)
    wpc_data = wpc_data[wpc_data["year"] == 2023]
    total_world_poor = wpc_data["wpc_share_world_poor"].sum() * 100
    malawi_world_poor = round(
        (
            wpc_data[(wpc_data["country_code"] == "MWI")][
                "wpc_share_world_poor"
            ].values[0]
            * 100
        ),
        2,
    )
    return total_world_poor, malawi_world_poor


def get_macros_survey_info(countries, metadata):
    df = metadata.preprocess_country_aux_data()

    weights = np.array(
        [
            df[df["country_code"] == country]["total_population_survey_year"].values[0]
            for country in countries
        ]
    )
    weights = weights / weights.sum()

    pov_rates = np.array(
        [
            df[df["country_code"] == country][
                f"survey_poverty_rate_povertyline_{metadata.year}"
            ].values[0]
            * 100
            for country in countries
        ]
    )
    pov_gaps = np.array(
        [
            df[df["country_code"] == country][
                f"survey_poverty_gap_index_povertyline_{metadata.year}"
            ].values[0]
            * 100
            for country in countries
        ]
    )
    initial_pov_rate = np.sum(weights * pov_rates)
    initial_pov_gap = np.sum(weights * pov_gaps)

    min_pov_rate = min(pov_rates)
    max_pov_rate = max(pov_rates)
    arg_min_pov_rate = np.argmin(pov_rates)
    arg_max_pov_rate = np.argmax(pov_rates)
    min_country = get_country_name(countries[arg_min_pov_rate], metadata)
    max_country = get_country_name(countries[arg_max_pov_rate], metadata)

    return (
        initial_pov_rate,
        initial_pov_gap,
        min_pov_rate,
        max_pov_rate,
        min_country,
        max_country,
    )


def get_headline_numbers(countries, metadata):

    national_poverty_rate_for_global = get_national_poverty_rate_target(metadata)

    agg_results = []
    methods = ["continuous_gap", "binary_gap", "oracle_gap", "pmt", "ubi"]
    for method in methods:
        agg_results.append(
            AggregatePovertyResults(
                countries=countries, method=method, metadata=metadata
            )
        )

    cost = {}
    for i, method in enumerate(methods):
        if method == "ubi":
            cost[method + "_variable"] = (
                agg_results[i]
                .aggregate_interpolator_rate_to_cost(national_poverty_rate_for_global)
                .item()
            )
        elif method == "oracle_gap":
            gap_induced = (
                agg_results[0]
                .aggregate_interpolator_rate_to_gap(national_poverty_rate_for_global)
                .item()
            )  # want oracle to correpond to global poverty gap

            cost[method] = (
                agg_results[i].aggregate_interpolator_gap_to_cost(gap_induced).item()
            )
        else:
            cost[method] = (
                agg_results[i]
                .aggregate_interpolator_rate_to_cost(national_poverty_rate_for_global)
                .item()
            )

        cost["ubi"] = metadata.povertyline * sum(
            [
                agg_results[i].country_results[countries[j]].conversion_factor
                for j in range(len(countries))
            ]
        )
    return cost, agg_results


def get_macros_oracle_ratios(countries, metadata):
    ratios = []
    cont_gap_results = [
        CountryMethodPovertyResults(country, "continuous_gap", metadata=metadata)
        for country in countries
    ]
    oracle_results = [
        cont_gap_results[i].get_poverty_gap() for i in range(len(countries))
    ]
    for i, country in enumerate(countries):
        ratios.append(
            cont_gap_results[i]
            .rate_to_cost_interpolator(metadata.nationalPovertyRate)
            .item()
            / oracle_results[i].item()
        )
    min_ratio = min(ratios)
    max_ratio = max(ratios)
    return min_ratio, max_ratio


def get_macro_povertyline_comparison(countries, metadata):

    old_metadata = Metadata(
        povertyline=2.15,
        year=2017,
        auxpath=metadata.auxpath,
        wpcpath=metadata.wpcpath,
        secondaryauxpath=metadata.secondaryauxpath,
        nationalPovertyRate=metadata.nationalPovertyRate,
        globalPovertyRate=metadata.globalPovertyRate,
    )

    new_metadata = Metadata(
        povertyline=3.0,
        year=2021,
        auxpath=metadata.auxpath,
        wpcpath=metadata.wpcpath,
        secondaryauxpath=metadata.secondaryauxpath,
        nationalPovertyRate=metadata.nationalPovertyRate,
        globalPovertyRate=metadata.globalPovertyRate,
    )
    cont_gap1 = AggregatePovertyResults(
        countries=countries, method="continuous_gap", metadata=old_metadata
    )
    cont_gap2 = AggregatePovertyResults(
        countries=countries, method="continuous_gap", metadata=new_metadata
    )

    cost1 = cont_gap1.aggregate_interpolator_rate_to_cost(
        metadata.nationalPovertyRate
    ).item()
    cost2 = cont_gap2.aggregate_interpolator_rate_to_cost(
        metadata.nationalPovertyRate
    ).item()
    ratio = cost2 / cost1
    return cost2, ratio


def get_percentages(countries, costs, metadata):

    df = metadata.preprocess_country_aux_data()
    gdp = (
        df[df["country_code"].isin(countries)][["country_code", "GDP_survey_year"]]
        .set_index("country_code")
        .to_dict()["GDP_survey_year"]
    )
    gdp = {country: gdp[country] for country in countries}

    govt_revenue_percentage = (
        df[df["country_code"].isin(countries)][
            ["country_code", "government_revenue_percentage_survey_year"]
        ]
        .set_index("country_code")
        .to_dict()["government_revenue_percentage_survey_year"]
    )
    govt_revenue = {
        country: govt_revenue_percentage[country] * gdp[country] / 100
        for country in countries
    }

    oda = (
        df[df["country_code"].isin(countries)][["country_code", "ODA_most_recent"]]
        .set_index("country_code")
        .to_dict()["ODA_most_recent"]
    )
    oda = {country: oda[country] for country in countries}
    percent_increase_gdp = []
    percent_increase_govt_revenue = []
    percent_oda = []

    for country in countries:
        percent_increase_gdp.append(costs[country] * 100 / gdp[country])
        percent_increase_govt_revenue.append(
            costs[country] * 100 / govt_revenue[country]
        )
        percent_oda.append(oda[country] * 100 / gdp[country])

    return (
        np.mean(percent_increase_gdp),
        np.mean(percent_increase_govt_revenue),
        np.mean(percent_oda),
    )


def get_extrapolation(countries, metadata, save_as=None):

    extrapolation = ExtrapolationResults(
        countries,
        insample_data_source="wpc",
        outofsample_data_source="wpc",
        metadata=metadata,
    )
    extrapolation.fit_regression_model()
    regression_r2 = extrapolation.score
    insample_df = extrapolation.get_in_sample_costs(survey_year=False, use_reg=True)
    insample_policy_cost = insample_df["Policy Cost"].loc["Total"]
    outofsample_df = extrapolation.get_out_of_sample_costs()
    outofsample_policy_cost = outofsample_df["Policy Cost"].loc["Total"]

    insample_oracle_cost = insample_df["Oracle Cost"].loc["Total"]
    outofsample_oracle_cost = outofsample_df["Oracle Cost"].loc["Total"]
    oracle_cost = insample_oracle_cost + outofsample_oracle_cost
    total_cost_out_of_sample_costs = insample_policy_cost + outofsample_policy_cost
    dropped_countries = extrapolation.dropped_countries
    extrapolated_cost = insample_policy_cost + outofsample_policy_cost

    return (
        extrapolated_cost,
        insample_policy_cost,
        total_cost_out_of_sample_costs,
        oracle_cost,
        dropped_countries,
        regression_r2,
    )


def plot_scatter_poverty_countries(countries, metadata, save_as):

    extrapolation = ExtrapolationResults(
        countries,
        insample_data_source="wpc",
        outofsample_data_source="wpc",
        metadata=metadata,
    )
    extrapolation.plot_figure(save_as=save_as)


def make_macro_file(countries, metadata, save_as):
    countries = sorted(countries)
    all_countries_string = make_string_country_list(countries, metadata=metadata)

    # SHARE WORLD'S POOR METRICS
    total_world_poor, malawi_world_poor = get_macros_share_world_poor(
        countries, metadata=metadata
    )

    # POVERTY RATES AND GAPS OF THE SAMPLE
    (
        initial_pov_rate,
        initial_pov_gap,
        min_pov_rate,
        max_pov_rate,
        min_country,
        max_country,
    ) = get_macros_survey_info(countries, metadata)

    # GET HEADLINE NUMBERS FOR GLOBAL POVERTY RATE TARGET
    national_poverty_rate_for_global = get_national_poverty_rate_target(metadata)
    global_cost, _ = get_headline_numbers(countries, metadata)

    # GET HEADLINE NUMBERS FOR NATIONAL POVERTY RATE TARGET
    national_cost, _ = get_headline_numbers(countries, metadata)
    global_poverty_rate_for_national = get_global_poverty_rate_target(metadata)

    # GET PERCENTAGES
    national_cost_by_country = {
        country: CountryMethodPovertyResults(
            country, method="continuous_gap", metadata=metadata
        )
        .rate_to_cost_interpolator(national_poverty_rate_for_global)
        .item()
        for country in countries
    }
    percent_increase_gdp, percent_increase_govt_revenue, percent_oda = get_percentages(
        countries,
        costs={country: national_cost_by_country[country] for country in countries},
        metadata=metadata,
    )

    # GET MALAWI HEADLINE NUMBERS
    malawi_cost, agg_results = get_headline_numbers(["MWI"], metadata=metadata)
    conversion_factor_malawi = (
        agg_results[0].country_results["MWI"]._get_conversion_factor()
    )

    malawi_variable_amt = malawi_cost["ubi_variable"] / conversion_factor_malawi

    # GET ORACLE RATIOS
    min_ratio, max_ratio = get_macros_oracle_ratios(countries, metadata=metadata)

    # GET EXTRAPOLATION
    (
        extrapolated_cost,
        _,
        total_cost_out_of_sample_costs,
        global_poverty_gap,
        dropped_countries,
        regression_r2,
    ) = get_extrapolation(countries, metadata=metadata, save_as=None)
    (
        percentage_oecd_gdp,
        percentage_oecd_plus_china_gdp,
        percentage_oecd_govt_revenue,
        percentage_oecd_plus_china_govt_revenue,
        percentage_feasible_global_gdp,
        percentage_oracle_global_gdp,
    ) = get_macros_relative_cost(
        extrapolated_cost, oracle_cost=global_poverty_gap, metadata=metadata
    )

    # GET POVERTY LINE COMPARISON
    (
        headlineGapNewPovertyLineNationalTarget,
        relativeContGapNewOldPovertyLineNationalTarget,
    ) = get_macro_povertyline_comparison(countries, metadata)

    dropped_countries_string = make_string_country_list(
        dropped_countries, metadata=metadata
    )

    malawi_n, malawi_d = get_data_dimension("MWI")
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
        f.write(
            "\\newcommand{\\sampleCostPercentGDP}"
            + f"{{{round(percent_increase_gdp)}}}\n"
        )
        f.write(
            "\\newcommand{\\sampleCostPercentGovtRevenue}"
            + f"{{{round(percent_increase_govt_revenue)}}}\n"
        )
        f.write("\\newcommand{\\sampleOdaPercentGDP}" + f"{{{round(percent_oda)}}}\n")
        f.write(
            "\\newcommand{\\nationalTarget}"
            + f"{{{int(metadata.nationalPovertyRate)}}}\n"
        )
        f.write(
            "\\newcommand{\\globalTarget}" + f"{{{int(metadata.globalPovertyRate)}}}\n"
        )
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
            + "{{{}}}\n".format(round(national_cost["ubi"]))
        )
        f.write(
            "\\newcommand{\\headlinePMTNationalTarget}"
            + "{{{}}}\n".format(round(national_cost["pmt"]))
        )
        f.write(
            "\\newcommand{\\headlineGapNationalTarget}"
            + "{{{}}}\n".format(round(national_cost["continuous_gap"]))
        )
        f.write(
            "\\newcommand{\\headlineOracleNationalTarget}"
            + "{{{}}}\n".format(round(national_cost["oracle_gap"]))
        )
        f.write(
            "\\newcommand{\\headlineBinaryGapNationalTarget}"
            + "{{{}}}\n".format(round(national_cost["binary_gap"]))
        )
        f.write(
            "\\newcommand{\\headlineUBIVariableNationalTarget}"
            + "{{{}}}\n".format(round(national_cost["ubi_variable"]))
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
            + "{{{}}}\n".format(round(global_cost["ubi"]))
        )
        f.write(
            "\\newcommand{\\headlinePMTGlobalTarget}"
            + "{{{}}}\n".format(round(global_cost["pmt"]))
        )
        f.write(
            "\\newcommand{\\headlineGapGlobalTarget}"
            + "{{{}}}\n".format(round(global_cost["continuous_gap"]))
        )
        f.write(
            "\\newcommand{\\headlineOracleGlobalTarget}"
            + "{{{}}}\n".format(round(global_cost["oracle_gap"]))
        )
        f.write(
            "\\newcommand{\\headlineBinaryGapGlobalTarget}"
            + "{{{}}}\n".format(round(global_cost["binary_gap"]))
        )
        f.write(
            "\\newcommand{\\headlineUBIVariableGlobalTarget}"
            + "{{{}}}\n".format(round(global_cost["ubi_variable"]))
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
                round((global_cost["continuous_gap"] / global_cost["oracle_gap"]))
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
            "\\newcommand{\\headlineGapNewPovertyLineNationalTarget}"
            + "{{{}}}\n".format(round(headlineGapNewPovertyLineNationalTarget))
        )
        f.write(
            "\\newcommand{\\relativeContGapNewOldPovertyLineNationalTarget}"
            + "{{{}}}\n".format(
                round(relativeContGapNewOldPovertyLineNationalTarget, 1)
            )
        )
        f.write(
            "\\newcommand{\\extrapolationCost}"
            + "{{{}}}\n".format(round(extrapolated_cost))
        )
        f.write(
            "\\newcommand{\\extrapolationRSquared}"
            + "{{{}}}\n".format(round(regression_r2, 2))
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
            + "{{{}}}\n".format(round(total_cost_out_of_sample_costs))
        )
        f.write(
            "\\newcommand{\\extrapolationGlobalGDP}"
            + "{{{}}}\n".format(round(percentage_feasible_global_gdp, 2))
        )
        f.write(
            "\\newcommand{\\oracleGlobalGDP}"
            + "{{{}}}\n".format(round(percentage_oracle_global_gdp, 2))
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
                round(malawi_cost["continuous_gap"] / malawi_cost["oracle_gap"])
            )
        )
        f.write(
            "\\newcommand{\\malawiGapUBIPercent}"
            + "{{{}}}\n".format(
                round((malawi_cost["continuous_gap"] * 100 / malawi_cost["ubi"]))
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
                round(malawi_cost["continuous_gap"] * 100 / malawi_cost["pmt"], 0)
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
