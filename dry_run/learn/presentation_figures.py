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

    # Generate results for all methods
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
    fontsize = 30

    oracle_results = CountryMethodPovertyResults(
        country,
        "oracle_gap",
        metadata,
    )
    fig, ax = plt.subplots(1, 2, figsize=(24, 8))

    # From any of the CountryMethodPovertyResults,
    # get initial poverty rate and initial poverty gap index of country
    xlims = [oracle_results.initial_rate, oracle_results.initial_gap_index]

    # Basic plot formatting
    # Set xlims and ylims of plot
    # Also plot UBI cost line by getting cost from any of the CountryMethodPovertyResultsif ubi_on is True
    for i in range(2):
        ax[i].set_xlim(-xlims[i] * 0.05, xlims[i] * 1.05)
        ax[i].set_ylim(-0.1, oracle_results.get_ubi_cost() * 1.05)
        ax[i].set_ylabel("Policy Cost ($ Billion Per Year)", fontsize=fontsize)
        ax[i].grid(True)
        ax[i].tick_params(axis="x", labelsize=fontsize * 0.75)
        ax[i].tick_params(axis="y", labelsize=fontsize * 0.75)
        if ubi_on:
            ax[i].plot(
                np.linspace(0.0, xlims[i]),
                np.ones(50) * oracle_results.get_ubi_cost(),
                linestyle="--",
                color=METHODS["ubi_standard"]["color"],
                label=METHODS["ubi_standard"]["name"]
                + " (${})".format(metadata.povertyline),
                linewidth=3,
            )

    # Draw a vertical line at national poverty rate target
    ax[0].vlines(
        x=metadata.nationalPovertyRate,
        color="grey",
        linestyle="solid",
        linewidth=3,
        ymin=0,
        ymax=oracle_results.get_ubi_cost() * 1.05,
    )

    # For each method add line to the rate-cost subplot and gap-cost subplot
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
            df["policy_cost_per_capita"] * oracle_results.conversion_factor
        )

        xs = [rates, gaps]
        for j in range(2):
            if method != "oracle_gap":
                ax[j].plot(
                    xs[j],
                    costs,
                    marker="o",
                    label=dic["name"],
                    color=dic["color"],
                    linestyle=dic["linestyle"],
                    linewidth=3,
                )

            # For presentations only plot a horizontal line for the aggregate poverty gap.
            if method == "oracle_gap":
                ax[j].plot(
                    xs[j],
                    max(costs) * np.ones(len(xs[j])),
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

    ax[0].legend(fontsize=fontsize * 0.75)  # , #bbox_to_anchor=(1.05, 0.5)
    # fig.tight_layout(rect=[0, 0, 0.85, 1])
    plt.tight_layout()
    plt.savefig("{}.pdf".format(save_as), bbox_inches="tight")
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
