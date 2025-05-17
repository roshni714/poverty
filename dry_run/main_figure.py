import os
from learn import (
    make_plot_for_country,
    aggregate_plot_x_axis_population_weighted_poverty_measure_global_gap,
    aggregate_plot_x_axis_population_weighted_poverty_measure,
    plot_bar_chart_policy_amt_as_percent_of_gdp,
    get_table_policy_cost_gdp_oda,
)

METHODS_ALL = [
    "ubi",
    "pmt",
    "modern_pmt",
    "oracle_gap",
    "binary_gap",
    "continuous_gap",
    "binary_rate",
    "continuous_rate",
]
METHODS_HEADLINE = [
    "ubi",
    "pmt",
    "modern_pmt",
    "binary_gap",
    "continuous_gap",
    "oracle_gap",
]
METHODS_RATE_VS_GAP = ["binary_gap", "continuous_gap", "binary_rate", "continuous_rate"]


def get_malawi_rate_vs_gap_figure_1():
    make_plot_for_country(
        "malawi",
        METHODS_RATE_VS_GAP,
        geo_extrapolation=True,
        save_as="figs/paper-figure-1-malawi_rate_vs_gap",
        ubi_off=False,
    )


def get_malawi_headline_figure_2():
    make_plot_for_country(
        "malawi",
        METHODS_HEADLINE,
        geo_extrapolation=True,
        save_as="figs/paper-figure-2-malawi_headline",
        ubi_off=False,
    )


def get_headline_figure_3():
    countries = os.listdir("learn/results")
    aggregate_plot_x_axis_population_weighted_poverty_measure_global_gap(
        countries,
        METHODS_HEADLINE,
        geo_extrapolation=True,
        save_as="figs/paper-figure-3-headline",
    )


def get_rate_vs_gap_headline_figure_4():
    countries = os.listdir("learn/results")
    aggregate_plot_x_axis_population_weighted_poverty_measure(
        countries,
        METHODS_RATE_VS_GAP,
        geo_extrapolation=True,
        save_as="figs/paper-figure-4-rate_gap_comparison",
    )


def get_gdp_plot_figure_5():
    countries = os.listdir("learn/results")
    plot_bar_chart_policy_amt_as_percent_of_gdp(
        countries, True, save_as="figs/paper-figure-6-policy_cost_gdp"
    )


def get_appendix_table_2():
    countries = os.listdir("learn/results")
    get_table_policy_cost_gdp_oda(
        countries, True, save_as="tables/paper-table-appendix-2-policy_cost_gdp_oda"
    )


def get_appendix_table_3():
    countries = os.listdir("learn/results")


def get_country_level_analysis():
    countries = os.listdir("learn/results")
    for country in countries:
        make_plot_for_country(
            country,
            METHODS_ALL,
            geo_extrapolation=True,
            save_as="app-figure-{}",
            ubi_off=False,
        )


if __name__ == "__main__":
    os.makedirs("figs", exist_ok=True)
    os.makedirs("tables", exist_ok=True)
    get_malawi_rate_vs_gap_figure_1()
    get_malawi_headline_figure_2()
    get_headline_figure_3()
    get_rate_vs_gap_headline_figure_4()
    get_gdp_plot_figure_5()
    get_appendix_table_2()
    get_country_level_analysis()
