import os
from learn import (
    make_plot_for_country,
    aggregate_plot,
    plot_bar_chart_policy_amt_as_percent_of_gdp,
    get_table_policy_cost_gdp_oda,
    get_table_oecd,
    get_table_oecd_plus_china,
    get_table_diff_between_ubi_and_targeting,
    plot_bar_chart_ubi_ratio,
    plot_bar_chart_oracle_ratio,
    get_table_out_of_sample_rmse,
    get_extrapolation,
    make_macro_file,
)

METHODS_ALL = [
    "ubi",
    "modern_pmt",
    "pmt",
    "binary_gap",
    "continuous_gap",
    "binary_rate",
    "continuous_rate",
    "oracle_gap",
]
METHODS_HEADLINE = [
    "ubi",
    "modern_pmt",
    "pmt",
    "binary_gap",
    "continuous_gap",
    "oracle_gap",
]

METHODS_RATE_VS_GAP = ["binary_gap", "continuous_gap", "binary_rate", "continuous_rate"]
METHODS_RATE_VS_GAP_CONT = ["continuous_gap", "continuous_rate"]

COUNTRIES = [
    "benin",
    "burkina_faso",
    "cote_divoire",
    "ghana",
    "guinea_bissau",
    "senegal",
    "mali",
    "malawi",
    "niger",
    "nigeria",
    "senegal",
]


def get_malawi_rate_vs_gap_figure_1():
    make_plot_for_country(
        "malawi",
        METHODS_RATE_VS_GAP,
        geo_extrapolation=True,
        save_as="exhibits/figs/paper-figure-1-malawi_rate_vs_gap",
        ubi_off=True,
    )


def get_malawi_rate_vs_gap_figure_1a_cont():
    make_plot_for_country(
        "malawi",
        METHODS_RATE_VS_GAP_CONT,
        geo_extrapolation=True,
        save_as="exhibits/figs/presentation-figure-malawi_rate_vs_gap_cont",
        ubi_off=True,
    )


def get_malawi_headline_figure_2():
    make_plot_for_country(
        "malawi",
        METHODS_HEADLINE,
        geo_extrapolation=True,
        save_as="exhibits/figs/paper-figure-2-malawi_headline",
        ubi_off=False,
    )


def get_headline_figure_3():
    aggregate_plot(
        COUNTRIES,
        METHODS_HEADLINE,
        geo_extrapolation=True,
        save_as="exhibits/figs/paper-figure-3-headline",
    )


def get_rate_vs_gap_headline_figure_4():
    aggregate_plot(
        COUNTRIES,
        METHODS_RATE_VS_GAP,
        geo_extrapolation=True,
        save_as="exhibits/figs/paper-figure-4-rate_gap_comparison",
    )


def get_oracle_ratio_figure_5():
    plot_bar_chart_oracle_ratio(
        COUNTRIES, save_as="exhibits/figs/paper-figure-5-oracle_ratio"
    )


def get_ubi_ratio_figure_6():
    plot_bar_chart_ubi_ratio(
        COUNTRIES, save_as="exhibits/figs/paper-figure-6-ubi_ratio"
    )


def get_gdp_plot_figure_7():
    plot_bar_chart_policy_amt_as_percent_of_gdp(
        COUNTRIES, True, save_as="exhibits/figs/paper-figure-7-policy_cost_gdp"
    )


def get_appendix_table_out_of_sample_rmse():
    get_table_out_of_sample_rmse(
        COUNTRIES,
        save_as="exhibits/tables/appendix-table-2-out_of_sample_rmse",
    )


def get_appendix_table_policy_cost_insample():
    get_table_policy_cost_gdp_oda(
        COUNTRIES, save_as="exhibits/tables/appendix-table-3-policy_cost_gdp_oda"
    )


def get_appendix_table_policy_cost_out_of_sample():
    get_extrapolation(
        COUNTRIES, save_as="exhibits/tables/appendix-table-4-policy_cost_out_of_sample"
    )


def get_appendix_table_oecd():
    total_cost, _, _, _ = get_extrapolation(COUNTRIES)
    get_table_oecd(total_cost, save_as="exhibits/tables/appendix-table-5-oecd")


def get_appendix_table_oecd_plus_china():
    total_cost, _, _, _ = get_extrapolation(COUNTRIES)
    get_table_oecd_plus_china(
        total_cost, save_as="exhibits/tables/appendix-table-6-oecd_plus_china"
    )


def get_appendix_table_diff_between_ubi_and_targeting():
    get_table_diff_between_ubi_and_targeting(
        COUNTRIES,
        save_as="exhibits/tables/appendix-table-7-diff_between_ubi_and_targeting",
    )


def get_country_level_analysis():
    for country in COUNTRIES:
        make_plot_for_country(
            country,
            METHODS_ALL,
            geo_extrapolation=True,
            save_as="exhibits/figs/appendix-figure-{}".format(country),
            ubi_off=False,
        )


def make_presentation_figures():
    aggregate_headline_rate_plot(
        COUNTRIES,
        ["ubi"],
        geo_extrapolation=True,
        save_as="exhibits/figs/presentation-figure-1-ubi",
    )
    aggregate_headline_rate_plot(
        COUNTRIES,
        ["ubi", "oracle_gap"],
        geo_extrapolation=True,
        save_as="exhibits/figs/presentation-figure-2-oracle",
    )
    aggregate_headline_rate_plot(
        COUNTRIES,
        ["ubi", "modern_pmt", "pmt", "oracle_gap"],
        geo_extrapolation=True,
        save_as="exhibits/figs/presentation-figure-3-modern_pmt-pmt",
    )
    aggregate_headline_rate_plot(
        COUNTRIES,
        ["ubi", "modern_pmt", "pmt", "continuous_gap", "oracle_gap"],
        geo_extrapolation=True,
        save_as="exhibits/figs/presentation-figure-4-gap",
    )
    aggregate_headline_rate_plot(
        COUNTRIES,
        ["ubi", "modern_pmt", "pmt", "binary_gap", "continuous_gap", "oracle_gap"],
        geo_extrapolation=True,
        save_as="exhibits/figs/presentation-figure-5-binary-gap",
    )


if __name__ == "__main__":
    os.makedirs("exhibits", exist_ok=True)
    os.makedirs("exhibits/figs", exist_ok=True)
    os.makedirs("exhibits/tables", exist_ok=True)
    get_malawi_rate_vs_gap_figure_1()
    # get_malawi_rate_vs_gap_figure_1a_cont()
    get_malawi_headline_figure_2()
    get_headline_figure_3()
    # get_ubi_ratio_figure_6()
    # get_oracle_ratio_figure_5()
    get_rate_vs_gap_headline_figure_4()
    # get_gdp_plot_figure_7()
    # get_appendix_table_out_of_sample_rmse()
    # get_appendix_table_policy_cost_insample()
    # get_appendix_table_policy_cost_out_of_sample()
    # get_appendix_table_oecd()
    # get_appendix_table_oecd_plus_china()
    # get_appendix_table_diff_between_ubi_and_targeting()
    get_country_level_analysis()
    # make_macro_file(COUNTRIES, save_as="exhibits/empirical_macros")
    # make_presentation_figures()
