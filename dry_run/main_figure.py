import os
from learn import (
    make_plot_for_country,
    aggregate_plot,
    plot_bar_chart_policy_amt_as_percent_of_gdp,
    get_table_policy_cost_gdp,
    plot_bar_chart_ubi_ratio,
    plot_bar_chart_oracle_ratio,
    get_table_wpc,
    get_extrapolation,
    get_table_survey_info,
    make_macro_file,
)
import argh

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
    # "modern_pmt",
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
    "kenya",
    "malawi",
    "mali",
    "niger",
    "nigeria",
    "senegal",
    "south_africa",
    "south_sudan",
    "tanzania",
    "togo",
    "uganda",
]


def get_malawi_rate_vs_gap_figure_1(povertyline, year):
    make_plot_for_country(
        "malawi",
        METHODS_RATE_VS_GAP,
        povertyline=povertyline,
        year=year,
        geo_extrapolation=True,
        save_as="exhibits/year={}/figs/paper-figure-1-malawi_rate_vs_gap".format(year),
        ubi_off=True,
    )


def get_headline_figure_2(povertyline, year):
    aggregate_plot(
        COUNTRIES,
        METHODS_HEADLINE,
        povertyline=povertyline,
        year=year,
        geo_extrapolation=True,
        save_as="exhibits/year={}/figs/paper-figure-2-headline".format(year),
    )


def get_rate_vs_gap_headline_figure_3(povertyline, year):
    aggregate_plot(
        COUNTRIES,
        METHODS_RATE_VS_GAP,
        povertyline=povertyline,
        year=year,
        geo_extrapolation=True,
        save_as="exhibits/year={}/figs/paper-figure-3-rate_gap_comparison".format(year),
    )


def get_oracle_ratio_figure_4(povertyline, year, globalPovertyRate):
    plot_bar_chart_oracle_ratio(
        COUNTRIES,
        povertyline=povertyline,
        year=year,
        globalPovertyRate=globalPovertyRate,
        save_as="exhibits/year={}/figs/paper-figure-4-oracle_ratio".format(year),
    )


def get_ubi_ratio_figure_5(povertyline, year, globalPovertyRate):
    plot_bar_chart_ubi_ratio(
        COUNTRIES,
        povertyline=povertyline,
        year=year,
        globalPovertyRate=globalPovertyRate,
        save_as="exhibits/year={}/figs/paper-figure-5-ubi_ratio".format(year),
    )


def get_gdp_plot_figure_6(povertyline, year, globalPovertyRate):
    plot_bar_chart_policy_amt_as_percent_of_gdp(
        COUNTRIES,
        True,
        povertyline=povertyline,
        year=year,
        globalPovertyRate=globalPovertyRate,
        save_as="exhibits/year={}/figs/paper-figure-6-policy_cost_gdp".format(year),
    )


def get_appendix_table_survey(year):
    get_table_survey_info(
        COUNTRIES,
        save_as="exhibits/year={}/tables/appendix-table-1-survey_info".format(year),
    )


def get_appendix_table_wpc(year):
    get_table_wpc(
        COUNTRIES, save_as="exhibits/year={}/tables/appendix-table-2-wpc".format(year)
    )


def get_appendix_table_policy_cost_insample(povertyline, year, globalPovertyRate):
    get_table_policy_cost_gdp(
        COUNTRIES,
        povertyline=povertyline,
        year=year,
        globalPovertyRate=globalPovertyRate,
        save_as="exhibits/year={}/tables/appendix-table-3-policy_cost_gdp_oda".format(
            year
        ),
    )


# def get_appendix_table_out_of_sample_rmse(povertyline, year):
#     get_table_out_of_sample_rmse(
#         COUNTRIES,
#         povertyline=povertyline,
#         year=year,
#         save_as="exhibits/year={}/tables/appendix-table-2-out_of_sample_rmse".format(
#             year
#         ),
#     )


# def get_appendix_table_policy_cost_out_of_sample(povertyline, year):
#     get_extrapolation(
#         COUNTRIES,
#         povertyline=povertyline,
#         year=year,
#         save_as="exhibits/year={}/tables/appendix-table-4-policy_cost_out_of_sample".format(
#             year
#         ),
#     )


def get_country_level_analysis(povertyline, year):
    for country in COUNTRIES:
        make_plot_for_country(
            country,
            METHODS_ALL,
            geo_extrapolation=True,
            povertyline=povertyline,
            year=year,
            save_as="exhibits/year={}/figs/appendix-figure-{}".format(year, country),
            ubi_off=False,
        )


def get_headline_figure_3_dollar(year):
    aggregate_plot(
        COUNTRIES,
        METHODS_HEADLINE,
        povertyline=3,
        year=2021,
        geo_extrapolation=True,
        save_as="exhibits/year={}/figs/paper-figure-6-new_povertyline_headline".format(
            year
        ),
    )


def get_table_extrapolation(povertyline, year, globalPovertyRate):
    total_cost, _, _, _ = get_extrapolation(
        COUNTRIES,
        povertyline=povertyline,
        year=year,
        globalPovertyRate=globalPovertyRate,
    )


def make_presentation_figures(povertyline, year):
    aggregate_plot(
        COUNTRIES,
        ["ubi"],
        povertyline=povertyline,
        year=year,
        geo_extrapolation=True,
        save_as="exhibits/presentation/year={}/figure-1-ubi".format(year),
    )
    aggregate_plot(
        COUNTRIES,
        ["ubi", "oracle_gap"],
        povertyline=povertyline,
        year=year,
        geo_extrapolation=True,
        save_as="exhibits/presentation/figure-2-oracle",
    )
    aggregate_plot(
        COUNTRIES,
        povertyline=povertyline,
        year=year,
        methods=["ubi", "modern_pmt", "pmt", "oracle_gap"],
        geo_extrapolation=True,
        save_as="exhibits/presentation/figure-3-modern_pmt-pmt",
    )
    aggregate_plot(
        COUNTRIES,
        ["ubi", "modern_pmt", "pmt", "continuous_gap", "oracle_gap"],
        geo_extrapolation=True,
        save_as="exhibits/presentation/figure-4-gap",
    )
    aggregate_plot(
        COUNTRIES,
        ["ubi", "modern_pmt", "pmt", "binary_gap", "continuous_gap", "oracle_gap"],
        geo_extrapolation=True,
        save_as="exhibits/presentation/figure-5-binary-gap",
    )


@argh.arg("--year", help="Year to generate figures for", type=int, default=2021)
@argh.arg(
    "--nationalPovertyRate", help="National poverty rate target", type=float, default=2
)
@argh.arg(
    "--globalPovertyRate", help="Global poverty rate target", type=float, default=2
)
def main(year=2021, nationalPovertyRate=2, globalPovertyRate=2):
    os.makedirs(f"exhibits/year={year}/figs", exist_ok=True)
    os.makedirs(f"exhibits/year={year}/tables", exist_ok=True)

    if year == 2017:
        povertyline = 2.15
    elif year == 2021:
        povertyline = 3.0

    get_malawi_rate_vs_gap_figure_1(povertyline, year)
    get_headline_figure_2(povertyline, year)
    get_rate_vs_gap_headline_figure_3(povertyline, year)
    get_oracle_ratio_figure_4(povertyline, year, globalPovertyRate=globalPovertyRate)
    get_ubi_ratio_figure_5(povertyline, year, globalPovertyRate=globalPovertyRate)
    get_gdp_plot_figure_6(povertyline, year, globalPovertyRate=globalPovertyRate)
    get_headline_figure_3_dollar(year)
    # get_extrapolation(povertyline, year, globalPovertyRate=globalPovertyRate)
    # get_appendix_table_out_of_sample_rmse()
    get_appendix_table_survey(year)
    get_appendix_table_wpc(year)
    get_appendix_table_policy_cost_insample(
        povertyline, year, globalPovertyRate=globalPovertyRate
    )
    # get_appendix_table_policy_cost_out_of_sample()
    # get_appendix_table_oecd()
    # get_appendix_table_oecd_plus_china()
    # get_appendix_table_diff_between_ubi_and_targeting()
    get_country_level_analysis(povertyline, year)
    make_macro_file(
        COUNTRIES,
        povertyline,
        year,
        nationalPovertyRate=nationalPovertyRate,
        globalPovertyRate=globalPovertyRate,
        save_as="exhibits/empirical_macros",
    )
    # make_presentation_figures()


if __name__ == "__main__":
    _parser = argh.ArghParser()
    _parser.add_commands([main])
    _parser.dispatch()
