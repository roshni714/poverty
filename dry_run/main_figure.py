import os
from learn import (
    make_plot_for_country,
    make_plot_for_country_presentation,
    aggregate_plot,
    aggregate_plot_roshni_presentation,
    aggregate_plot_presentation,
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
    #    "oracle_gap",
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
    "colombia",
    "cote_divoire",
    "ethiopia",
    "ghana",
    "guinea_bissau",
    "kenya",
    "india",
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
        ubi_on=False,
    )


def get_headline_figure_2(povertyline, year):
    aggregate_plot(
        COUNTRIES,
        METHODS_HEADLINE,
        povertyline=povertyline,
        year=year,
        geo_extrapolation=True,
        save_as="exhibits/year={}/figs/paper-figure-2-headline".format(year),
        ubi_on=True,
    )


def get_rate_vs_gap_headline_figure_3(povertyline, year):
    aggregate_plot(
        COUNTRIES,
        METHODS_RATE_VS_GAP,
        povertyline=povertyline,
        year=year,
        geo_extrapolation=True,
        save_as="exhibits/year={}/figs/paper-figure-3-rate_gap_comparison".format(year),
        ubi_on=False,
    )


def get_oracle_ratio_figure_4(povertyline, year, nationalPovertyRate):
    plot_bar_chart_oracle_ratio(
        COUNTRIES,
        povertyline=povertyline,
        year=year,
        nationalPovertyRate=nationalPovertyRate,
        save_as="exhibits/year={}/figs/paper-figure-4-oracle_ratio".format(year),
    )


def get_ubi_ratio_figure_5(povertyline, year, nationalPovertyRate):
    plot_bar_chart_ubi_ratio(
        COUNTRIES,
        povertyline=povertyline,
        year=year,
        nationalPovertyRate=nationalPovertyRate,
        save_as="exhibits/year={}/figs/paper-figure-5-ubi_ratio".format(year),
    )


def get_gdp_plot_figure_6(povertyline, year, nationalPovertyRate):
    plot_bar_chart_policy_amt_as_percent_of_gdp(
        COUNTRIES,
        True,
        povertyline=povertyline,
        year=year,
        nationalPovertyRate=nationalPovertyRate,
        save_as="exhibits/year={}/figs/paper-figure-6-policy_cost_gdp".format(year),
    )


def get_appendix_table_survey(year):
    get_table_survey_info(
        COUNTRIES,
        year=year,
        save_as="exhibits/year={}/tables/appendix-table-1-survey_info".format(year),
    )
    get_table_survey_info(
        COUNTRIES,
        year=year,
        slides=True,
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
            ubi_on=True,
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
    total_cost, _, _, _, _ = get_extrapolation(
        COUNTRIES,
        povertyline=povertyline,
        year=year,
        globalPovertyRate=globalPovertyRate,
        save_as="exhibits/year={}/tables/appendix-table-4-extrapolation".format(year),
    )


def make_presentation_figures(povertyline, year):
    make_plot_for_country_presentation(
        "malawi",
        [],
        povertyline=povertyline,
        year=year,
        geo_extrapolation=True,
        save_as="exhibits/year={}/presentation/figure-0-malawi_0".format(year),
        ubi_on=False,
    )

    make_plot_for_country_presentation(
        "malawi",
        [],
        povertyline=povertyline,
        year=year,
        geo_extrapolation=True,
        save_as="exhibits/year={}/presentation/figure-0-malawi_1".format(year),
        ubi_on=True,
    )

    make_plot_for_country_presentation(
        "malawi",
        ["oracle_gap"],
        povertyline=povertyline,
        year=year,
        geo_extrapolation=True,
        save_as="exhibits/year={}/presentation/figure-0-malawi_2".format(year),
        ubi_on=True,
    )

    make_plot_for_country_presentation(
        "malawi",
        ["pmt", "continuous_rate", "continuous_gap", "oracle_gap"],
        povertyline=povertyline,
        year=year,
        geo_extrapolation=True,
        save_as="exhibits/year={}/presentation/figure-0-malawi_3".format(year),
        ubi_on=True,
    )

    aggregate_plot_roshni_presentation(
        COUNTRIES,
        show_method_list=[],
        povertyline=povertyline,
        year=year,
        geo_extrapolation=True,
        ubi_on=False,
        save_as="exhibits/year={}/presentation/figure-1-build-0".format(year),
        vertical_arrow_rate=False,
        vertical_arrow_gap=False,
    )

    aggregate_plot_roshni_presentation(
        COUNTRIES,
        show_method_list=["oracle_gap"],
        povertyline=povertyline,
        year=year,
        geo_extrapolation=True,
        ubi_on=True,
        save_as="exhibits/year={}/presentation/figure-1-build-1".format(year),
        vertical_arrow_rate=False,
        vertical_arrow_gap=False,
    )

    aggregate_plot_roshni_presentation(
        COUNTRIES,
        show_method_list=["pmt", "continuous_rate", "continuous_gap", "oracle_gap"],
        povertyline=povertyline,
        year=year,
        geo_extrapolation=True,
        ubi_on=True,
        save_as="exhibits/year={}/presentation/figure-1-build-2".format(year),
        vertical_arrow_rate=True,
        vertical_arrow_gap=False,
    )

    aggregate_plot_roshni_presentation(
        COUNTRIES,
        show_method_list=["binary_gap", "continuous_gap"],
        povertyline=povertyline,
        year=year,
        geo_extrapolation=True,
        ubi_on=False,
        save_as="exhibits/year={}/presentation/figure-1-build-3".format(year),
        vertical_arrow_rate=True,
        vertical_arrow_gap=False,
    )

    aggregate_plot_roshni_presentation(
        COUNTRIES,
        show_method_list=["ubi", "binary_gap", "continuous_gap"],
        povertyline=povertyline,
        year=year,
        geo_extrapolation=True,
        ubi_on=False,
        save_as="exhibits/year={}/presentation/figure-1-build-4".format(year),
        vertical_arrow_rate=True,
        vertical_arrow_gap=False,
    )

    aggregate_plot_presentation(
        COUNTRIES,
        show_method_list=["continuous_gap", "oracle_gap"],
        povertyline=povertyline,
        year=year,
        geo_extrapolation=True,
        ubi_on=False,
        save_as="exhibits/year={}/presentation/figure-1-feasible_vs_oracle_all_series_0".format(
            year
        ),
        vertical_arrow_rate=False,
        vertical_arrow_gap=False,
    )
    aggregate_plot_presentation(
        COUNTRIES,
        show_method_list=["continuous_gap", "ubi"],
        povertyline=povertyline,
        year=year,
        geo_extrapolation=True,
        ubi_on=True,
        save_as="exhibits/year={}/presentation/figure-2-feasible_vs_ubi_all_series_0".format(
            year
        ),
        vertical_arrow_rate=False,
        vertical_arrow_gap=False,
    )

    aggregate_plot_presentation(
        COUNTRIES,
        povertyline=povertyline,
        year=year,
        show_method_list=["continuous_gap", "binary_gap"],
        geo_extrapolation=True,
        ubi_on=False,
        save_as="exhibits/year={}/presentation/figure-3-cont_gap_vs_binary_gap_all_series_0".format(
            year
        ),
        vertical_arrow_rate=False,
        vertical_arrow_gap=False,
    )

    aggregate_plot_presentation(
        COUNTRIES,
        show_method_list=["continuous_gap", "oracle_gap"],
        povertyline=povertyline,
        year=year,
        geo_extrapolation=True,
        ubi_on=False,
        save_as="exhibits/year={}/presentation/figure-1-feasible_vs_oracle_all_series_1".format(
            year
        ),
        vertical_arrow_rate=True,
        vertical_arrow_gap=False,
    )
    aggregate_plot_presentation(
        COUNTRIES,
        show_method_list=["continuous_gap", "ubi"],
        povertyline=povertyline,
        year=year,
        geo_extrapolation=True,
        ubi_on=True,
        save_as="exhibits/year={}/presentation/figure-2-feasible_vs_ubi_all_series_1".format(
            year
        ),
        vertical_arrow_rate=True,
        vertical_arrow_gap=False,
    )
    aggregate_plot_presentation(
        COUNTRIES,
        povertyline=povertyline,
        year=year,
        show_method_list=["continuous_gap", "binary_gap"],
        geo_extrapolation=True,
        ubi_on=False,
        save_as="exhibits/year={}/presentation/figure-3-cont_gap_vs_binary_gap_all_series_1".format(
            year
        ),
        vertical_arrow_rate=True,
        vertical_arrow_gap=False,
    )

    aggregate_plot_presentation(
        COUNTRIES,
        show_method_list=["continuous_gap", "oracle_gap"],
        povertyline=povertyline,
        year=year,
        geo_extrapolation=True,
        ubi_on=False,
        save_as="exhibits/year={}/presentation/figure-1-feasible_vs_oracle_all_series_2".format(
            year
        ),
        vertical_arrow_rate=True,
        vertical_arrow_gap=True,
    )
    aggregate_plot_presentation(
        COUNTRIES,
        show_method_list=["continuous_gap", "ubi"],
        povertyline=povertyline,
        year=year,
        geo_extrapolation=True,
        ubi_on=True,
        save_as="exhibits/year={}/presentation/figure-2-feasible_vs_ubi_all_series_2".format(
            year
        ),
        vertical_arrow_rate=True,
        vertical_arrow_gap=True,
    )
    aggregate_plot_presentation(
        COUNTRIES,
        povertyline=povertyline,
        year=year,
        show_method_list=["continuous_gap", "binary_gap"],
        geo_extrapolation=True,
        ubi_on=False,
        save_as="exhibits/year={}/presentation/figure-3-cont_gap_vs_binary_gap_all_series_2".format(
            year
        ),
        vertical_arrow_rate=True,
        vertical_arrow_gap=True,
    )


@argh.arg("--year", help="Year to generate figures for", type=int, default=2021)
@argh.arg(
    "--nationalPovertyRate", help="National poverty rate target", type=float, default=1
)
@argh.arg(
    "--globalPovertyRate", help="Global poverty rate target", type=float, default=1
)
def main(year=2021, nationalPovertyRate=1, globalPovertyRate=1):
    os.makedirs(f"exhibits/year={year}/figs", exist_ok=True)
    os.makedirs(f"exhibits/year={year}/tables", exist_ok=True)
    os.makedirs(f"exhibits/year={year}/presentation", exist_ok=True)

    if year == 2017:
        povertyline = 2.15
    elif year == 2021:
        povertyline = 3.0

    get_malawi_rate_vs_gap_figure_1(povertyline, year)
    get_headline_figure_2(povertyline, year)
    get_rate_vs_gap_headline_figure_3(povertyline, year)
    get_oracle_ratio_figure_4(
        povertyline, year, nationalPovertyRate=nationalPovertyRate
    )
    get_ubi_ratio_figure_5(povertyline, year, nationalPovertyRate=nationalPovertyRate)
    get_gdp_plot_figure_6(povertyline, year, nationalPovertyRate=nationalPovertyRate)
    get_headline_figure_3_dollar(year)
    get_table_extrapolation(povertyline, year, globalPovertyRate=globalPovertyRate)
    # get_appendix_table_out_of_sample_rmse()
    get_appendix_table_survey(year)
    get_appendix_table_wpc(year)
    get_appendix_table_policy_cost_insample(
        povertyline, year, globalPovertyRate=globalPovertyRate
    )
    get_country_level_analysis(povertyline, year)
    make_macro_file(
        COUNTRIES,
        povertyline,
        year,
        nationalPovertyRate=nationalPovertyRate,
        globalPovertyRate=globalPovertyRate,
        save_as="exhibits/empirical_macros",
    )
    make_presentation_figures(povertyline=povertyline, year=year)


if __name__ == "__main__":
    _parser = argh.ArghParser()
    _parser.add_commands([main])
    _parser.dispatch()
