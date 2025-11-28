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
    plot_scatter_poverty_countries,
    Metadata,
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
    "BEN",
    "BFA",
    "BGD",
    "CIV",
    "ETH",
    "GHA",
    "GNB",
    "KEN",
    "MDG",
    "MLI",
    "MWI",
    "NER",
    "RWA",
    "SEN",
    "TGO",
    "TZA",
    "UGA",
    "ZAF",
]


def get_malawi_rate_vs_gap_figure_1(metadata):
    make_plot_for_country(
        "MWI",
        METHODS_RATE_VS_GAP,
        metadata,
        save_as="exhibits/year={}/figs/paper-figure-1-malawi_rate_vs_gap".format(
            metadata.year
        ),
        ubi_on=False,
    )


def get_headline_figure_2(metadata):
    aggregate_plot(
        COUNTRIES,
        METHODS_HEADLINE,
        metadata=metadata,
        save_as="exhibits/year={}/figs/paper-figure-2-headline".format(metadata.year),
        ubi_on=True,
    )


def get_rate_vs_gap_headline_figure_3(metadata):
    aggregate_plot(
        COUNTRIES,
        METHODS_RATE_VS_GAP,
        metadata=metadata,
        save_as="exhibits/year={}/figs/paper-figure-3-rate_gap_comparison".format(
            metadata.year
        ),
        ubi_on=False,
    )


def get_oracle_ratio_figure_4(metadata):
    plot_bar_chart_oracle_ratio(
        COUNTRIES,
        metadata=metadata,
        save_as="exhibits/year={}/figs/paper-figure-4-oracle_ratio".format(
            metadata.year
        ),
    )


def get_ubi_ratio_figure_5(metadata):
    plot_bar_chart_ubi_ratio(
        COUNTRIES,
        metadata=metadata,
        save_as="exhibits/year={}/figs/paper-figure-5-ubi_ratio".format(metadata.year),
    )


def get_gdp_plot_figure_6(metadata):
    plot_bar_chart_policy_amt_as_percent_of_gdp(
        COUNTRIES,
        metadata=metadata,
        save_as="exhibits/year={}/figs/paper-figure-6-policy_cost_gdp".format(
            metadata.year
        ),
    )


def get_scatter_figure_7(metadata):
    plot_scatter_poverty_countries(
        COUNTRIES,
        metadata=metadata,
        save_as="exhibits/year={}/figs/paper-figure-7-scatter".format(metadata.year),
    )


def get_appendix_table_survey(metadata):
    get_table_survey_info(
        COUNTRIES,
        metadata=metadata,
        save_as="exhibits/year={}/tables/appendix-table-1-survey_info".format(
            metadata.year
        ),
    )
    get_table_survey_info(
        COUNTRIES,
        metadata=metadata,
        slides=True,
        save_as="exhibits/year={}/tables/appendix-table-1-survey_info".format(
            metadata.year
        ),
    )


def get_appendix_table_wpc(metadata):
    get_table_wpc(
        COUNTRIES,
        metadata=metadata,
        save_as="exhibits/year={}/tables/appendix-table-2-wpc".format(metadata.year),
    )


def get_appendix_table_policy_cost_insample(metadata):
    get_table_policy_cost_gdp(
        COUNTRIES,
        metadata=metadata,
        save_as="exhibits/year={}/tables/appendix-table-3-policy_cost_gdp_oda".format(
            metadata.year
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


def get_country_level_analysis(metadata):
    for country in COUNTRIES:
        make_plot_for_country(
            country,
            METHODS_ALL,
            metadata=metadata,
            save_as="exhibits/year={}/figs/appendix-figure-{}".format(
                metadata.year, country
            ),
            ubi_on=True,
        )


def get_headline_figure_3_dollar(metadata):

    new_metadata = Metadata(
        povertyline=3,
        year=2021,
        nationalPovertyRate=metadata.nationalPovertyRate,
        globalPovertyRate=metadata.globalPovertyRate,
        auxpath=metadata.auxpath,
        secondaryauxpath=metadata.secondaryauxpath,
        wpcpath=metadata.wpcpath,
    )
    aggregate_plot(
        COUNTRIES,
        METHODS_HEADLINE,
        metadata=new_metadata,
        save_as="exhibits/year={}/figs/paper-figure-6-new_povertyline_headline".format(
            metadata.year
        ),
    )


def get_table_extrapolation(metadata):
    total_cost, _, _, _, _, _ = get_extrapolation(
        COUNTRIES,
        metadata=metadata,
        save_as="exhibits/year={}/tables/appendix-table-4-extrapolation".format(
            metadata.year
        ),
    )


def make_presentation_figures(metadata):
    make_plot_for_country_presentation(
        "MWI",
        [],
        metadata=metadata,
        save_as="exhibits/year={}/presentation/figure-0-malawi_0".format(metadata.year),
        ubi_on=False,
    )

    make_plot_for_country_presentation(
        "MWI",
        [],
        metadata=metadata,
        save_as="exhibits/year={}/presentation/figure-0-malawi_1".format(metadata.year),
        ubi_on=True,
    )

    make_plot_for_country_presentation(
        "MWI",
        ["oracle_gap"],
        metadata=metadata,
        save_as="exhibits/year={}/presentation/figure-0-malawi_2".format(metadata.year),
        ubi_on=True,
    )

    make_plot_for_country_presentation(
        "MWI",
        ["pmt", "continuous_gap", "oracle_gap"],
        metadata=metadata,
        save_as="exhibits/year={}/presentation/figure-0-malawi_3".format(metadata.year),
        ubi_on=True,
    )

    aggregate_plot_roshni_presentation(
        COUNTRIES,
        show_method_list=[],
        metadata=metadata,
        ubi_on=False,
        save_as="exhibits/year={}/presentation/figure-1-build-0".format(metadata.year),
        vertical_arrow_rate=False,
        vertical_arrow_gap=False,
    )

    aggregate_plot_roshni_presentation(
        COUNTRIES,
        show_method_list=[],
        metadata=metadata,
        ubi_on=True,
        save_as="exhibits/year={}/presentation/figure-1-build-1".format(metadata.year),
        vertical_arrow_rate=False,
        vertical_arrow_gap=False,
    )

    aggregate_plot_roshni_presentation(
        COUNTRIES,
        show_method_list=["oracle_gap"],
        metadata=metadata,
        ubi_on=True,
        save_as="exhibits/year={}/presentation/figure-1-build-2".format(metadata.year),
        vertical_arrow_rate=False,
        vertical_arrow_gap=False,
    )

    aggregate_plot_roshni_presentation(
        COUNTRIES,
        show_method_list=["pmt", "continuous_gap", "oracle_gap"],
        metadata=metadata,
        ubi_on=True,
        save_as="exhibits/year={}/presentation/figure-1-build-3".format(metadata.year),
        vertical_arrow_rate=True,
        vertical_arrow_gap=False,
    )

    aggregate_plot_roshni_presentation(
        COUNTRIES,
        show_method_list=["pmt", "oracle_gap"],
        metadata=metadata,
        ubi_on=True,
        save_as="exhibits/year={}/presentation/figure-1-build-4".format(metadata.year),
        vertical_arrow_rate=True,
        vertical_arrow_gap=False,
    )

    aggregate_plot_roshni_presentation(
        COUNTRIES,
        show_method_list=["binary_gap", "continuous_gap"],
        metadata=metadata,
        ubi_on=False,
        save_as="exhibits/year={}/presentation/figure-1-build-5".format(metadata.year),
        vertical_arrow_rate=True,
        vertical_arrow_gap=False,
    )

    aggregate_plot_roshni_presentation(
        COUNTRIES,
        show_method_list=["ubi", "binary_gap", "continuous_gap"],
        metadata=metadata,
        ubi_on=False,
        save_as="exhibits/year={}/presentation/figure-1-build-6".format(metadata.year),
        vertical_arrow_rate=True,
        vertical_arrow_gap=False,
    )

    aggregate_plot_roshni_presentation(
        COUNTRIES,
        show_method_list=["continuous_gap", "oracle_gap"],
        metadata=metadata,
        ubi_on=True,
        save_as="exhibits/year={}/presentation/figure-1-build-7".format(metadata.year),
        vertical_arrow_rate=True,
        vertical_arrow_gap=False,
    )

    aggregate_plot_presentation(
        COUNTRIES,
        show_method_list=["continuous_gap", "oracle_gap"],
        metadata=metadata,
        ubi_on=False,
        save_as="exhibits/year={}/presentation/figure-1-feasible_vs_oracle_all_series_0".format(
            metadata.year
        ),
        vertical_arrow_rate=False,
        vertical_arrow_gap=False,
    )
    aggregate_plot_presentation(
        COUNTRIES,
        show_method_list=["continuous_gap", "ubi"],
        metadata=metadata,
        ubi_on=True,
        save_as="exhibits/year={}/presentation/figure-2-feasible_vs_ubi_all_series_0".format(
            metadata.year
        ),
        vertical_arrow_rate=False,
        vertical_arrow_gap=False,
    )

    aggregate_plot_presentation(
        COUNTRIES,
        metadata=metadata,
        show_method_list=["continuous_gap", "binary_gap"],
        ubi_on=False,
        save_as="exhibits/year={}/presentation/figure-3-cont_gap_vs_binary_gap_all_series_0".format(
            metadata.year
        ),
        vertical_arrow_rate=False,
        vertical_arrow_gap=False,
    )

    aggregate_plot_presentation(
        COUNTRIES,
        show_method_list=["continuous_gap", "oracle_gap"],
        metadata=metadata,
        ubi_on=False,
        save_as="exhibits/year={}/presentation/figure-1-feasible_vs_oracle_all_series_1".format(
            metadata.year
        ),
        vertical_arrow_rate=True,
        vertical_arrow_gap=False,
    )
    aggregate_plot_presentation(
        COUNTRIES,
        show_method_list=["continuous_gap", "ubi"],
        metadata=metadata,
        ubi_on=True,
        save_as="exhibits/year={}/presentation/figure-2-feasible_vs_ubi_all_series_1".format(
            metadata.year
        ),
        vertical_arrow_rate=True,
        vertical_arrow_gap=False,
    )
    aggregate_plot_presentation(
        COUNTRIES,
        metadata=metadata,
        ubi_on=False,
        show_method_list=["continuous_gap", "binary_gap"],
        save_as="exhibits/year={}/presentation/figure-3-cont_gap_vs_binary_gap_all_series_1".format(
            metadata.year
        ),
        vertical_arrow_rate=True,
        vertical_arrow_gap=False,
    )

    aggregate_plot_presentation(
        COUNTRIES,
        show_method_list=["continuous_gap", "oracle_gap"],
        metadata=metadata,
        ubi_on=False,
        save_as="exhibits/year={}/presentation/figure-1-feasible_vs_oracle_all_series_2".format(
            metadata.year
        ),
        vertical_arrow_rate=True,
        vertical_arrow_gap=True,
    )
    aggregate_plot_presentation(
        COUNTRIES,
        show_method_list=["continuous_gap", "ubi"],
        metadata=metadata,
        ubi_on=True,
        save_as="exhibits/year={}/presentation/figure-2-feasible_vs_ubi_all_series_2".format(
            metadata.year
        ),
        vertical_arrow_rate=True,
        vertical_arrow_gap=True,
    )
    aggregate_plot_presentation(
        COUNTRIES,
        metadata=metadata,
        show_method_list=["continuous_gap", "binary_gap"],
        ubi_on=False,
        save_as="exhibits/year={}/presentation/figure-3-cont_gap_vs_binary_gap_all_series_2".format(
            metadata.year
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
@argh.arg(
    "--auxpath",
    help="Path to auxiliary data",
    type=str,
    default="data/auxiliary_data/auxiliary_data_11262025.csv",
)
@argh.arg(
    "--secondaryauxpath",
    help="Path to secondary auxiliary data",
    type=str,
    default="data/auxiliary_data/secondary_auxiliary_data_11262025.csv",
)
@argh.arg(
    "--wpcpath",
    help="Path to WPC data",
    type=str,
    default="data/auxiliary_data/wdl_pov_clock_oct_2024/wdl_pov_clock_oct_2024.csv",
)
def main(
    year=2021,
    nationalPovertyRate=1,
    globalPovertyRate=1,
    auxpath="",
    secondaryauxpath="",
    wpcpath="",
):
    os.makedirs(f"exhibits/year={year}/figs", exist_ok=True)
    os.makedirs(f"exhibits/year={year}/tables", exist_ok=True)
    os.makedirs(f"exhibits/year={year}/presentation", exist_ok=True)

    if year == 2017:
        povertyline = 2.15
    elif year == 2021:
        povertyline = 3.0

    metadata = Metadata(
        povertyline=povertyline,
        year=year,
        nationalPovertyRate=nationalPovertyRate,
        globalPovertyRate=globalPovertyRate,
        auxpath=auxpath,
        secondaryauxpath=secondaryauxpath,
        wpcpath=wpcpath,
    )

    get_malawi_rate_vs_gap_figure_1(metadata)
    get_headline_figure_2(metadata)
    get_rate_vs_gap_headline_figure_3(metadata)
    get_oracle_ratio_figure_4(metadata)
    get_ubi_ratio_figure_5(metadata)
    get_gdp_plot_figure_6(metadata)
    get_headline_figure_3_dollar(metadata)
    get_scatter_figure_7(metadata)
    get_table_extrapolation(metadata)
    # get_appendix_table_out_of_sample_rmse()
    get_appendix_table_survey(metadata)
    get_appendix_table_wpc(metadata)
    get_appendix_table_policy_cost_insample(metadata)
    get_country_level_analysis(metadata)
    make_macro_file(
        COUNTRIES,
        metadata=metadata,
        save_as="exhibits/empirical_macros",
    )
    make_presentation_figures(metadata)


if __name__ == "__main__":
    _parser = argh.ArghParser()
    _parser.add_commands([main])
    _parser.dispatch()
