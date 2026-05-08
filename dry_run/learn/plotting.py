import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from learn.aggregation import (
    AggregatePovertyResults,
    CountryMethodPovertyResults,
    AveragedCountryMethodPovertyResults,
)
from learn.aux_data_prep import Metadata
from learn.formatting import METHODS
from extrapolation import get_national_poverty_rate_target, ExtrapolationResults
from learn.post_processing_utils import (
    get_data_dimension,
    get_country_name,
    make_string_country_list,
)
from adjustText import adjust_text
import bisect


def make_plot_for_country(
    country,
    method_list,
    metadata,
    save_as,
    ubi_on=True,
):
    """
    Make plot for a single country showing policy cost vs post-transfer poverty rate and gap index.

    :param country: 3-letter country code
    :param method_list: List of method names to include in the plot
    :param metadata: Metadata object containing result-specific information
    :param save_as: Filename to save the plot as (without extension)
    :param ubi_on: Boolean indicating whether to include UBI cost line in the plot
    """

    methods = METHODS.copy()
    results = []
    for i, method in enumerate(method_list):
        results.append(CountryMethodPovertyResults(country, method, metadata))

    fontsize = 30
    fig, ax = plt.subplots(1, 2, figsize=(24, 8))

    xlims = [results[0].initial_rate, results[0].initial_gap_index]
    xlabels = ["Post-Transfer Poverty Rate (%)", "Post-Transfer Poverty Gap Index (%)"]
    for i in range(2):
        ax[i].set_xlim(-xlims[i] * 0.05, xlims[i] * 1.05)
        ax[i].set_ylim(
            -0.01 * results[0].get_ubi_cost(), results[0].get_ubi_cost() * 1.05
        )
        ax[i].tick_params(axis="x", labelsize=fontsize * 0.75)
        ax[i].tick_params(axis="y", labelsize=fontsize * 0.75)
        ax[i].set_ylabel("Policy Cost ($ Billion Per Year)", fontsize=fontsize)
        ax[i].grid(True)
        ax[i].set_xlabel(xlabels[i], fontsize=fontsize)
        if ubi_on:
            ax[i].plot(
                np.linspace(0.0, xlims[i]),
                np.ones(50) * results[0].get_ubi_cost(),
                linestyle="--",
                color=METHODS["ubi_standard"]["color"],
                label=METHODS["ubi_standard"]["name"]
                + " (${})".format(metadata.povertyline),
                linewidth=3,
            )

    for i, method in enumerate(method_list):
        dic = methods[method]
        df = results[i]._load_data()

        rates = [results[0].initial_rate] + list(df["post_transfer_poverty_rate"] * 100)
        gaps = [results[0].initial_gap_index] + list(
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
    """
    Make aggregate plot for all countries showing policy cost vs post-transfer poverty rate and gap index.

    :param countries: List of 3-letter country codes
    :param method_list: List of methods to include in the plot
    :param metadata: Metadata object containing result-specific information
    :param save_as: Filename to save the plot as (without extension)
    :param ubi_on: Boolean indicating whether to include UBI cost line in the plot
    """
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
    initial_gap_index, initial_rate, _ = oracle_results.get_initial()
    xlims = [initial_rate, initial_gap_index]
    xlabels = [
        "Post-Transfer Poverty Rate\n (%)",
        "Post-Transfer Poverty Gap Index\n (%)",
    ]
    for i in range(2):
        ax[i].set_ylabel("Policy Cost ($ Billion Per Year)", fontsize=fontsize)
        ax[i].grid(True)
        ax[i].tick_params(axis="x", labelsize=fontsize * 0.75)
        ax[i].tick_params(axis="y", labelsize=fontsize * 0.75)
        ax[i].set_xlabel(xlabels[i], fontsize=fontsize)
        if ubi_on:
            ubi_cost = results[0].get_aggregate_ubi_cost()
            ax[i].plot(
                np.linspace(0.0, xlims[i]),
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

    ax[1].legend(fontsize=fontsize * 0.75)  # , #bbox_to_anchor=(1.05, 0.5)
    # fig.tight_layout(rect=[0, 0, 0.85, 1])
    plt.tight_layout()
    plt.savefig("{}.pdf".format(save_as), dpi=300, bbox_inches="tight")
    plt.close()


def get_policy_costs_as_percent_of_gdp(countries, metadata, global_rate=False):
    """
    Get policy costs as percent of GDP and government revenue for each country.

    :param countries: List of 3-letter country codes
    :param metadata: Metadata object containing result-specific information
    """
    # Get policy costs in 2023 nominal USD for each country to attain national poverty rate target
    if global_rate:
        rate = get_national_poverty_rate_target(metadata)
    else:
        rate = metadata.nationalPovertyRate
    results = [
        CountryMethodPovertyResults(country, "continuous_gap", metadata=metadata)
        for country in countries
    ]
    amts = []
    for i in range(len(countries)):
        amt = results[i].rate_to_cost_interpolator(rate).item()
        amts.append(amt)

    new_df = pd.DataFrame({"country_code": countries, "policy_cost": amts})

    # Compare policy costs as percent of GDP and govt revenue
    df = metadata.preprocess_country_aux_data()
    subdf = df[df["country_code"].isin(countries)][
        [
            "country_code",
            "survey_year",
            "GDP_survey_year",
            "government_revenue_percentage_survey_year",
            "ODA_most_recent",
        ]
    ]
    new_df = new_df.merge(subdf, on="country_code", how="left")
    new_df["govt_revenue_survey_year"] = (
        new_df["government_revenue_percentage_survey_year"]
        * new_df["GDP_survey_year"]
        / 100
    )
    new_df["policy_cost_as_percent_of_gdp"] = (
        new_df["policy_cost"] * 100 / new_df["GDP_survey_year"]
    )
    new_df["policy_cost_as_percent_of_govt_revenue"] = (
        new_df["policy_cost"] * 100 / new_df["govt_revenue_survey_year"]
    )
    new_df["oda_as_percent_of_gdp"] = (
        new_df["ODA_most_recent"] * 100 / new_df["GDP_survey_year"]
    )
    new_df["country_name"] = new_df["country_code"].apply(
        lambda x: get_country_name(x, metadata)
    )
    return new_df


def plot_bar_chart_policy_amt_as_percent_of_gdp(countries, metadata, save_as):
    """
    Make bar chart for policy amount as percent of GDP and government revenue for each country.

    :param countries: List of 3-letter country codes
    :param metadata: Metadata object containing result-specific information
    :param save_as: Filename to save the plot as (without extension)
    """

    new_df = get_policy_costs_as_percent_of_gdp(countries, metadata, global_rate=False)

    fig, axes = plt.subplots(2, 1, figsize=(30, 8 * 2))
    fontsize = 30
    # Bar plot for amts_as_percent_of_gdp

    ylabels = [
        "policy_cost_as_percent_of_gdp",
        "policy_cost_as_percent_of_govt_revenue",
    ]
    ylabel_names = ["% of GDP", "% of Gov't Revenue"]
    for i in range(2):
        new_df.sort_values(by=[ylabels[i]], inplace=True, ascending=False)
        axes[i].bar(new_df["country_name"], new_df[ylabels[i]], zorder=3)
        axes[i].set_xlabel("Country", fontsize=fontsize)
        axes[i].set_ylabel(ylabel_names[i], fontsize=fontsize)
        axes[i].set_xticklabels(new_df["country_name"], rotation=90, fontsize=fontsize)
        axes[i].set_yticklabels(
            np.round(axes[0].get_yticks()).astype(int), fontsize=fontsize
        )
        axes[i].grid(axis="y", zorder=0)

    plt.tight_layout()
    plt.savefig("{}.pdf".format(save_as), bbox_inches="tight")


def get_global_poverty_rate_target(metadata):
    """
    Method to attain global poverty rate that is attained when national poverty rate target is met.

    :param metadata: Metadata object containing result-specific information
    """
    df = metadata.preprocess_country_aux_data()
    national_poverty_rate_target = metadata.nationalPovertyRate / 100

    def global_poverty_rate(national_ceiling):
        national_poverty_rates = np.minimum(
            df["wb_poverty_rate_2023_povertyline_{}".format(metadata.year)],
            national_ceiling,
        ).to_numpy()
        global_poverty_rate = (
            national_poverty_rates * df["total_population_2023"]
        ).sum() / df["total_population_2023"].sum()
        return global_poverty_rate.item() * 100

    ceilings = np.linspace(
        0, df["wb_poverty_rate_2023_povertyline_{}".format(metadata.year)].max(), 100
    )
    global_poverty_rates = [global_poverty_rate(c) for c in ceilings]
    global_poverty_rate_target = interp1d(ceilings, global_poverty_rates)(
        national_poverty_rate_target
    )
    return global_poverty_rate_target


def get_rate_vs_gap_restricted_feature_set(countries, metadata, save_as):

    all_results = [
        AggregatePovertyResults(
            countries=countries, method="continuous_rate", metadata=metadata
        )
    ]

    metadata_new = Metadata(
        povertyline=metadata.povertyline,
        nationalPovertyRate=metadata.nationalPovertyRate,
        globalPovertyRate=metadata.globalPovertyRate,
        restricted_feature_set=True,
        year=metadata.year,
        auxpath=metadata.auxpath,
        secondaryauxpath=metadata.secondaryauxpath,
        wpcpath=metadata.wpcpath,
        refugeepath=metadata.refugeepath,
    )

    restricted_feature_set_results = AggregatePovertyResults(
        countries=countries, method="continuous_gap", metadata=metadata_new
    )
    all_results.append(restricted_feature_set_results)
    all_results.append(
        AggregatePovertyResults(
            countries=countries, method="continuous_gap", metadata=metadata
        )
    )
    method_list = ["continuous_rate", "continuous_gap", "continuous_gap"]
    fig, ax = plt.subplots(1, 2, figsize=(24, 8))
    append_to_method_name = [": d=20", ": d=20", ": d=all"]
    colors = ["orange", "royalblue", "blue"]

    for i, method in enumerate(method_list):
        color = colors[i]
        rate_domain = all_results[i].aggregate_interpolator_rate_domain
        rate_interpolator = all_results[i].aggregate_interpolator_rate_to_cost
        gap_domain = all_results[i].aggregate_interpolator_gap_domain
        gap_interpolator = all_results[i].aggregate_interpolator_gap_to_cost
        ax[0].plot(
            np.linspace(rate_domain[0], rate_domain[1], 200),
            rate_interpolator(np.linspace(rate_domain[0], rate_domain[1], 200)),
            label=METHODS[method]["name"] + append_to_method_name[i],
            color=color,
            linestyle=METHODS[method]["linestyle"],
            linewidth=3,
        )
        ax[1].plot(
            np.linspace(gap_domain[0], gap_domain[1], 200),
            gap_interpolator(np.linspace(gap_domain[0], gap_domain[1], 200)),
            label=METHODS[method]["name"] + append_to_method_name[i],
            color=color,
            linestyle=METHODS[method]["linestyle"],
            linewidth=3,
        )
    ax[0].legend(fontsize=30 * 0.75)  # , #bbox_to_anchor=(1.05, 0.5)
    ax[0].set_xlabel("Post-Transfer Poverty Rate (%)", fontsize=30)
    ax[1].set_xlabel("Post-Transfer Poverty Gap Index (%)", fontsize=30)
    ax[0].set_ylabel("Policy Cost ($ Billion Per Year)", fontsize=30)
    for i in range(2):
        ax[i].set_ylabel("Policy Cost ($ Billion Per Year)", fontsize=30)
        ax[i].grid(True)
        ax[i].tick_params(axis="x", labelsize=20)
        ax[i].tick_params(axis="y", labelsize=20)
    plt.savefig("{}.pdf".format(save_as), dpi=300, bbox_inches="tight")


def get_refugee_cost(metadata):

    refugee_df = metadata.preprocess_refugee_data()
    df = metadata.preprocess_country_aux_data()
    secondary_df = metadata.preprocess_secondary_aux_data()
    cols = [
        "PPP_conversion_factor_{}".format(metadata.year),
        "market_exchange_rate_{}".format(metadata.year),
        "country_code",
    ]
    df = df[cols]
    join_df = refugee_df.merge(df, on="country_code", how="left")

    inflation_adjustment = (
        1
        / secondary_df[
            secondary_df["indicator"]
            == "conversion_factor_nominal_USD_2023_to_{}".format(metadata.year)
        ]["value"]
        .values[0]
        .item()
    )

    total = 0.0
    dropped_countries = []
    for index, row in join_df.iterrows():
        PPP_conversion_factor = row["PPP_conversion_factor_{}".format(metadata.year)]
        market_exchange_rate = row["market_exchange_rate_{}".format(metadata.year)]

        conversion_factor = (
            (
                365  # days per year
                * inflation_adjustment  # from year to 2023 nominal
                * (
                    PPP_conversion_factor / market_exchange_rate
                )  # from PPP to nominal USD in year
            )
            * metadata.povertyline
            / 1000000000
        )  # to billion USD

        if np.isnan(conversion_factor):
            print(
                "Skipping country {} due to missing conversion factor".format(
                    row["country_code"]
                )
            )
            dropped_countries.append(row["country_code"])
            continue

        country_total = row["num_beneficiaries"] * conversion_factor
        total += country_total

    global_gdp = (
        secondary_df[
            secondary_df["indicator"] == "global_GDP_2023".format(metadata.year)
        ]["value"]
        .values[0]
        .item()
    )

    percentage_refugee_global_gdp = 100 * total / global_gdp

    return percentage_refugee_global_gdp, dropped_countries


def get_table_policy_cost_gdp(countries, metadata, save_as):
    """
    Get insample policy costs needed to attain a global poverty rate target.

    :param countries: List of 3-letter country codes
    :param metadata: Metadata object containing result-specific information
    :param save_as: Filename to save the plot as (without extension)
    """
    df = get_policy_costs_as_percent_of_gdp(countries, metadata, global_rate=False)
    df["survey_year"] = df["survey_year"].astype(int)
    df.rename(
        columns={
            "policy_cost": "Policy Cost",
            "GDP_survey_year": "GDP",
            "survey_year": "Reference Year",
            "govt_revenue_survey_year": "Gov't Revenue",
            "country_name": "Country",
        },
        inplace=True,
    )
    df["Policy Cost / GDP"] = df["policy_cost_as_percent_of_gdp"] / 100
    df["Policy Cost / Gov't Revenue"] = (
        df["policy_cost_as_percent_of_govt_revenue"] / 100
    )
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


def get_macros_relative_cost(extrapolation_results, metadata):
    """
    Get relative cost compared to OECD GDP, OECD + China GDP and OECD govt revenue, OECD + China govt revenue,
    global GDP.

    :param extrapolation_results: Dictionary containing insample policy cost, outofsample policy cost, oracle cost and extrapolated quadratic cost for each extrapolation method
    :param metadata: Description
    """
    df = metadata.preprocess_secondary_aux_data()

    oecd_gdp = df[df["indicator"] == "OECD_GDP_2023"]["value"].item()
    oecd_govt_revenue = (
        df[df["indicator"] == "OECD_GDP_2023"]["value"].item()
        * df[df["indicator"] == "OECD_govt_revenue_percentage_2023"]["value"].item()
        / 100
    )
    china_gdp = df[df["indicator"] == "china_GDP_2023"]["value"].item()
    china_govt_revenue = (
        df[df["indicator"] == "china_GDP_2023"]["value"].item()
        * df[df["indicator"] == "china_govt_revenue_percentage_2023"]["value"].item()
        / 100
    )
    global_gdp = df[df["indicator"] == "global_GDP_2023"]["value"].item()

    relative_cost_results = {"wb": {}, "wpc": {}}

    for key in extrapolation_results:
        policy_cost = extrapolation_results[key]["extrapolated_cost"]
        oracle_cost = extrapolation_results[key]["oracle_cost"]
        extrapolated_quadratic_cost = extrapolation_results[key][
            "quadratic_extrapolated_cost"
        ]
        percentage_oecd_gdp = 100 * policy_cost / oecd_gdp
        percentage_oecd_plus_china_gdp = 100 * policy_cost / (oecd_gdp + china_gdp)
        percentage_oecd_govt_revenue = 100 * policy_cost / (oecd_govt_revenue)

        percentage_oecd_plus_china_govt_revenue = (
            100 * policy_cost / (oecd_govt_revenue + china_govt_revenue)
        )

        percentage_feasible_global_gdp = 100 * policy_cost / global_gdp
        percentage_oracle_global_gdp = 100 * oracle_cost / global_gdp
        percentage_feasible_quadratic_global_gdp = (
            100 * extrapolated_quadratic_cost / global_gdp
        )

        relative_cost_results[key] = {
            "percentage_oecd_gdp": percentage_oecd_gdp,
            "percentage_oecd_plus_china_gdp": percentage_oecd_plus_china_gdp,
            "percentage_oecd_govt_revenue": percentage_oecd_govt_revenue,
            "percentage_oecd_plus_china_govt_revenue": percentage_oecd_plus_china_govt_revenue,
            "percentage_feasible_global_gdp": percentage_feasible_global_gdp,
            "percentage_feasible_quadratic_global_gdp": percentage_feasible_quadratic_global_gdp,
            "percentage_oracle_global_gdp": percentage_oracle_global_gdp,
        }

    return relative_cost_results


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
        "Ghana Living Standards Survey 7 2016-2017": "Living Standards Survey 7",
        "Integrated Household Living Conditions Survey (EICV7) 2023-2024": "EICV7",
        "Household Income and Expenditure Survey (HIES) 2022": "HIES",
        "Permanent Household Survey 2021-22": "Permanent Household Survey",
        "Household Budget Survey (HBS) 2014": "Household Budget Survey",
        "National Socio-Economic Survey (SUSENAS) 2018": "SUSENAS",
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

    df["survey_year"] = df["survey_year"].astype(int)
    df["survey_poverty_rate_povertyline_{}".format(metadata.year)] *= 100
    df["wb_poverty_rate_povertyline_{}_survey_year".format(metadata.year)] *= 100
    df["WB Rate PIP Using"] = (
        df["wb_poverty_rate_povertyline_{}_survey_year".format(metadata.year)]
        * df["pip_using"]
    )
    df["WB Rate PIP Not Using"] = df[
        "wb_poverty_rate_povertyline_{}_survey_year".format(metadata.year)
    ] * (1 - df["pip_using"])
    df.sort_values(by=["country_name"], inplace=True)
    columns = [
        "country_name",
        "survey_name",
        "survey_year",
        "sample_size",
        "covariate_dimension",
        "WB Rate PIP Using",
        "WB Rate PIP Not Using",
        "survey_poverty_rate_povertyline_{}".format(metadata.year),
    ]
    df = df[columns]
    df.rename(
        columns={
            "country_name": "Country",
            "sample_size": "$n$",
            "covariate_dimension": "$d$",
            "survey_poverty_rate_povertyline_{}".format(
                metadata.year
            ): "Survey Poverty Rate",
            "survey_name": "Survey Name",
            "survey_year": "Survey Year",
        },
        inplace=True,
    )
    df["WB Rate PIP Using"] = df["WB Rate PIP Using"].apply(
        lambda x: x if x != 0 else np.nan
    )
    df["WB Rate PIP Not Using"] = df["WB Rate PIP Not Using"].apply(
        lambda x: x if x != 0 else np.nan
    )

    if save_as:
        df.to_latex(
            save_as + ".tex",
            index=False,
            float_format="%.1f",
            escape=False,
        )
    if slides:
        df.drop(
            columns=[
                "WB Rate PIP Using",
                "WB Rate PIP Not Using",
                "Survey Poverty Rate",
            ],
            inplace=True,
        )
        df.to_latex(
            save_as + "_slides.tex",
            index=False,
            float_format="%.1f",
            escape=False,
        )
    return df


def get_table_share_world_poor(countries, metadata, save_as):
    df = metadata.preprocess_country_aux_data()
    df = df[
        [
            "country_code",
            "wb_poverty_rate_2023_povertyline_{}".format(metadata.year),
            "total_population_2023",
        ]
    ]
    total_population_poor = (
        df["wb_poverty_rate_2023_povertyline_{}".format(metadata.year)]
        * df["total_population_2023"]
    ).sum()
    df["wb_share_world_poor"] = (
        df["wb_poverty_rate_2023_povertyline_{}".format(metadata.year)]
        * df["total_population_2023"]
        / total_population_poor
    )
    df["wb_poverty_rate_2023_povertyline_{}".format(metadata.year)] *= 100
    df["wb_share_world_poor"] *= 100
    df = df[df["country_code"].isin(countries)]
    df.rename(
        columns={
            "country_code": "Country Code",
            "wb_poverty_rate_2023_povertyline_{}".format(metadata.year): "Poverty Rate",
            "wb_share_world_poor": "Share of World's Poor",
        },
        inplace=True,
    )
    df["Country"] = df["Country Code"].apply(lambda x: get_country_name(x, metadata))
    df = df.sort_values(by=["Country"])
    df[["Country", "Poverty Rate", "Share of World's Poor"]].to_latex(
        save_as + ".tex",
        index=False,
        float_format="%.1f",
        escape=False,
    )


def plot_bar_chart_ubi_ratio(countries, metadata, save_as):
    cont_gap_results = [
        CountryMethodPovertyResults(country, method="continuous_gap", metadata=metadata)
        for country in countries
    ]
    ubi_results = [
        CountryMethodPovertyResults(country, method="ubi", metadata=metadata)
        for country in countries
    ]

    res = []
    for i, country in enumerate(countries):
        ubi_cost = ubi_results[i].rate_to_cost_interpolator(
            metadata.nationalPovertyRate
        )
        targeting_cost = cont_gap_results[i].rate_to_cost_interpolator(
            metadata.nationalPovertyRate
        )
        res.append(
            {
                "country_code": country,
                "targeting_to_usi_perc": targeting_cost * 100 / ubi_cost,
                "initial_poverty_rate_perc": cont_gap_results[i].initial_rate,
            }
        )

    df = pd.DataFrame(res)
    fig, ax = plt.subplots(figsize=(12, 8))
    fontsize = 30
    pointsize = 70
    ax.scatter(
        x=df["initial_poverty_rate_perc"], y=df["targeting_to_usi_perc"], s=pointsize
    )
    offset = False
    texts = [
        ax.text(
            row["initial_poverty_rate_perc"],
            row["targeting_to_usi_perc"],
            row["country_code"],
            fontsize=fontsize / 2,
            ha="center",
            va="center",
        )
        for _, row in df.iterrows()
    ]
    ax.set_xlabel("Pre-Transfer Poverty Rate (%)", fontsize=fontsize)
    ax.set_ylabel("Cost Ratio: Targeting/USI (%)", fontsize=fontsize)
    ax.tick_params(axis="x", labelsize=fontsize * 0.75)
    ax.tick_params(axis="y", labelsize=fontsize * 0.75)
    (xlow, xhigh) = ax.get_xlim()
    ax.set_xlim(0, xhigh)
    ax.grid(True)
    plt.tight_layout()

    # Ghost points at the marker circumference (diagonal x) so adjust_text
    # avoids the full dot area rather than just the zero-size center coordinate.
    # fig.canvas.draw()
    radius_pts = np.sqrt(pointsize / np.pi) * 1.2
    disp_to_data = ax.transData.inverted()
    origin = disp_to_data.transform([0, 0])
    pix_per_pt = fig.dpi / 72
    radius_x = disp_to_data.transform([radius_pts * pix_per_pt, 0])[0] - origin[0]
    radius_y = disp_to_data.transform([0, radius_pts * pix_per_pt])[1] - origin[1]
    d = 1 / np.sqrt(2)
    xs = df["initial_poverty_rate_perc"].values
    ys = df["targeting_to_usi_perc"].values
    ghost_x = np.concatenate(
        [xs, xs + d * radius_x, xs - d * radius_x, xs + d * radius_x, xs - d * radius_x]
    )
    ghost_y = np.concatenate(
        [ys, ys + d * radius_y, ys - d * radius_y, ys - d * radius_y, ys + d * radius_y]
    )

    show_ghost_points = False

    if show_ghost_points:
        ax.scatter(ghost_x, ghost_y, s=10, color="red", zorder=10, linewidths=0)

    adjust_text(
        texts,
        ax=ax,
        force_text=2,
        x=ghost_x,
        y=ghost_y,
        avoid_self=True,
    )
    plt.savefig("{}.pdf".format(save_as), bbox_inches="tight")


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
    plt.ylabel("Cost Ratio: Targeting/Oracle", fontsize=fontsize)
    # plt.suptitle("Cost Ratio between UBI (Variable) and Gap Targeting (Continuous) vs. Country", fontsize=fontsize)
    plt.xticks(rotation=90, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.savefig("{}.pdf".format(save_as), bbox_inches="tight")


def get_macros_share_world_poor(countries, metadata):
    df = metadata.preprocess_country_aux_data()
    df = df[
        [
            "country_code",
            "wb_poverty_rate_2023_povertyline_{}".format(metadata.year),
            "total_population_2023",
        ]
    ]
    total_population_poor = (
        df["wb_poverty_rate_2023_povertyline_{}".format(metadata.year)]
        * df["total_population_2023"]
    ).sum()
    df["wb_share_world_poor"] = (
        df["wb_poverty_rate_2023_povertyline_{}".format(metadata.year)]
        * df["total_population_2023"]
        / total_population_poor
    )
    total_world_poor = round(
        df[df["country_code"].isin(countries)]["wb_share_world_poor"].sum() * 100, 2
    )

    togo_world_poor = round(
        (df[(df["country_code"] == "TGO")]["wb_share_world_poor"].values[0] * 100),
        2,
    )
    return total_world_poor, togo_world_poor


def get_macros_survey_info(countries, metadata):
    """
    Get basic statistics about the surveys in our sample

    :param countries: List of 3-letter country codes
    :param metadata: Metadata object containing result-specific information
    """
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


def get_headline_numbers(countries, metadata, global_rate=False):

    if global_rate:
        national_poverty_rate = get_national_poverty_rate_target(metadata)
    else:
        national_poverty_rate = metadata.nationalPovertyRate
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
                .aggregate_interpolator_rate_to_cost(national_poverty_rate)
                .item()
            )
        elif method == "oracle_gap":
            gap_induced = (
                agg_results[0]
                .aggregate_interpolator_rate_to_gap(national_poverty_rate)
                .item()
            )  # want oracle to correpond to global poverty gap

            cost[method] = (
                agg_results[i].aggregate_interpolator_gap_to_cost(gap_induced).item()
            )
        else:
            cost[method] = (
                agg_results[i]
                .aggregate_interpolator_rate_to_cost(national_poverty_rate)
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
        restricted_feature_set=False,
        refugeepath=metadata.refugeepath,
    )

    new_metadata = Metadata(
        povertyline=3.0,
        year=2021,
        auxpath=metadata.auxpath,
        wpcpath=metadata.wpcpath,
        secondaryauxpath=metadata.secondaryauxpath,
        nationalPovertyRate=metadata.nationalPovertyRate,
        globalPovertyRate=metadata.globalPovertyRate,
        restricted_feature_set=False,
        refugeepath=metadata.refugeepath,
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


def get_percentages(countries, metadata):
    df = get_policy_costs_as_percent_of_gdp(countries, metadata, global_rate=False)

    return (
        df["policy_cost_as_percent_of_gdp"].mean().item(),
        df["policy_cost_as_percent_of_govt_revenue"].mean().item(),
        df["oda_as_percent_of_gdp"].mean().item(),
    )


# def plot_country_policy(country, metadata, save_as):
# fontsize = 30
# x_clip = 15

# tgo_res = CountryMethodPovertyResults(country="TGO", method="continuous_gap", metadata=metadata)


# gap_data = tgo_transfers[tgo_transfers["method"] == "output_gt_continuous_gap"].copy()
# gap_data["post_transfer_consumption"] = gap_data["consumption"] + gap_data["ev_transfer"]

# color = METHODS["continuous_gap"]["color"]

# fig, ax = plt.subplots(figsize=(12, 8))
# sns.histplot(data=gap_data, x="consumption", bins=50, binrange=(0, x_clip),
#              color='grey', alpha=0.5, label="Pre-transfer", ax=ax)
# sns.histplot(data=gap_data, x="post_transfer_consumption", bins=50, binrange=(0, x_clip),
#              color=color, alpha=0.5, label="Post-transfer (Gap Minimization)", ax=ax)
# vline = ax.axvline(2.15, color="grey", linestyle="--", linewidth=3)

# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles=handles + [vline], labels=labels + ["Poverty line ($2.15)"],
#           fontsize=fontsize * 0.75)

# ax.set_xlabel("Consumption (Dollars/Day)", fontsize=fontsize)
# ax.set_ylabel("Count", fontsize=fontsize)
# ax.tick_params(axis="x", labelsize=fontsize * 0.75)
# ax.tick_params(axis="y", labelsize=fontsize * 0.75)
# ax.set_xlim(0, x_clip)
# ax.grid(True)
# plt.tight_layout()
# plt.savefig(paper_figures_out_path / "tgo_consumption_pre_post_transfer_continuous_gap_formatted.pdf", dpi=300, bbox_inches="tight")
# plt.show()


def get_welfare_comparison(countries, metadata, save_as):

    # take the amount it would cost to get a 1% poverty rate in each country
    # how much could welfare be maximized.
    res = AggregatePovertyResults(countries, method="continuous_gap", metadata=metadata)
    weights = res.country_weights["weight"]
    fontsize = 30
    initial_rates = [
        res.country_results[country].initial_rate for country in res.countries
    ]
    rates = np.linspace(1, max(initial_rates), 100)
    costs = np.zeros(rates.shape)
    gap_welfares = np.zeros(rates.shape)
    welfare_welfares = np.zeros(rates.shape)
    fig, ax = plt.subplots(figsize=(12, 8))
    for i, country in enumerate(countries):
        cont_gap = CountryMethodPovertyResults(
            country=country, method="continuous_gap", metadata=metadata
        )
        welfare_max = CountryMethodPovertyResults(
            country=country, method="welfare", metadata=metadata
        )
        new_rates = np.clip(
            rates,
            cont_gap.rate_to_cost_interpolator_domain[0],
            cont_gap.rate_to_cost_interpolator_domain[1],
        )
        gap_policy_costs = cont_gap.rate_to_cost_interpolator(new_rates)
        gap_welfares += (
            cont_gap.cost_to_welfare_interpolator(gap_policy_costs) * weights[country]
        )

        welfare_welfares += (
            welfare_max.cost_to_welfare_interpolator(gap_policy_costs)
            * weights[country]
        )
        costs += gap_policy_costs

    ax.plot(
        costs,
        gap_welfares,
        label=METHODS["continuous_gap"]["name"],
        color=METHODS["continuous_gap"]["color"],
        linestyle=METHODS["continuous_gap"]["linestyle"],
        linewidth=3,
    )
    ax.plot(
        costs,
        welfare_welfares,
        label=METHODS["welfare"]["name"],
        color=METHODS["welfare"]["color"],
        linestyle=METHODS["welfare"]["linestyle"],
        linewidth=3,
    )
    ax.set_xlabel("Policy Cost ($ Billion Per Year)", fontsize=fontsize)
    ax.set_ylabel("Post-Transfer Welfare\n (Log Nominal 2023 USD)", fontsize=fontsize)
    ax.grid(True)
    ax.legend(fontsize=fontsize * 0.75)
    ax.tick_params(axis="x", labelsize=fontsize * 0.75)
    ax.tick_params(axis="y", labelsize=fontsize * 0.75)
    plt.tight_layout()
    plt.savefig("{}.pdf".format(save_as), dpi=300, bbox_inches="tight")
    plt.close()


def plot_country_welfare(country, metadata, save_as):
    cont_gap = CountryMethodPovertyResults(
        country=country, method="continuous_gap", metadata=metadata
    )
    welfare = CountryMethodPovertyResults(
        country=country, method="welfare", metadata=metadata
    )

    fig, ax = plt.subplots(1, 2, figsize=(30, 8))
    fontsize = 30
    cont_gap_df = cont_gap._load_data()
    welfare_df = welfare._load_data()

    for method in ["continuous_gap", "welfare"]:
        dic = METHODS[method]
        df = cont_gap_df if method == "continuous_gap" else welfare_df
        ax[0].plot(
            df["policy_cost_per_capita"] * cont_gap.conversion_factor,
            df["post_transfer_welfare"] + np.log(welfare.nominal_conversion_factor),
            marker="o",
            label=dic["name"],
            color=dic["color"],
            linestyle=dic["linestyle"],
            linewidth=3,
        )

    ax[0].set_xlabel("Policy Cost ($ Billion Per Year)", fontsize=fontsize)
    ax[0].set_ylabel(
        "Post-Transfer Welfare \n (Log Nominal 2023 USD)", fontsize=fontsize
    )
    ax[0].grid(True)
    ax[0].tick_params(axis="x", labelsize=fontsize * 0.75)
    ax[0].tick_params(axis="y", labelsize=fontsize * 0.75)
    ax[0].legend(fontsize=fontsize * 0.75)

    budgets, welfare_transfers = welfare._load_transfer_data()
    budgets, gap_transfers = cont_gap._load_transfer_data()
    

    budget_idx = len(budgets) // 2  # use last budget level
    welfare_t = welfare_transfers[budget_idx]
    gap_t = gap_transfers[budget_idx]

    for i, (method, transfers_df) in enumerate(
        [("continuous_gap", gap_t), ("welfare", welfare_t)]
    ):
        
        ax[1].hist(
            transfers_df["ev_transfer"],
            weights=transfers_df["headcount_adjusted_hh_wgt"]
            / transfers_df["headcount_adjusted_hh_wgt"].sum(),
            bins=50,
            color=METHODS[method]["color"],
            alpha=0.6,
            edgecolor=METHODS[method]["color"],
            linewidth=0.8,
            label=METHODS[method]["name"],
            density=True,
        )

    ax[1].set_xlabel("Transfer Amount (Dollars/Day)", fontsize=fontsize)
    ax[1].set_ylabel("Density", fontsize=fontsize)
    ax[1].tick_params(axis="x", labelsize=fontsize * 0.75)
    ax[1].tick_params(axis="y", labelsize=fontsize * 0.75)
    ax[1].legend(fontsize=fontsize * 0.75)
    ax[1].grid(True)
    ax[1].set_title(
        f"Policy Cost: {budgets[budget_idx] * cont_gap.conversion_factor:.2f}B",
        fontsize=fontsize * 0.75,
    )

    plt.tight_layout()
    plt.savefig("{}.pdf".format(save_as), dpi=300, bbox_inches="tight")
    plt.close()

def plot_aggregate_welfare(countries, metadata, save_as):
    fig, ax = plt.subplots(1, 2, figsize=(24, 8))
    fontsize = 30

    results = []
    method_list = ["continuous_gap", "welfare"]

    for method in method_list:
        results.append(
            AggregatePovertyResults(
                countries=countries,
                method=method,
                metadata=metadata,
            )
        )

    xlabels = ["Post-Transfer Welfare", "Transfer Amounts"]

    ax[0].set_ylabel("Policy Cost ($ Billion Per Year)", fontsize=fontsize)

    for i in range(2):
        ax[i].grid(True)
        ax[i].tick_params(axis="x", labelsize=fontsize * 0.75)
        ax[i].tick_params(axis="y", labelsize=fontsize * 0.75)
        ax[i].set_xlabel(xlabels[i], fontsize=fontsize)

    for i, method in enumerate(method_list):
        color = METHODS[method]["color"]
        welfare_domain = results[i].aggregate_interpolator_welfare_domain
        welfare_interpolator = results[i].aggregate_interpolator_welfare_to_cost
        ax[0].plot(
            welfare_interpolator(
                np.linspace(welfare_domain[0], welfare_domain[1], 200)
            ),
            np.linspace(welfare_domain[0], welfare_domain[1], 200),
            label=METHODS[method]["name"],
            color=color,
            linestyle=METHODS[method]["linestyle"],
            linewidth=3,
        )

    ax[0].legend(fontsize=fontsize * 0.75)
    plt.tight_layout()
    plt.savefig("{}.pdf".format(save_as), dpi=300, bbox_inches="tight")
    plt.close()


def get_extrapolation_comparison(countries, metadata):
    extrapolation_wpc = ExtrapolationResults(
        countries,
        insample_data_source="wpc",
        outofsample_data_source="wpc",
        metadata=metadata,
        degree=1,
    )
    extrapolation_wpc.fit_regression_model()

    insample_wpc = extrapolation_wpc.get_in_sample_costs(
        survey_year=False, use_reg=True
    )
    insample_wpc.to_latex("exhibits/extrapolation_wpc_215_insample.tex", index=False)

    insample_policy_cost_wpc = insample_wpc["Policy Cost"].loc["Total"]
    outofsample_wpc = extrapolation_wpc.get_out_of_sample_costs()
    outofsample_wpc.to_latex(
        "exhibits/extrapolation_wpc_215_outofsample.tex", index=False
    )
    outofsample_policy_cost_wpc = outofsample_wpc["Policy Cost"].loc["Total"]

    total_policy_cost_wpc = insample_policy_cost_wpc + outofsample_policy_cost_wpc

    extrapolation_wb = ExtrapolationResults(
        countries,
        insample_data_source="wb",
        outofsample_data_source="wb",
        metadata=metadata,
        degree=1,
    )

    extrapolation_wb.fit_regression_model()

    insample_wb = extrapolation_wb.get_in_sample_costs(survey_year=False, use_reg=True)
    insample_wb.to_latex("exhibits/extrapolation_wb_215_insample.tex", index=False)
    insample_policy_cost_wb = insample_wb["Policy Cost"].loc["Total"]
    outofsample_policy_cost_wb = extrapolation_wb.get_out_of_sample_costs()
    outofsample_policy_cost_wb.to_latex(
        "exhibits/extrapolation_wb_215_outofsample.tex", index=False
    )
    outofsample_policy_cost_wb = outofsample_policy_cost_wb["Policy Cost"].loc["Total"]

    total_policy_cost_wb = insample_policy_cost_wb + outofsample_policy_cost_wb

    print(
        "WPC: Insample Policy Cost: {}, Out-of-sample Policy Cost: {}, Total Policy Cost: {}".format(
            insample_policy_cost_wpc, outofsample_policy_cost_wpc, total_policy_cost_wpc
        )
    )

    print(
        "WB: Insample Policy Cost: {}, Out-of-sample Policy Cost: {}, Total Policy Cost: {}".format(
            insample_policy_cost_wb, outofsample_policy_cost_wb, total_policy_cost_wb
        )
    )

    metadata_new = Metadata(
        povertyline=3,
        year=2021,
        nationalPovertyRate=metadata.nationalPovertyRate,
        globalPovertyRate=metadata.globalPovertyRate,
        auxpath=metadata.auxpath,
        secondaryauxpath=metadata.secondaryauxpath,
        wpcpath=metadata.wpcpath,
        restricted_feature_set=False,
        refugeepath=metadata.refugeepath,
    )

    extrapolation_wb_3 = ExtrapolationResults(
        countries,
        insample_data_source="wb",
        outofsample_data_source="wb",
        metadata=metadata_new,
        degree=1,
    )

    extrapolation_wb_3.fit_regression_model()

    insample_wb_3 = extrapolation_wb_3.get_in_sample_costs(
        survey_year=False, use_reg=True
    )
    insample_policy_cost_wb_3 = insample_wb_3["Policy Cost"].loc["Total"]
    insample_wb_3.to_latex("exhibits/extrapolation_wb_3_insample.tex", index=False)
    outofsample_wb_3 = extrapolation_wb_3.get_out_of_sample_costs()
    outofsample_wb_3.to_latex(
        "exhibits/extrapolation_wb_3_outofsample.tex", index=False
    )
    outofsample_policy_cost_wb_3 = outofsample_wb_3["Policy Cost"].loc["Total"]

    total_policy_cost_wb_3 = insample_policy_cost_wb_3 + outofsample_policy_cost_wb_3
    print(
        "WB $3 poverty: Insample Policy Cost: {}, Out-of-sample Policy Cost: {}, Total Policy Cost: {}".format(
            insample_policy_cost_wb_3,
            outofsample_policy_cost_wb_3,
            total_policy_cost_wb_3,
        )
    )


def get_extrapolation(countries, metadata):

    data_sources = ["wb", "wpc"]

    extrapolation_results = {}

    for data_source in data_sources:
        extrapolation = ExtrapolationResults(
            countries,
            insample_data_source=data_source,
            outofsample_data_source=data_source,
            metadata=metadata,
            degree=1,
        )
        extrapolation.fit_regression_model()
        regression_r2 = extrapolation.score
        insample_df = extrapolation.get_in_sample_costs(survey_year=False, use_reg=True)
        insample_policy_cost = insample_df["Policy Cost"].loc["Total"]
        outofsample_df = extrapolation.get_out_of_sample_costs()
        outofsample_policy_cost = outofsample_df["Policy Cost"].loc["Total"]
        oracle_cost, dropped_countries_gap = extrapolation.get_global_poverty_gap()
        dropped_countries = extrapolation.dropped_countries
        extrapolated_cost = insample_policy_cost + outofsample_policy_cost
        extrapolation_quadratic = ExtrapolationResults(
            countries,
            insample_data_source=data_source,
            outofsample_data_source=data_source,
            metadata=metadata,
            degree=2,
        )
        extrapolation_quadratic.fit_regression_model()
        insample_quadratic_df = extrapolation_quadratic.get_in_sample_costs(
            survey_year=False, use_reg=True
        )
        insample_quadratic_policy_cost = insample_quadratic_df["Policy Cost"].loc[
            "Total"
        ]
        outofsample_quadratic_df = extrapolation_quadratic.get_out_of_sample_costs()
        outofsample_quadratic_policy_cost = outofsample_quadratic_df["Policy Cost"].loc[
            "Total"
        ]
        quadratic_extrapolated_cost = (
            insample_quadratic_policy_cost + outofsample_quadratic_policy_cost
        )

        results = {
            "extrapolated_cost": extrapolated_cost,
            "oracle_cost": oracle_cost,
            "in_sample_policy_cost": insample_policy_cost,
            "out_of_sample_policy_cost": outofsample_policy_cost,
            "dropped_countries": dropped_countries,
            "dropped_countries_gap": dropped_countries_gap,
            "regression_r2": regression_r2,
            "quadratic_extrapolated_cost": quadratic_extrapolated_cost,
        }

        extrapolation_results[data_source] = results

    return extrapolation_results


def plot_scatter_poverty_countries(countries, metadata, save_as):

    extrapolation = ExtrapolationResults(
        countries,
        insample_data_source="wb",
        outofsample_data_source="wb",
        metadata=metadata,
    )
    extrapolation.plot_wb_figure(save_as=save_as)


def get_number_of_people_targeted(countries, metadata, save_as):

    method_results = [
        CountryMethodPovertyResults(country, method="continuous_gap", metadata=metadata)
        for country in countries
    ]

    number_targeted = []
    pops = []
    aux_df = metadata.preprocess_country_aux_data()

    for i, country in enumerate(countries):
        interpolator = method_results[i].get_number_of_people_targeted()
        number_targeted.append(interpolator(metadata.nationalPovertyRate).item())
        total_population = aux_df[aux_df["country_code"] == country][
            "total_population_survey_year"
        ].values[0]
        pops.append(total_population)

    df = pd.DataFrame(
        {
            "country_code": countries,
            "population_targeted": np.round(np.array(number_targeted), 0),
            "total_population_survey_year": pops,
        }
    )

    if save_as:
        df.to_csv(f"{save_as}.csv", index=False)

    return df


def make_sample_size_aggregate_plot_alternative(countries, metadata, save_as):
    fontsize = 30
    gap_res = []

    train_fracs = [0.05, 0.1, 0.2, 0.5, 0.7, 0.9, None]
    for train_frac in train_fracs:
        gap_results = AggregatePovertyResults(
            countries=countries,
            method="continuous_gap",
            metadata=metadata,
            train_frac=train_frac,
        )

        if train_frac is None:
            train_frac = 1.0

        gap_res.append(gap_results)

    train_fracs[-1] = 1.0

    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True, height_ratios=[1, 3])
    for i in range(2):
        axs[i].tick_params(axis="x", labelsize=fontsize * 0.75)
        axs[i].tick_params(axis="y", labelsize=fontsize * 0.75)
        axs[i].grid(True)

    fig.supylabel("% Increase in Policy Cost", fontsize=fontsize)
    axs[1].set_xlabel("% " + " of Training Set", fontsize=fontsize)

    axs[0].spines.bottom.set_visible(False)
    axs[1].spines.top.set_visible(False)
    axs[0].xaxis.tick_top()
    axs[0].tick_params(labeltop=False)  # don't put tick labels at the top
    axs[1].xaxis.tick_bottom()

    d = 0.5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(
        marker=[(-1, -d), (1, d)],
        markersize=12,
        linestyle="none",
        color="k",
        mec="k",
        mew=1,
        clip_on=False,
    )
    axs[0].plot([0, 1], [0, 0], transform=axs[0].transAxes, **kwargs)
    axs[1].plot([0, 1], [1, 1], transform=axs[1].transAxes, **kwargs)

    dic = METHODS["continuous_gap"]
    costs = [
        r.aggregate_interpolator_rate_to_cost(metadata.nationalPovertyRate).item()
        for r in gap_res
    ]

    costs = [gap_res[-1].get_aggregate_ubi_cost()] + costs
    train_fracs = [0.0] + train_fracs
    percent_inc_agg = ((np.array(costs) - min(costs)) / min(costs)) * 100

    agg_max = 100
    axs[1].set_ylim(-0.01 * agg_max, agg_max * 1.1)

    country_max = 0
    for country in countries:
        gap_res = []
        for train_frac in [0.05, 0.1, 0.2, 0.5, 0.7, 0.9, None]:
            gap_results = AveragedCountryMethodPovertyResults(
                country,
                method="continuous_gap",
                metadata=metadata,
                train_frac=train_frac,
            )

            if train_frac is None:
                train_frac = 1.0

            gap_res.append(gap_results)
        policy_costs = [gap_results.get_ubi_cost()] + [
            r.rate_to_cost_interpolator(metadata.nationalPovertyRate) for r in gap_res
        ]
        percent_inc = (
            (np.array(policy_costs) - min(policy_costs)) / min(policy_costs)
        ) * 100
        country_max = max(country_max, max(percent_inc))

        for i in range(2):
            axs[i].plot(
                (np.array(train_fracs) * 100),
                percent_inc,
                marker="o",
                color="gray",
                alpha=0.2,
                linestyle=dic["linestyle"],
                linewidth=3,
            )

    for i in range(2):
        axs[i].plot(
            (np.array(train_fracs) * 100),
            (percent_inc_agg),
            marker="o",
            label="Aggregate",
            color=dic["color"],
            linestyle=dic["linestyle"],
            linewidth=3,
        )

    axs[0].set_ylim(agg_max * 1.2, 2000)

    axs[0].legend(fontsize=fontsize * 0.75)
    plt.tight_layout()
    plt.savefig("{}.pdf".format(save_as), bbox_inches="tight")
    plt.close()


def plot_satellite_image(metadata, save_as):
    results = []
    results.append(CountryMethodPovertyResults("TGO", "ubi", metadata))
    results.append(CountryMethodPovertyResults("TGO", "continuous_gap", metadata))
    results.append(
        CountryMethodPovertyResults("TGO_alpha_earth", "continuous_gap", metadata)
    )
    results.append(
        CountryMethodPovertyResults(
            "TGO_alpha_earth_and_survey", "continuous_gap", metadata
        )
    )
    colors = ["purple", "blue", "cyan", "cornflowerblue"]
    labels = [" (Survey)", " (Survey)", " (Satellite)", " (Survey + Satellite)"]

    fontsize = 30
    fig, ax = plt.subplots(1, 2, figsize=(24, 8))

    xlims = [results[0].initial_rate, results[0].initial_gap_index]
    xlabels = ["Post-Transfer Poverty Rate (%)", "Post-Transfer Poverty Gap Index (%)"]
    for i in range(2):
        ax[i].set_xlim(-xlims[i] * 0.05, xlims[i] * 1.05)
        ax[i].set_ylim(
            -0.01 * results[0].get_ubi_cost(), results[0].get_ubi_cost() * 1.05
        )
        ax[i].tick_params(axis="x", labelsize=fontsize * 0.75)
        ax[i].tick_params(axis="y", labelsize=fontsize * 0.75)
        ax[i].set_ylabel("Policy Cost ($ Billion Per Year)", fontsize=fontsize)
        ax[i].grid(True)
        ax[i].set_xlabel(xlabels[i], fontsize=fontsize)
        ax[i].plot(
            np.linspace(0.0, xlims[i]),
            np.ones(50) * results[0].get_ubi_cost(),
            linestyle="--",
            color=METHODS["ubi_standard"]["color"],
            label=METHODS["ubi_standard"]["name"]
            + " (${})".format(metadata.povertyline),
            linewidth=3,
        )

    for i in range(len(results)):
        method = results[i].method
        country = results[i].country
        dic = METHODS[method]
        df = results[i]._load_data()

        rates = [results[0].initial_rate] + list(df["post_transfer_poverty_rate"] * 100)
        gaps = [results[0].initial_gap_index] + list(
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
                label=dic["name"] + labels[i],
                color=colors[i],
                linestyle=dic["linestyle"],
                linewidth=3,
            )

        ax[1].plot(
            gaps,
            costs,
            marker="o",
            label=dic["name"] + labels[i],
            color=colors[i],
            linestyle=dic["linestyle"],
            linewidth=3,
        )

    ax[1].legend(fontsize=fontsize * 0.75)
    plt.tight_layout()
    plt.savefig("{}.pdf".format(save_as), bbox_inches="tight")
    plt.close()


def make_sample_size_aggregate_plot(countries, metadata, save_as):
    fontsize = 30
    gap_res = []

    train_fracs = [0.05, 0.1, 0.2, 0.5, 0.7, 0.9, None]
    for train_frac in train_fracs:
        gap_results = AggregatePovertyResults(
            countries=countries,
            method="continuous_gap",
            metadata=metadata,
            train_frac=train_frac,
        )

        if train_frac is None:
            train_frac = 1.0

        gap_res.append(gap_results)

    train_fracs[-1] = 1.0

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))  # sharex=True, height_ratios=[1, 3])
    ax.tick_params(axis="x", labelsize=fontsize * 0.75)
    ax.tick_params(axis="y", labelsize=fontsize * 0.75)
    ax.grid(True)

    fig.supylabel("% Increase in Policy Cost", fontsize=fontsize)
    ax.set_xlabel("% " + " of Training Set", fontsize=fontsize)

    dic = METHODS["continuous_gap"]
    costs = [
        r.aggregate_interpolator_rate_to_cost(metadata.nationalPovertyRate).item()
        for r in gap_res
    ]

    costs = [gap_res[-1].get_aggregate_ubi_cost()] + costs
    train_fracs = [0.0] + train_fracs
    percent_inc_agg = ((np.array(costs) - min(costs)) / min(costs)) * 100

    country_max = 0
    for country in countries:
        gap_res = []
        for train_frac in [0.05, 0.1, 0.2, 0.5, 0.7, 0.9, None]:
            gap_results = AveragedCountryMethodPovertyResults(
                country,
                method="continuous_gap",
                metadata=metadata,
                train_frac=train_frac,
            )

            if train_frac is None:
                train_frac = 1.0

            gap_res.append(gap_results)
        policy_costs = [gap_results.get_ubi_cost()] + [
            r.rate_to_cost_interpolator(metadata.nationalPovertyRate) for r in gap_res
        ]
        percent_inc = (
            (np.array(policy_costs) - min(policy_costs)) / min(policy_costs)
        ) * 100
        country_max = max(country_max, max(percent_inc))

        ax.plot(
            (np.array(train_fracs) * 100),
            percent_inc,
            marker="o",
            color="gray",
            alpha=0.2,
            linestyle=dic["linestyle"],
            linewidth=3,
        )

    ax.legend(fontsize=fontsize * 0.75)

    ax.plot(
        (np.array(train_fracs) * 100),
        (percent_inc_agg),
        marker="o",
        label="Aggregate",
        color=dic["color"],
        linestyle=dic["linestyle"],
        linewidth=3,
    )
    ax.set_ylim(-0.01 * max(percent_inc_agg), 100)
    plt.tight_layout()
    plt.savefig("{}.pdf".format(save_as), bbox_inches="tight")
    plt.close()


def plot_targeting_efficiency(country, metadata, save_as):
    res = CountryMethodPovertyResults(
        country=country, method="continuous_gap", metadata=metadata
    )
    budgets, transfers = res._load_transfer_data()
    data = res._load_data()
    fontsize = 30

    efficiencies = []
    for i in range(len(budgets)):
        transfer_data = transfers[i]
        weights = pd.read_parquet("data/{}/test.parquet".format(country))[
            "headcount_adjusted_hh_wgt"
        ]
        transfer_data["headcount_adjusted_hh_wgt"] = weights

        gap = (metadata.povertyline - transfer_data["consumption"]).clip(lower=0)
        transfer = transfer_data["ev_transfer"]
        gap_closing = np.minimum(transfer, gap)
        fraction_excess_transfers = (
            (weights * transfer).sum() - (weights * gap_closing).sum()
        ) / (weights * transfer).sum()

        efficiencies.append(
            {
                "excess_transfers_perc": fraction_excess_transfers * 100,
                "post_transfer_poverty_rate_perc": (
                    data["post_transfer_poverty_rate"].iloc[i]
                    / data["initial_poverty_rate"].iloc[i]
                )
                * 100,
            }
        )

    efficiencies.append(
        {"excess_transfers_perc": 0.0, "post_transfer_poverty_rate_perc": 100.0}
    )

    efficiency_df = pd.DataFrame(efficiencies).sort_values(
        ["post_transfer_poverty_rate_perc"]
    )

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(
        efficiency_df["post_transfer_poverty_rate_perc"],
        efficiency_df["excess_transfers_perc"],
        color=METHODS["continuous_gap"]["color"],
        linestyle=METHODS["continuous_gap"]["linestyle"],
        linewidth=3,
        marker="o",
        clip_on=False,
        zorder=5,
    )
    ax.set_xlabel(
        "Share Still Poor \n (% of Pre-Transfer Poverty Poor)", fontsize=fontsize
    )
    ax.set_ylabel("Share Excess Transfer\n (% of Total Transfer)", fontsize=fontsize)

    ax.tick_params(axis="x", labelsize=fontsize * 0.75)
    ax.tick_params(axis="y", labelsize=fontsize * 0.75)
    ax.grid(True)
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    plt.tight_layout()
    plt.savefig("{}.pdf".format(save_as), dpi=300, bbox_inches="tight")
    plt.close()


def make_transfer_plot(country, metadata, save_as):

    fig, ax = plt.subplots(1, 2, figsize=(24, 8))
    fontsize = 30

    def get_transfer_data(country, method, metadata):
        results = CountryMethodPovertyResults(
            country=country, method=method, metadata=metadata
        )
        budgets, transfers = results._load_transfer_data()
        cost = (
            results.rate_to_cost_interpolator(metadata.nationalPovertyRate)
            / results.conversion_factor
        )

        idx = bisect.bisect_left(budgets, cost)
        if method == "oracle_gap":
            return transfers[-1]
        else:
            if np.abs(budgets[idx] - cost) > np.abs(budgets[idx - 1] - cost):
                return transfers[idx - 1]
            else:
                return transfers[idx]

    gap_transfers = get_transfer_data(country, "continuous_gap", metadata)
    pmt_transfers = get_transfer_data(country, "binary_gap", metadata)
    oracle_transfers = get_transfer_data(country, "oracle_gap", metadata)

    usi_res = CountryMethodPovertyResults(
        country=country, method="ubi", metadata=metadata
    )

    usi_amt = (
        usi_res.rate_to_cost_interpolator(metadata.nationalPovertyRate).item()
        / usi_res.conversion_factor
    )

    methods = ["binary_gap", "continuous_gap", "oracle_gap"]
    transfers = [pmt_transfers, gap_transfers, oracle_transfers]

    all_vals = np.concatenate([t["ev_transfer"] for t in transfers])
    _, bin_edges = np.histogram(all_vals, bins=30)
    bin_width = bin_edges[1] - bin_edges[0]
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bar_width = bin_width / len(methods)
    offsets = np.linspace(
        -bin_width / 2 + bar_width / 2, bin_width / 2 - bar_width / 2, len(methods)
    )

    for i, method in enumerate(methods):
        vals = transfers[i]["ev_transfer"]
        weights = transfers[i]["headcount_adjusted_hh_wgt"]
        total_weight = weights.sum()
        pct, _ = np.histogram(
            vals, bins=bin_edges, weights=weights / total_weight * 100
        )
        if method == "binary_gap":
            factor = 2
        else:
            factor = 1
        ax[0].bar(
            centers + offsets[i],
            pct,
            width=bar_width,
            color=METHODS[method]["color"],
            alpha=0.6 / factor,
            edgecolor=METHODS[method]["color"],
            linewidth=0.8,
            label=METHODS[method]["name"],
        )

    ax[0].axvline(
        usi_amt,
        label=METHODS["ubi"]["name"],
        color=METHODS["ubi"]["color"],
        linestyle=METHODS["ubi"]["linestyle"],
    )

    ax[0].legend(fontsize=fontsize * 0.75, loc="upper left", bbox_to_anchor=(0.08, 1))
    ax[0].set_xlabel("Transfer Amount (Dollars/Day)", fontsize=fontsize)
    ax[0].set_ylabel("Share of Population (%)", fontsize=fontsize)
    ax[0].tick_params(axis="x", labelsize=fontsize * 0.75)
    ax[0].tick_params(axis="y", labelsize=fontsize * 0.75)

    x_clip = 15

    pre_transfer_consumption = gap_transfers["consumption"]
    post_transfer_consumption = (
        gap_transfers["consumption"] + gap_transfers["ev_transfer"]
    )

    ax[1].hist(
        pre_transfer_consumption,
        bins=np.linspace(0, x_clip, 50),
        label="Pre-transfer",
        color="grey",
        alpha=0.5,
    )
    ax[1].hist(
        post_transfer_consumption,
        bins=np.linspace(0, x_clip, 50),
        label="Post-transfer (Gap Minimization)",
        color=METHODS["continuous_gap"]["color"],
        alpha=0.5,
    )
    vline = ax[1].axvline(2.15, color="black", linestyle="--", linewidth=3)

    handles, labels = ax[1].get_legend_handles_labels()
    ax[1].legend(
        handles=handles + [vline],
        labels=labels + ["Poverty line ($2.15)"],
        fontsize=fontsize * 0.75,
    )

    ax[1].set_xlabel("Consumption (Dollars/Day)", fontsize=fontsize)
    ax[1].set_ylabel("Count", fontsize=fontsize)
    ax[1].tick_params(axis="x", labelsize=fontsize * 0.75)
    ax[1].tick_params(axis="y", labelsize=fontsize * 0.75)
    ax[1].set_xlim(0, x_clip)
    ax[1].grid(True)
    plt.tight_layout()
    plt.savefig("{}.pdf".format(save_as), dpi=300, bbox_inches="tight")
    plt.close()


def make_sample_size_plot(country, metadata, save_as):
    fontsize = 30
    n, d = get_data_dimension(country)
    n_train = n * 0.6
    n_samples = []
    gap_res = []

    for train_frac in [0.05, 0.1, 0.2, 0.5, 0.7, 0.9, None]:
        if train_frac is not None:
            gap_results = AveragedCountryMethodPovertyResults(
                country,
                method="continuous_gap",
                metadata=metadata,
                train_frac=train_frac,
            )
        else:
            train_frac = 1.0
            gap_results = CountryMethodPovertyResults(
                country, method="continuous_gap", metadata=metadata
            )

        n_samples.append(n_train * train_frac)
        gap_res.append(gap_results)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.tick_params(axis="x", labelsize=fontsize * 0.75)
    ax.tick_params(axis="y", labelsize=fontsize * 0.75)
    ax.set_ylabel("% Increase in Policy Cost", fontsize=fontsize)
    ax.grid(True)
    ax.set_xlabel("Number of Training Samples", fontsize=fontsize)
    dic = METHODS["continuous_gap"]

    n_samples = [0] + n_samples
    policy_costs = np.array(
        [gap_results.get_ubi_cost()]
        + [r.rate_to_cost_interpolator(metadata.nationalPovertyRate) for r in gap_res]
    )
    std_dev = np.array(
        [0]
        + [
            r.std_dev_rate_to_cost_interpolator(metadata.nationalPovertyRate)
            for r in gap_res[:-1]
        ]
        + [0]
    )
    upper_policy_costs = policy_costs + std_dev
    lower_policy_costs = policy_costs - std_dev
    percent_inc = (np.array(policy_costs) - min(policy_costs)) / min(policy_costs) * 100
    upper_percent_inc = (
        (np.array(upper_policy_costs) - min(policy_costs)) / min(policy_costs) * 100
    )
    lower_percent_inc = (
        (np.array(lower_policy_costs) - min(policy_costs)) / min(policy_costs) * 100
    )
    ax.plot(
        n_samples,
        percent_inc,
        marker="o",
        label=dic["name"],
        color=dic["color"],
        linestyle=dic["linestyle"],
        linewidth=3,
    )
    ax.fill_between(
        n_samples, lower_percent_inc, upper_percent_inc, alpha=0.1, color=dic["color"]
    )
    # ubi_dic = METHODS["ubi"]
    # ax.plot(n_samples, [ubi_results.rate_to_cost_interpolator(1)] * len(n_samples),
    #         linestyle=ubi_dic["linestyle"],
    #         linewidth=3,
    #         color=ubi_dic["color"],
    #         label=ubi_dic["name"])

    ax.legend(fontsize=fontsize * 0.75)
    plt.tight_layout()
    plt.savefig("{}.pdf".format(save_as), bbox_inches="tight")
    plt.close()


def make_macro_file(countries, metadata, save_as):
    countries = sorted(countries)
    all_countries_string = make_string_country_list(countries, metadata=metadata)

    # SHARE WORLD'S POOR METRICS
    total_world_poor, togo_world_poor = get_macros_share_world_poor(
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
    global_cost, _ = get_headline_numbers(countries, metadata, global_rate=True)

    # GET HEADLINE NUMBERS FOR NATIONAL POVERTY RATE TARGET
    national_cost, _ = get_headline_numbers(countries, metadata, global_rate=False)
    global_poverty_rate_for_national = get_global_poverty_rate_target(metadata)

    # GET PERCENTAGES
    avg_percent_gdp, avg_percent_govt_revenue, avg_percent_oda = get_percentages(
        countries,
        metadata=metadata,
    )

    # GET REFUGEE COST
    refugee_cost_percentage, refugee_dropped_countries = get_refugee_cost(metadata)
    refugee_dropped_countries_string = make_string_country_list(
        refugee_dropped_countries, metadata=metadata
    )

    # GET TOGO HEADLINE NUMBERS
    togo_cost, agg_results = get_headline_numbers(["TGO"], metadata=metadata)
    conversion_factor_togo = (
        agg_results[0].country_results["TGO"]._get_conversion_factor()
    )

    togo_variable_amt = togo_cost["ubi_variable"] / conversion_factor_togo

    # GET ORACLE RATIOS
    min_ratio, max_ratio = get_macros_oracle_ratios(countries, metadata=metadata)

    # GET EXTRAPOLATION
    extrapolation_results = get_extrapolation(countries, metadata=metadata)
    relative_cost_results = get_macros_relative_cost(
        extrapolation_results=extrapolation_results,
        metadata=metadata,
    )

    # GET POVERTY LINE COMPARISON
    (
        headlineGapNewPovertyLineNationalTarget,
        relativeContGapNewOldPovertyLineNationalTarget,
    ) = get_macro_povertyline_comparison(countries, metadata)

    dropped_countries_string = make_string_country_list(
        extrapolation_results["wb"]["dropped_countries"], metadata=metadata
    )
    dropped_countries_gap_string = make_string_country_list(
        extrapolation_results["wb"]["dropped_countries_gap"], metadata=metadata
    )

    togo_n, togo_d = get_data_dimension("TGO")
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
            "\\newcommand{\\sampleCostPercentGDP}" + f"{{{round(avg_percent_gdp)}}}\n"
        )
        f.write(
            "\\newcommand{\\sampleCostPercentGovtRevenue}"
            + f"{{{round(avg_percent_govt_revenue)}}}\n"
        )
        f.write(
            "\\newcommand{\\sampleOdaPercentGDP}" + f"{{{round(avg_percent_oda)}}}\n"
        )
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
            "\\newcommand{\\extrapolationWBCost}"
            + "{{{}}}\n".format(round(extrapolation_results["wb"]["extrapolated_cost"]))
        )
        f.write(
            "\\newcommand{\\extrapolationWPCCost}"
            + "{{{}}}\n".format(
                round(extrapolation_results["wpc"]["extrapolated_cost"])
            )
        )
        f.write(
            "\\newcommand{\\extrapolationWBRSquared}"
            + "{{{}}}\n".format(round(extrapolation_results["wb"]["regression_r2"], 2))
        )
        f.write(
            "\\newcommand{\\extrapolationWPCRSquared}"
            + "{{{}}}\n".format(round(extrapolation_results["wpc"]["regression_r2"], 2))
        )
        f.write(
            "\\newcommand{\\extrapolationWBOECDGDPPercent}"
            + "{{{}}}\n".format(
                round(relative_cost_results["wb"]["percentage_oecd_gdp"], 2)
            )
        )
        f.write(
            "\\newcommand{\\extrapolationWPCOECDGDPPercent}"
            + "{{{}}}\n".format(
                round(relative_cost_results["wpc"]["percentage_oecd_gdp"], 2)
            )
        )
        f.write(
            "\\newcommand{\\extrapolationWBOECDGovtRevPercent}"
            + "{{{}}}\n".format(
                round(relative_cost_results["wb"]["percentage_oecd_govt_revenue"], 2)
            )
        )
        f.write(
            "\\newcommand{\\extrapolationWBOECDPlusChinaGDPPercent}"
            + "{{{}}}\n".format(
                round(relative_cost_results["wb"]["percentage_oecd_plus_china_gdp"], 2)
            )
        )
        f.write(
            "\\newcommand{\\extrapolationWBOECDPlusChinaGovtRevPercent}"
            + "{{{}}}\n".format(
                round(
                    relative_cost_results["wb"][
                        "percentage_oecd_plus_china_govt_revenue"
                    ],
                    2,
                )
            )
        )
        f.write(
            "\\newcommand{\\extrapolationWBDroppedCountries}"
            + "{{{}}}\n".format(dropped_countries_string)
        )
        f.write(
            "\\newcommand{\\extrapolationWBDroppedCountriesGap}"
            + "{{{}}}\n".format(dropped_countries_gap_string)
        )
        f.write(
            "\\newcommand{\\extrapolationWBOutOfSampleCost}"
            + "{{{}}}\n".format(
                round(extrapolation_results["wb"]["out_of_sample_policy_cost"])
            )
        )
        f.write(
            "\\newcommand{\\extrapolationWBGlobalGDP}"
            + "{{{}}}\n".format(
                round(relative_cost_results["wb"]["percentage_feasible_global_gdp"], 2)
            )
        )
        f.write(
            "\\newcommand{\\extrapolationWPCGlobalGDP}"
            + "{{{}}}\n".format(
                round(relative_cost_results["wpc"]["percentage_feasible_global_gdp"], 2)
            )
        )
        f.write(
            "\\newcommand{\\extrapolationWBQuadraticGlobalGDP}"
            + "{{{}}}\n".format(
                round(
                    relative_cost_results["wb"][
                        "percentage_feasible_quadratic_global_gdp"
                    ],
                    2,
                )
            )
        )
        f.write(
            "\\newcommand{\\oracleWBGlobalGDP}"
            + "{{{}}}\n".format(
                round(relative_cost_results["wb"]["percentage_oracle_global_gdp"], 2)
            )
        )
        f.write(
            "\\newcommand{\\togoShareWorldsPoor}" + "{{{}}}\n".format(togo_world_poor)
        )
        f.write(
            "\\newcommand{\\togoUBIVariableAmount}"
            + "{{{}}}\n".format(round(togo_variable_amt, 2))
        )
        f.write(
            "\\newcommand{\\togoGapOracleRatio}"
            + "{{{}}}\n".format(
                round(togo_cost["continuous_gap"] / togo_cost["oracle_gap"])
            )
        )
        f.write(
            "\\newcommand{\\togoGapUBIPercent}"
            + "{{{}}}\n".format(
                round((togo_cost["continuous_gap"] * 100 / togo_cost["ubi"]))
            )
        )
        f.write(
            "\\newcommand{\\togoGapUBIVariablePercent}"
            + "{{{}}}\n".format(
                round((togo_cost["continuous_gap"] * 100 / togo_cost["ubi_variable"])),
                1,
            )
        )
        f.write(
            "\\newcommand{\\togoGapPMTPercent}"
            + "{{{}}}\n".format(
                round(togo_cost["continuous_gap"] * 100 / togo_cost["pmt"], 0)
            )
        )
        f.write(
            "\\newcommand{\\togoBinaryContPercentIncrease}"
            + "{{{}}}\n".format(
                round(
                    (togo_cost["binary_gap"] - togo_cost["continuous_gap"])
                    * 100
                    / togo_cost["continuous_gap"],
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
        f.write("\\newcommand{\\togoCovariateDimension}" + "{{{}}}\n".format(togo_d))
        f.write("\\newcommand{\\togoSampleSize}" + "{{{}}}\n".format(togo_n))
        f.write("\\newcommand{\\minDimension}" + "{{{}}}\n".format(min_d))
        f.write("\\newcommand{\\maxDimension}" + "{{{}}}\n".format(max_d))
        f.write(
            "\\newcommand{\\refugeeGlobalGDP}"
            + "{{{}}}\n".format(round(refugee_cost_percentage, 2))
        )
        f.write(
            "\\newcommand{\\refugeePlusExtrapolationGlobalGDP}"
            + "{{{}}}\n".format(
                round(
                    refugee_cost_percentage
                    + relative_cost_results["wb"]["percentage_feasible_global_gdp"],
                    2,
                )
            )
        )
        f.write(
            "\\newcommand{\\refugeeDroppedCountries}"
            + "{{{}}}\n".format(refugee_dropped_countries_string)
        )
