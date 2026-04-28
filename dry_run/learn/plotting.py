import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from learn.aggregation import (
    AggregatePovertyResults,
    CountryMethodPovertyResults,
)
from learn.aux_data_prep import Metadata
from learn.formatting import METHODS
from extrapolation import get_national_poverty_rate_target, ExtrapolationResults
from learn.post_processing_utils import (
    get_data_dimension,
    get_country_name,
    make_string_country_list,
)


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
        ax[i].set_ylim(-0.1, results[0].get_ubi_cost() * 1.05)
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
    initial_gap_index, initial_rate = (
        oracle_results.get_initial_aggregate_gap_index_and_rate()
    )
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
                "ratio_of_usi_and_targeting": ubi_cost / targeting_cost,
            }
        )

    df = pd.DataFrame(res)
    df.to_csv(f"{save_as}.csv", index=False)
    print(df)
    df.sort_values(by=["ratio_of_usi_and_targeting"], ascending=False, inplace=True)
    fontsize = 30
    plt.figure(figsize=(30, 8))

    plt.bar(
        [get_country_name(c, metadata) for c in df["country_code"]],
        df["ratio_of_usi_and_targeting"],
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
        [res[i]["ratio_of_usi_and_targeting"] for i in range(len(countries))]
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


def get_welfare_comparison(countries, metadata):

    # take the amount it would cost to get a 1% poverty rate in each country
    # how much could welfare be maximized.
    res = AggregatePovertyResults(countries, method="continuous_gap", metadata=metadata)
    weights = res.country_weights
    weighted_avg_ratio = 0.0
    for country in countries:
        cont_gap = CountryMethodPovertyResults(
            country=country, method="continuous_gap", metadata=metadata
        )
        welfare_max = CountryMethodPovertyResults(
            country=country, method="welfare", metadata=metadata
        )
        initial_welfare = cont_gap.initial_welfare
        gap_policy_cost = cont_gap.rate_to_cost_interpolator(1)
        gap_welfare = cont_gap.cost_to_welfare_interpolator(gap_policy_cost)
        welfare_welfare = welfare_max.cost_to_welfare_interpolator(gap_policy_cost)
        ratio = (gap_welfare - initial_welfare) / (welfare_welfare - initial_welfare)
        weighted_avg_ratio += ratio * weights[country]
    return weighted_avg_ratio


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


def make_sample_size_plot(country, metadata, save_as):
    fontsize = 30
    n, d = get_data_dimension(country)
    n_train = n * 0.6
    n_samples = []
    res = []
    for train_frac in [0.1, 0.2, 0.5, 0.7, 1.0]:
        results = CountryMethodPovertyResults(
            country, method="continuous_gap", metadata=metadata, train_frac=train_frac
        )
        if train_frac is None:
            train_frac = 1.0

        n_samples.append(n_train * train_frac)
        res.append(results)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    xlims = [0.0, n_train]
    ax.set_xlim(xlims[0], xlims[1])
    ax.set_ylim(-0.1, res[-1].get_ubi_cost() * 1.05)
    ax.tick_params(axis="x", labelsize=fontsize * 0.75)
    ax.tick_params(axis="y", labelsize=fontsize * 0.75)
    ax.set_ylabel("Policy Cost ($ Billion Per Year)", fontsize=fontsize)
    ax.grid(True)
    ax.set_xlabel("Number of Training Samples", fontsize=fontsize)
    dic = METHODS["continuous_gap"]
    ax.plot(
        n_samples,
        [r.rate_to_cost_interpolator(1) for r in res],
        marker="o",
        label=dic["name"],
        color=dic["color"],
        linestyle=dic["linestyle"],
        linewidth=3,
    )

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
