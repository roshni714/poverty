import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.interpolate import interp1d
import numpy as np
from learn.formatting import METHODS
from learn.aggregation import (
    CountryMethodPovertyResults,
)
import matplotlib.pyplot as plt


def get_national_poverty_rate_target(metadata):
    df = metadata.preprocess_wpc_data()
    df = df[df["year"] == 2023]

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
    national_poverty_rate_target = interp1d(global_poverty_rates, ceilings)(
        metadata.globalPovertyRate
    )
    return national_poverty_rate_target * 100


class ExtrapolationResults:
    def __init__(
        self,
        countries,
        metadata,
        insample_data_source="survey",
        outofsample_data_source="wb",
    ):
        self.in_sample_countries = countries
        self.povertyline = metadata.povertyline
        self.year = metadata.year
        self.globalPovertyRate = metadata.globalPovertyRate
        self.metadata = metadata
        main_national_target = get_national_poverty_rate_target(metadata)
        self.nationalPovertyRate = main_national_target
        self.insample_data_source = insample_data_source
        self.outofsample_data_source = outofsample_data_source

    def get_true_feasible_oracle_costs(self):
        cont_gap_results = [
            CountryMethodPovertyResults(
                country, "continuous_gap", metadata=self.metadata
            )
            for country in self.in_sample_countries
        ]

        oracle_results = [
            CountryMethodPovertyResults(country, "oracle_gap", metadata=self.metadata)
            for country in self.in_sample_countries
        ]

        in_sample_country_ratios = []
        in_sample_costs = []
        oracle_costs = []

        for i, _ in enumerate(self.in_sample_countries):
            oracle_cost = cont_gap_results[i].get_poverty_gap()
            in_sample_cost = cont_gap_results[i].rate_to_cost_interpolator(
                self.nationalPovertyRate
            )
            in_sample_country_ratios.append(
                in_sample_cost / oracle_cost
            )  # ratio to achieve 1% poverty reduction
            actual_oracle_cost = oracle_results[i].get_poverty_gap()
            in_sample_costs.append(in_sample_cost)
            oracle_costs.append(
                actual_oracle_cost
            )  # to achieve full poverty elimination
        survey_df = pd.DataFrame(
            {
                "Country": self.in_sample_countries,
                "Policy Cost": np.array(in_sample_costs),
                "Oracle Cost": np.array(oracle_costs),
            }
        )
        self.survey_in_sample_df = survey_df
        return in_sample_country_ratios

    def fit_regression_model(self):
        X = []
        y = []

        cont_gap_results = [
            CountryMethodPovertyResults(
                country, "continuous_gap", metadata=self.metadata
            )
            for country in self.in_sample_countries
        ]
        aux_data = self.metadata.preprocess_country_aux_data()
        for i, country in enumerate(self.in_sample_countries):
            # X.append(wpc_df[
            #    (wpc_df["country_code"] == country) & (wpc_df["year"] == aux_data[aux_data["country_code"] == country]["survey_year"].values[0])
            # ]["wpc_poverty_rate"].item())
            pov_rate = aux_data[aux_data["country_code"] == country][
                "survey_poverty_rate_povertyline_{}".format(self.year)
            ].item()
            X.append(pov_rate)
            print(country, pov_rate)
        y = self.get_true_feasible_oracle_costs()
        X = np.array(X).reshape(len(X), 1)
        y = np.array(y).reshape(len(X), 1)
        model = LinearRegression(fit_intercept=True)
        model.fit(X, y)
        self.model = model
        self.score = model.score(X, y)
        return X, y, model

    def plot_figure(self, save_as):
        X, y, model = self.fit_regression_model()
        fig, ax = plt.subplots(1, 2, figsize=(20, 8))
        fontsize = 20
        ax[0].scatter(X * 100, y, color="orange", alpha=0.5, s=100)
        ax[0].plot(X * 100, model.predict(X), "-", color="black")
        ax[0].set_xlabel("Survey Poverty Rate (%)", fontsize=fontsize)
        ax[0].set_ylabel("Feasible/Oracle Ratio", fontsize=fontsize)
        wpc_data = self.metadata.preprocess_wpc_data()
        wpc_data = wpc_data[wpc_data["year"] == 2023]
        nationalPovertyRate = get_national_poverty_rate_target(self.metadata) / 100
        in_sample_wpc = wpc_data[
            wpc_data["country_code"].isin(self.in_sample_countries)
        ]
        out_of_sample_wpc = wpc_data[
            ~wpc_data["country_code"].isin(self.in_sample_countries)
        ]
        out_of_sample_wpc = out_of_sample_wpc[
            out_of_sample_wpc["wpc_poverty_rate"] > nationalPovertyRate
        ]
        ax[1].scatter(
            out_of_sample_wpc["wpc_poverty_rate"] * 100,
            out_of_sample_wpc["wpc_poverty_gap_index"] * 100,
            label="Out of Sample Countries",
            s=100,
            alpha=0.5,
        )
        ax[1].scatter(
            in_sample_wpc["wpc_poverty_rate"] * 100,
            in_sample_wpc["wpc_poverty_gap_index"] * 100,
            label="In Sample Countries",
            s=100,
            alpha=0.5,
        )
        ax[1].set_xlabel("WPC (2023) Poverty Rate (%)", fontsize=fontsize)
        ax[1].set_ylabel("WPC (2023) Poverty Gap Index (%)", fontsize=fontsize)
        for i in range(2):
            ax[i].set_xticklabels(ax[i].get_xticks(), fontsize=fontsize * 0.75)
            ax[i].set_yticklabels(ax[i].get_yticks(), fontsize=fontsize * 0.75)
            ax[i].legend(fontsize=fontsize)
        plt.savefig("{}.pdf".format(save_as), bbox_inches="tight")
        plt.close()

    def get_conversion_factor(self, country):
        df = self.metadata.preprocess_country_aux_data()
        ppp_exchange_rate = (
            df[df["country_code"] == country][
                "PPP_conversion_factor_{}".format(self.year)
            ]
            .values[0]
            .item()
        )
        market_exchange_rate = (
            df[df["country_code"] == country][
                "market_exchange_rate_{}".format(self.year)
            ]
            .values[0]
            .item()
        )
        population = (
            df[df["country_code"] == country]["total_population_2023"].values[0].item()
        )

        if self.year == 2021:
            inflation_adjustment = 1.14
        elif self.year == 2017:
            inflation_adjustment = 1.23

        return (
            self.povertyline
            * inflation_adjustment
            * population
            * 365
            * (ppp_exchange_rate / market_exchange_rate)
            / (10**9)
        )

    def get_in_sample_costs(self, survey_year, use_reg):
        if self.insample_data_source == "survey" and use_reg == False:
            df = self.survey_in_sample_df.copy(deep=True)
            df.loc["Total"] = self.survey_in_sample_df.sum(numeric_only=True)
            return df

        elif self.insample_data_source == "survey" and use_reg == True:
            X_test = []
            df = self.metadata.preprocess_country_aux_data()
            for i, country in enumerate(self.in_sample_countries):
                X_test.append(
                    [
                        df[df["country_code"] == country][
                            "survey_poverty_rate_povertyline_{}".format(self.year)
                        ].item()
                    ]
                )
            oracle_costs = self.survey_in_sample_df["Oracle Cost"].tolist()
        elif self.insample_data_source == "wb" and survey_year == False:
            X_test = []
            oracle_costs = []
            df = self.metadata.preprocess_country_aux_data()
            for i, country in enumerate(self.in_sample_countries):
                X_test.append(
                    [
                        df[df["country_code"] == country][
                            "wb_poverty_rate_povertyline_{}_most_recent".format(
                                self.year
                            )
                        ].item()
                    ]
                )
                gap_index = df[df["country_code"] == country][
                    "wb_poverty_gap_index_povertyline_{}_most_recent".format(self.year)
                ].item()
                conversion_factor = self.get_conversion_factor(country)
                oracle_costs.append(gap_index * conversion_factor)
        elif self.insample_data_source == "wpc" and survey_year == True:
            X_test = []
            oracle_costs = []
            df = self.metadata.preprocess_wpc_data()
            country_df = self.metadata.preprocess_country_aux_data()
            for i, country in enumerate(self.in_sample_countries):
                country_year = country_df[country_df["country_code"] == country][
                    "survey_year"
                ].values[0]
                X_test.append(
                    [
                        df[
                            (df["country_code"] == country)
                            & (df["year"] == country_year)
                        ]["wpc_poverty_rate"].item()
                    ]
                )
                gap_index = df[
                    (df["country_code"] == country) & (df["year"] == country_year)
                ]["wpc_poverty_gap_index"].item()
                conversion_factor = self.get_conversion_factor(country)
                oracle_costs.append(gap_index * conversion_factor)
        elif self.insample_data_source == "wpc" and survey_year == False:
            X_test = []
            oracle_costs = []
            df = self.metadata.preprocess_wpc_data()
            country_df = self.metadata.preprocess_country_aux_data()
            for i, country in enumerate(self.in_sample_countries):
                X_test.append(
                    [
                        df[(df["country_code"] == country) & (df["year"] == 2023)][
                            "wpc_poverty_rate"
                        ].item()
                    ]
                )
                gap_index = df[(df["country_code"] == country) & (df["year"] == 2023)][
                    "wpc_poverty_gap_index"
                ].item()
                conversion_factor = self.get_conversion_factor(country)
                oracle_costs.append(gap_index * conversion_factor)
        else:
            raise ValueError("not a valid data_source time combination")

        pred_ratios = np.maximum(self.model.predict(np.array(X_test)), 1)
        pred = pred_ratios * np.array(oracle_costs).reshape(len(X_test), 1)

        in_sample_cost_df = pd.DataFrame(
            {
                "Country": self.in_sample_countries,
                "Policy Cost": list(pred.flatten()),
                "Oracle Cost": list(oracle_costs),
            }
        )

        in_sample_cost_df.loc["Total"] = in_sample_cost_df.sum(numeric_only=True)

        return in_sample_cost_df

    def get_out_of_sample_countries(self):
        dropped_countries = []

        if self.outofsample_data_source == "wb":
            poverty_rate_key = "wb_poverty_rate_povertyline_{}_most_recent".format(
                self.year
            )
            poverty_gap_key = "wb_poverty_gap_index_povertyline_{}_most_recent".format(
                self.year
            )
            df = self.metadata.preprocess_country_aux_data()

        elif self.outofsample_data_source == "wpc":
            df = self.metadata.preprocess_wpc_data()
            poverty_rate_key = "wpc_poverty_rate"
            poverty_gap_key = "wpc_poverty_gap_index"
            df = df[df["year"] == 2023]

        poor_countries = []
        all_countries = df["country_code"].unique().tolist()
        excluded = []
        for country in all_countries:
            if df[df["country_code"] == country][poverty_rate_key].item() > 0.01:
                poor_countries.append(country)
            else:
                excluded.append(country)

        for country in poor_countries:
            if np.isnan(df[df["country_code"] == country][poverty_rate_key].item()):
                print(country, "poverty rate missing")
                dropped_countries.append(country)
            elif np.isnan(df[df["country_code"] == country][poverty_gap_key].item()):
                print(country, "poverty gap missing")
                dropped_countries.append(country)
            elif np.isnan(
                df[df["country_code"] == country]["total_population_2023"].item()
            ):
                print(country, "population missing")
                dropped_countries.append(country)
            elif np.isnan(
                df[df["country_code"] == country][
                    "PPP_conversion_factor_{}".format(self.year)
                ].item()
            ):
                print(country, "PPP missing")
                dropped_countries.append(country)
            elif np.isnan(
                df[df["country_code"] == country][
                    "market_exchange_rate_{}".format(self.year)
                ].item()
            ):
                print(country, "market exchange rate missing")
                dropped_countries.append(country)

        out_of_sample_countries = []
        for country in poor_countries:
            if country not in self.in_sample_countries + dropped_countries:
                out_of_sample_countries.append(country)
        self.dropped_countries = dropped_countries
        return out_of_sample_countries

    def get_out_of_sample_costs(self):
        X_test = []
        oracle_costs = []
        df = self.metadata.preprocess_country_aux_data()
        out_of_sample_countries = self.get_out_of_sample_countries()

        if self.outofsample_data_source == "wb":
            for i, country in enumerate(out_of_sample_countries):
                X_test.append(
                    [
                        df[df["country_code"] == country][
                            "wb_poverty_rate_povertyline_{}_most_recent".format(
                                self.year
                            )
                        ].item()
                    ]
                )
                gap_index = df[df["country_code"] == country][
                    "wb_poverty_gap_index_povertyline_{}_most_recent".format(self.year)
                ].item()
                conversion_factor = self.get_conversion_factor(country)
                oracle_costs.append(gap_index * conversion_factor)

        elif self.outofsample_data_source == "wpc":
            wpc_df = self.metadata.preprocess_wpc_data()
            for i, country in enumerate(out_of_sample_countries):
                X_test.append(
                    [
                        wpc_df[
                            (wpc_df["country_code"] == country)
                            & (wpc_df["year"] == 2023)
                        ]["wpc_poverty_rate"].item()
                    ]
                )
                gap_index = wpc_df[
                    (wpc_df["country_code"] == country) & (wpc_df["year"] == 2023)
                ]["wpc_poverty_gap_index"].item()
                conversion_factor = self.get_conversion_factor(country)
                oracle_costs.append(gap_index * conversion_factor)

        pred_ratios = np.maximum(self.model.predict(np.array(X_test)), 1)
        pred = pred_ratios * np.array(oracle_costs).reshape(len(X_test), 1)
        out_of_sample_cost_df = pd.DataFrame(
            {
                "Country": out_of_sample_countries,
                "Policy Cost": pred.flatten(),
                "Oracle Cost": oracle_costs,
            }
        )
        out_of_sample_cost_df.loc["Total"] = out_of_sample_cost_df.sum(
            numeric_only=True
        )

        return out_of_sample_cost_df


# COUNTRIES = [
#     "benin",
#     "burkina_faso",
#     "colombia",
#     "cote_divoire",
#     "ethiopia",
#     "ghana",
#     "guinea_bissau",
#     "kenya",
#     "india",
#     "malawi",
#     "mali",
#     "niger",
#     "nigeria",
#     "senegal",
#     "south_africa",
#     "south_sudan",
#     "tanzania",
#     "togo",
#     "uganda",
# ]

# def do_insample_extrapolation(insample_data_source="survey",
#                      use_reg=False,
#                      survey_year=True):

#     extrapolation = ExtrapolationResults(COUNTRIES,
#                                          insample_data_source=insample_data_source,
#                                          outofsample_data_source="wb",
#                                          povertyline=2.15, year=2017, globalPovertyRate=1)
#     extrapolation.fit_regression_model()
#     insample_df = extrapolation.get_in_sample_costs(survey_year=survey_year, use_reg=use_reg)
#     insample_policy_cost = insample_df["Policy Cost"].loc["Total"]
#     insample_df["Country"] = insample_df["Country"].astype(str)
#     insample_df["Country"] = insample_df["Country"].apply(get_country_name)
#     insample_df.to_latex("extrapolation_tables/insample_costs_{}_{}_{}.tex".format(insample_data_source, "no_reg" if use_reg==False else "reg", "survey_year" if survey_year==True else "current"), index=False, float_format="%.1f")

#     print("========================================")
#     print("in sample data:", insample_data_source, "use regression:", use_reg, "survey year:", survey_year)
#     print("In-sample Oracle Cost",  insample_df["Oracle Cost"].loc["Total"])
#     print("In-sample Policy Cost (in billion USD): ", insample_policy_cost)

#     return insample_df

# def do_outofsample_extrapolation(outofsample_data_source="wb"):

#     extrapolation = ExtrapolationResults(COUNTRIES,
#                                          insample_data_source="survey",
#                                          outofsample_data_source=outofsample_data_source,
#                                          povertyline=2.15, year=2017, globalPovertyRate=1)
#     extrapolation.fit_regression_model()
#     outofsample_df = extrapolation.get_out_of_sample_costs()
#     outofsample_policy_cost = outofsample_df["Policy Cost"].loc["Total"]
#     outofsample_df["Country"] = outofsample_df["Country"].astype(str)
#     outofsample_df["Country"] = outofsample_df["Country"].apply(get_country_name)
#     print("========================================")
#     print("Out-of-sample Oracle Cost",  outofsample_df["Oracle Cost"].loc["Total"])
#     print("Out-of-sample Policy Cost (in billion USD): ", outofsample_policy_cost)
#     outofsample_df.to_latex("extrapolation_tables/outofsample_costs_{}.tex".format(outofsample_data_source), index=False, float_format="%.1f")
#     return outofsample_df

# insample_survey1 = do_insample_extrapolation(insample_data_source="survey", use_reg=False)
# insample_survey2 = do_insample_extrapolation(insample_data_source="survey", use_reg=True)
# insample_wpc = do_insample_extrapolation(insample_data_source="wpc", use_reg=True, survey_year=True)
# insample_wpc_new= do_insample_extrapolation(insample_data_source="wpc", use_reg=True, survey_year=False)
# insample_wb_new = do_insample_extrapolation(insample_data_source="wb", use_reg=True, survey_year=False)

# def get_survey_year_comparison_feasible_insample(insample_survey1, insample_survey2, insample_wpc):
#     insample_survey1 = insample_survey1.rename(columns={"Policy Cost": "Survey (No Reg) (1)"})
#     insample_survey2 = insample_survey2.rename(columns={"Policy Cost": "Survey (2)"})
#     insample_wpc = insample_wpc.rename(columns={"Policy Cost": "WPC (3)"})
#     new_df = insample_survey1[["Country", "Survey (No Reg) (1)"]].copy(deep=True)
#     new_df = new_df.merge(insample_survey2[["Country", "Survey (2)"]], on="Country", how="left")
#     new_df = new_df.merge(insample_wpc[["Country", "WPC (3)"]], on="Country", how="left")
#     new_df.to_latex("extrapolation_tables/insample_feasible_costs_surveyyear_comparison.tex", index=False, float_format="%.2f")

# get_survey_year_comparison_feasible_insample(insample_survey1, insample_survey2, insample_wpc)


# def get_current_year_comparison_feasible_insample(insample_wpc_new, insample_wb_new):
#     insample_wpc_new = insample_wpc_new.rename(columns={"Policy Cost": "WPC (1)"})
#     insample_wb_new = insample_wb_new.rename(columns={"Policy Cost": "WB (2)"})
#     new_df = insample_wpc_new[["Country", "WPC (1)"]].copy(deep=True)
#     new_df = new_df.merge(insample_wb_new[["Country", "WB (2)"]], on="Country", how="left")
#     new_df.to_latex("extrapolation_tables/insample_feasible_costs_currentyear_comparison.tex", index=False, float_format="%.2f")

# get_current_year_comparison_feasible_insample(insample_wpc_new, insample_wb_new)


# def get_survey_year_comparison_oracle_insample(insample_survey1, insample_survey2, insample_wpc):
#     insample_survey1 = insample_survey1.rename(columns={"Oracle Cost": "Survey (No Reg) (1)"})
#     insample_survey2 = insample_survey2.rename(columns={"Oracle Cost": "Survey (2)"})
#     insample_wpc = insample_wpc.rename(columns={"Oracle Cost": "WPC (3)"})
#     new_df = insample_survey1[["Country", "Survey (No Reg) (1)"]].copy(deep=True)
#     new_df = new_df.merge(insample_survey2[["Country", "Survey (2)"]], on="Country", how="left")
#     new_df = new_df.merge(insample_wpc[["Country", "WPC (3)"]], on="Country", how="left")
#     new_df.to_latex("extrapolation_tables/insample_oracle_costs_surveyyear_comparison.tex", index=False, float_format="%.2f")

# get_survey_year_comparison_oracle_insample(insample_survey1, insample_survey2, insample_wpc)


# def get_current_year_comparison_oracle_insample(insample_wpc_new, insample_wb_new):
#     insample_wpc_new = insample_wpc_new.rename(columns={"Oracle Cost": "WPC (1)"})
#     insample_wb_new = insample_wb_new.rename(columns={"Oracle Cost": "WB (2)"})
#     new_df = insample_wpc_new[["Country", "WPC (1)"]].copy(deep=True)
#     new_df = new_df.merge(insample_wb_new[["Country", "WB (2)"]], on="Country", how="left")
#     new_df.to_latex("extrapolation_tables/insample_oracle_costs_currentyear_comparison.tex", index=False, float_format="%.2f")

# get_current_year_comparison_oracle_insample(insample_wpc_new, insample_wb_new)

# outofsample_wb = do_outofsample_extrapolation(outofsample_data_source="wb")
# outofsample_wpc = do_outofsample_extrapolation(outofsample_data_source="wpc")

# def get_current_year_comparison_feasible_outofsample(outofsample_wb, outofsample_wpc):
#     outofsample_wb = outofsample_wb.rename(columns={"Policy Cost": "WB (1)"})
#     outofsample_wpc = outofsample_wpc.rename(columns={"Policy Cost": "WPC (2)"})
#     new_df = outofsample_wb[["Country", "WB (1)"]].copy(deep=True)
#     new_df = new_df.merge(outofsample_wpc[["Country", "WPC (2)"]], on="Country", how="outer")
#     new_df.sort_values(by="WB (1)", ascending=False, inplace=True)
#     new_df.to_latex("extrapolation_tables/outofsample_feasible_costs_currentyear_comparison.tex", index=False, float_format="%.2f")

# get_current_year_comparison_feasible_outofsample(outofsample_wb, outofsample_wpc)

# def get_current_year_comparison_oracle_outofsample(outofsample_wb, outofsample_wpc):
#     outofsample_wb = outofsample_wb.rename(columns={"Oracle Cost": "WB (1)"})
#     outofsample_wpc = outofsample_wpc.rename(columns={"Oracle Cost": "WPC (2)"})
#     new_df = outofsample_wb[["Country", "WB (1)"]].copy(deep=True)
#     new_df = new_df.merge(outofsample_wpc[["Country", "WPC (2)"]], on="Country", how="outer")
#     new_df.sort_values(by="WB (1)", ascending=False, inplace=True)
#     new_df.to_latex("extrapolation_tables/outofsample_oracle_costs_currentyear_comparison.tex", index=False, float_format="%.2f")

# get_current_year_comparison_oracle_outofsample(outofsample_wb, outofsample_wpc)
