import pandas as pd


class Metadata:
    def __init__(
        self,
        povertyline,
        year,
        nationalPovertyRate,
        globalPovertyRate,
        auxpath,
        secondaryauxpath,
        wpcpath,
    ):
        self.povertyline = povertyline
        self.year = year
        self.auxpath = auxpath
        self.secondaryauxpath = secondaryauxpath
        self.wpcpath = wpcpath
        self.nationalPovertyRate = nationalPovertyRate
        self.globalPovertyRate = globalPovertyRate
        self.geo_extrapolation = True

    def preprocess_wpc_data(self, countries=None):
        df = pd.read_csv(self.wpcpath)
        df.rename(
            columns={
                "hcr_pov": "wpc_poverty_rate",
                "pgi": "wpc_poverty_gap_index",
                "ccode": "country_code",
                "country": "country_name",
            },
            inplace=True,
        )
        country_df = self.preprocess_country_aux_data()
        df = df.merge(
            country_df[
                [
                    "country_code",
                    "total_population_2023",
                    "PPP_conversion_factor_2017",
                    "market_exchange_rate_2017",
                ]
            ],
            on="country_code",
            how="left",
        )

        # SHARE WORLD POOR ONLY VALID FOR 2023
        total_world_poor = (
            df[df["year"] == 2023]["wpc_poverty_rate"] * df["total_population_2023"]
        ).sum()
        df["wpc_share_world_poor"] = (
            df[df["year"] == 2023]["wpc_poverty_rate"] * df["total_population_2023"]
        ) / total_world_poor

        columns = [
            "country_name",
            "country_code",
            "year",
            "wpc_poverty_rate",
            "wpc_share_world_poor",
            "wpc_poverty_gap_index",
            "total_population_2023",
            "PPP_conversion_factor_2017",
            "market_exchange_rate_2017",
        ]
        df.sort_values(by="country_code", inplace=True)
        df = df[columns]
        df.dropna(subset=["country_code"], inplace=True)
        df = df[df["country_code"].isin(countries)] if countries is not None else df
        return df

    def preprocess_country_aux_data(self):
        df = pd.read_csv(self.auxpath)
        return df

    def preprocess_secondary_aux_data(self):
        df = pd.read_csv(self.secondaryauxpath)
        return df
