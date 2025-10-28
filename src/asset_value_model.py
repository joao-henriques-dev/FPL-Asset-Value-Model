from src.clean_sheet_points_model import CleanSheetPointsModel
from src.fdr_value_model import FDRValueModel
import numpy as np
import pandas as pd
from numpy.typing import NDArray

class AssetValueModel:
    """
    A comprehensive model that uses submodels, deterministic and predictive,
    for calculating asset values based on several metrics that make up three
    key parts of the FPL asset value calculation:
        - player's expected points, which combinates both individual and collective statistics;
        - fixture difficulty rating (FDR) value of the player's team;
        - player's price.

    Attributes:
        goalkeepers (pandas.DataFrame): A dataframe containing goalkeeper data
        defenders (pandas.DataFrame): A dataframe containing defender data
        midfielders (pandas.DataFrame): A dataframe containing midfielder data
        forwards (pandas.DataFrame): A dataframe containing forward data
        fdr_value_model (FDRValueModel): Instance of the FDRValueModel class
            initialised with fixture data
        clean_sheet_points_model_gk_def (CleanSheetPointsModel): Instance of the
            CleanSheetPointsModel class initialised with data for goalkeepers and defenders
            (xGp90 and defensive points p/match)
        clean_sheet_points_model_mid (CleanSheetPointsModel): Instance of the
            CleanSheetPointsModel class initialised with data for midfielders
            (xGp90 and clean sheets p/match)
    """

    def __init__(self, watchlist: tuple, fixtures_df: pd.DataFrame,
                 xgcp90_data: NDArray[np.float64], defensive_points_data: NDArray[np.float64],
                 clean_sheets_per_match: NDArray[np.float64]):
        """
        Initialises the AssetValueModel object with the given data.

        Args:
            watchlist (tuple): A tuple containing four dataframes, with each representing
                a different dataframe for each position
            fixtures_df (pandas.DataFrame): Input dataframe containing
                fixture difficulty ratings for each team
            xgcp90_data (np.ndarray (m,)): xGCp90 values for each PL team
            defensive_points_data (np.ndarray (m,)): defensive points p/match for each PL team
            clean_sheets_per_match (np.ndarray (m,)): clean sheets p/match for each PL team
        """
        self.goalkeepers = watchlist[0]
        self.defenders = watchlist[1]
        self.midfielders = watchlist[2]
        self.forwards = watchlist[3]
        self.fdr_value_model = FDRValueModel(fixtures_df)
        self.clean_sheet_points_model_gk_def = CleanSheetPointsModel(xgcp90_data, defensive_points_data)
        self.clean_sheet_points_model_mid = CleanSheetPointsModel(xgcp90_data, clean_sheets_per_match)

    def get_asset_value_gk(self, gw: int, wc_gw: int, fh_gw: int) -> pd.DataFrame:
        """
        Calculates the asset value for each goalkeeper in the dataset by factoring
        in expected points, fixture difficulty rating (FDR) and their price. It uses
        a financial-like model to determine value relative to cost-effectiveness.

        Args:
            gw (int): The current game week number
            wc_gw (int): GW number where the user will use their Wildcard
            fh_gw (int): GW number where the user will use their Free Hit

        Returns:
            goalkeepers (pandas.DataFrame): A copy of the goalkeepers dataframe with
                updated asset values, expected points, and fixture difficulty ratings
        """
        xpts_list = []
        fdr_values = []
        asset_values = []

        for _, row in self.goalkeepers.iterrows():
            team = str(row["team"])
            price = row["price"]
            xgcp90 = row["xGCp90"]
            saves = row["saves p/match"]
            bps = row["BPS p/match"]
            yc = row["yellow cards p/match"]

            cs_pts = self.clean_sheet_points_model_gk_def.predict(xgcp90)
            xpts = 2 + cs_pts + saves / 3 + bps - yc

            fdr_value = self.get_fdr_value(gw, wc_gw, fh_gw, team)

            asset_value = (12 * xpts + 8 * fdr_value) / price ** 0.25
            xpts_list.append(xpts)
            fdr_values.append(fdr_value)
            asset_values.append(asset_value)

        AssetValueModel.update_df(self.goalkeepers, xpts_list, fdr_values, asset_values)

        return self.goalkeepers.copy()

    def get_asset_value_def(self, gw: int, wc_gw: int, fh_gw: int) -> pd.DataFrame:
        """
        Calculates the asset value for each defender in the dataset by factoring
        in expected points, fixture difficulty rating (FDR) and their price. It uses
        a financial-like model to determine value relative to cost-effectiveness.

        Args:
            gw (int): The current game week number
            wc_gw (int): GW number where the user will use their Wildcard
            fh_gw (int): GW number where the user will use their Free Hit

        Returns:
            defenders (pandas.DataFrame): A copy of the defenders dataframe with
                updated asset values, expected points, and fixture difficulty ratings
        """
        xpts_list = []
        fdr_values = []
        asset_values = []

        for _, row in self.defenders.iterrows():
            team = str(row["team"])
            price = row["price"]
            xgp90 = row["xGp90"]
            xap90 = row["xAp90"]
            xgcp90 = row["xGCp90"]
            bps = row["BPS p/match"]
            defcon = row["defensive contributions p/match"]
            yc = row["yellow cards p/match"]

            cs_pts = self.clean_sheet_points_model_gk_def.predict(xgcp90)
            if defcon < 5:
                xpts = 2 + xgp90 * 6 + xap90 * 3 + cs_pts + bps + defcon / 1000 - yc
            elif 5 <= defcon <= 13.33:
                xpts = 2 + xgp90 * 6 + xap90 * 3 + cs_pts + bps + ((((defcon - 5) ** 1.2) / 6.38) + 0.005) - yc
            else:
                xpts = 2 + xgp90 * 6 + xap90 * 3 + cs_pts + bps + 2 - yc

            fdr_value = self.get_fdr_value(gw, wc_gw, fh_gw, team)

            asset_value = (12 * xpts + 8 * fdr_value) / price ** 0.25
            xpts_list.append(xpts)
            fdr_values.append(fdr_value)
            asset_values.append(asset_value)

        AssetValueModel.update_df(self.defenders, xpts_list, fdr_values, asset_values)

        return self.defenders.copy()

    def get_asset_value_mid(self, gw: int, wc_gw: int, fh_gw: int) -> pd.DataFrame:
        """
        Calculates the asset value for each midfielder in the dataset by factoring
        in expected points, fixture difficulty rating (FDR) and their price. It uses
        a financial-like model to determine value relative to cost-effectiveness.

        Args:
            gw (int): The current game week number
            wc_gw (int): GW number where the user will use their Wildcard
            fh_gw (int): GW number where the user will use their Free Hit

        Returns:
            midfielders (pandas.DataFrame): A copy of the midfielders dataframe with
                updated asset values, expected points, and fixture difficulty ratings
        """
        xpts_list = []
        fdr_values = []
        asset_values = []

        for _, row in self.midfielders.iterrows():
            team = str(row["team"])
            price = row["price"]
            xgp90 = row["xGp90"]
            xap90 = row["xAp90"]
            xgcp90 = row["xGCp90"]
            defcon = row["defensive contributions p/match"]
            bps = row["BPS p/match"]
            yc = row["yellow cards p/match"]

            cs_pts = self.clean_sheet_points_model_mid.predict(xgcp90)
            if defcon < 6:
                xpts = 2 + xgp90 * 5 + xap90 * 3 + cs_pts + bps + defcon / 1000 - yc
            elif 6 <= defcon <= 16:
                xpts = 2 + xgp90 * 5 + xap90 * 3 + cs_pts + bps + ((((defcon - 6) ** 1.2) / 8) + 0.006) - yc
            else:
                xpts = 2 + xgp90 * 5 + xap90 * 3 + cs_pts + bps + 2 - yc

            fdr_value = self.get_fdr_value(gw, wc_gw, fh_gw, team)

            asset_value = (12 * xpts + 8 * fdr_value) / price ** 0.25
            xpts_list.append(xpts)
            fdr_values.append(fdr_value)
            asset_values.append(asset_value)

        AssetValueModel.update_df(self.midfielders, xpts_list, fdr_values, asset_values)

        return self.midfielders.copy()

    def get_asset_value_fwd(self, gw: int, wc_gw: int, fh_gw: int) -> pd.DataFrame:
        """
        Calculates the asset value for each forward in the dataset by factoring
        in expected points, fixture difficulty rating (FDR) and their price. It uses
        a financial-like model to determine value relative to cost-effectiveness.

        Args:
            gw (int): The current game week number
            wc_gw (int): GW number where the user will use their Wildcard
            fh_gw (int): GW number where the user will use their Free Hit

        Returns:
            forwards (pandas.DataFrame): A copy of the forwards dataframe with
                updated asset values, expected points, and fixture difficulty ratings
        """
        xpts_list = []
        fdr_values = []
        asset_values = []

        for _, row in self.forwards.iterrows():
            team = str(row["team"])
            price = row["price"]
            xgp90 = row["xGp90"]
            xap90 = row["xAp90"]
            defcon = row["defensive contributions p/match"]
            bps = row["BPS p/match"]
            yc = row["yellow cards p/match"]

            if defcon < 6:
                xpts = 2 + xgp90 * 4 + xap90 * 3 + bps + defcon / 1000 - yc
            elif 6 <= defcon <= 16:
                xpts = 2 + xgp90 * 4 + xap90 * 3 + bps + ((((defcon - 6) ** 1.2) / 8) + 0.006) - yc
            else:
                xpts = 2 + xgp90 * 4 + xap90 * 3 + bps + 2 - yc

            fdr_value = self.get_fdr_value(gw, wc_gw, fh_gw, team)

            asset_value = (12 * xpts + 8 * fdr_value) / price ** 0.25
            xpts_list.append(xpts)
            fdr_values.append(fdr_value)
            asset_values.append(asset_value)

        AssetValueModel.update_df(self.forwards, xpts_list, fdr_values, asset_values)

        return self.forwards.copy()

    def get_fdr_value(self, gw: int, wc_gw: int, fh_gw: int, team: str) -> float:
        """
        Retrieves the fixture difficulty rating (FDR) value based on the
        provided game week (gw), wildcard game week (wc_gw) and team name,
        using the associated FDR value model created during initialisation.

        Args:
            gw (int): The current game week number
            wc_gw (int): GW number where the user will use their Wildcard
            fh_gw (int): GW number where the user will use their Free Hit
            team (str): The name of the team

        Returns:
            fdr_value (float): The FDR value for the given team
        """
        self.fdr_value_model.get_fdr_value(gw, wc_gw, fh_gw)
        fdr_value = self.fdr_value_model.get_fdr_value_by_team(team)
        return fdr_value

    @staticmethod
    def update_df(df, xpts_list: list, fdr_values: list, asset_values: list) -> None:
        """
        Adds three new columns to the given dataframe with new data for player's
        expected points, fixture difficulty rating (FDR) and value; sorts it by the 'value'
        column in descending order and rounds numerical values to two decimal places.

        Args:
            df (pandas.DataFrame): The dataframe to be updated
            xpts_list (list): A list of expected points values
            fdr_values (list): A list of fixture difficulty rating (FDR) values
            asset_values (list): A list of asset values
        """
        asset_values = np.array(asset_values)
        df["xPts"] = xpts_list
        df["FDR value"] = fdr_values
        df["value"] = asset_values
        df.sort_values(by="value", ascending=False, inplace=True)
        df.round(2)