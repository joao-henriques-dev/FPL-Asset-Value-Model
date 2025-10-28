import math
import numpy as np
import pandas as pd

class FDRValueModel:
    """
    A deterministic model for determining FDR (Fixture Difficulty Rating) values for each
    English Premier League team based on custom algorithms and normalisation techniques.

    The model processes a dataframe containing all GWs of the season and computes
    FDR values based on custom algorithms and applies normalisation techniques.

    The class:
        - supports many distinct algorithms for determining FDR values based on user strategy;
        - provides sorted results ranked by FDR values in descending order;
        - allows querying of FDR values by specific teams;
        - offers various static normalisation methods for data preprocessing.

    Attributes:
        df (pandas.DataFrame): Input dataframe containing fixture difficulty ratings for each team
    """

    def __init__(self, fixtures_df):
        """
        Initialises the FDRValueModel object with the given dataframe.

        Args:
            fixtures_df (pandas.DataFrame): Input dataframe containing
                fixture difficulty ratings for each team
        """
        self.df = fixtures_df

    def get_fdr_value(self, gw: int, wc_gw: int, fh_gw: int) -> pd.DataFrame:
        """
        Calculates FDR values for each PL team based on the parameters that define
        the state of the game ('gw'), the GW where the user will use their wildcard
        ('wc_gw') and the GW where the user will use their free hit ('fh_gw').
        These parameters are key and directly affect the FDR values, but they had
        a layer of customisation that resonates with FPL's "Your own game!" motto.

        Args:
            gw (int): GW number
            wc_gw (int): GW number where the user will use their Wildcard
            fh_gw (int): GW number where the user will use their Free Hit
        """
        gws = [col for col in self.df.columns if col.startswith("gw")]
        fdr_results = []

        if wc_gw == 0:
            for _, row in self.df.iterrows():
                fdr_value = 0.0
                for i in range(gw, gw + 10):
                    if i == fh_gw:
                        continue
                    if i in range(gw, gw + 6):
                        fdr_value += row[gws[i-1]] * (math.log(-i + 8))
                    elif i in range(gw + 6, gw + 10):
                        fdr_value += row[gws[i-1]] * (-(i/25) + 0.5)
                fdr_results.append(fdr_value)

        else:
            for _, row in self.df.iterrows():
                fdr_value = 0.0
                for i in range(gw, wc_gw):
                    if i == fh_gw:
                        continue
                    fdr_value += row[gws[i-1]] * (math.log(-i + wc_gw + 1))
                fdr_results.append(fdr_value)

        fdr_results = FDRValueModel.zscore_normalisation(np.array(fdr_results))
        fdr_results = - (fdr_results / 2) + 5
        self.df["FDR value"] = fdr_results
        self.df = self.df.sort_values(by="FDR value", ascending=False)
        self.df = self.df.round(2)

        return self.df.copy()

    def get_fdr_value_by_team(self, team_name: str) -> float:
        """
        Retrieves the FDR value of a specific PL team.

        Args:
            team_name (str): Name of the team
        """
        value = self.df.loc[self.df["team"] == team_name, "FDR value"].values[0]
        return value

    @staticmethod
    def min_max_normalisation(array: np.ndarray) -> np.ndarray:
        """
        Computes min-max normalisation on the given array.

        Args:
            array (np.ndarray): Input data

        Returns:
            array_norm (np.ndarray): Min-max normalised data
        """
        array_min = np.min(array, axis=0)
        array_max = np.max(array, axis=0)
        array_norm = (array - array_min) / (array_max - array_min)
        return array_norm

    @staticmethod
    def mean_normalisation(array: np.ndarray) -> np.ndarray:
        """
        Computes Z-score normalisation on the given array.

        Args:
            array (np.ndarray): Input data

        Returns:
            array_norm (np.ndarray): Mean normalised data
        """
        mu = np.mean(array, axis=0)
        array_min = np.min(array, axis=0)
        array_max = np.max(array, axis=0)
        array_norm = (array - mu) / (array_max - array_min)
        return array_norm

    @staticmethod
    def zscore_normalisation(array: np.ndarray) -> np.ndarray:
        """
        Computes Z-score normalisation on the given array.

        Args:
          array (np.ndarray): Input data

        Returns:
          array_norm (np.ndarray): Z-score normalised data
        """
        mu = np.mean(array, axis=0)
        sigma = np.std(array, axis=0)
        array_norm = (array - mu) / sigma
        return array_norm